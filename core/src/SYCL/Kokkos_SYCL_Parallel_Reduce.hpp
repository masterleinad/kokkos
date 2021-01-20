/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_SYCL_PARALLEL_REDUCE_HPP
#define KOKKOS_SYCL_PARALLEL_REDUCE_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_SYCL)

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType,
                     Kokkos::Experimental::SYCL> {
 public:
  using Policy = Kokkos::RangePolicy<Traits...>;

 private:
  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;
  using execution_space = typename Analysis::execution_space;
  using value_type      = typename Analysis::value_type;
  using pointer_type    = typename Analysis::pointer_type;
  using reference_type  = typename Analysis::reference_type;

  using WorkTag = typename Policy::work_tag;

 public:
  // V - View
  template <typename V>
  ParallelReduce(
      const FunctorType& f, const Policy& p, const V& v,
      typename std::enable_if<Kokkos::is_view<V>::value, void*>::type = nullptr)
      : m_functor(f), m_policy(p), m_result_ptr(v.data()) {}

  ParallelReduce(const FunctorType& f, const Policy& p,
                 const ReducerType& reducer)
      : m_functor(f),
        m_policy(p),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {}

 private:
  template <typename T>
  struct HasJoin {
    template <typename U>
    static constexpr decltype(
        std::declval<U>().join(
            std::declval<
                typename FunctorValueTraits<T, WorkTag>::value_type&>(),
            std::declval<
                const typename FunctorValueTraits<T, WorkTag>::value_type&>()),
        bool())
    test_join(int) {
      return true;
    }

    template <typename U>
    static constexpr bool test_join(...) {
      return false;
    }

    static constexpr bool value = test_join<T>(int());
  };

  template <typename T>
  struct ExtendedReferenceWrapper : std::reference_wrapper<T> {
    using std::reference_wrapper<T>::reference_wrapper;

    using value_type = typename FunctorValueTraits<T, WorkTag>::value_type;

    template <typename Dummy = T>
    std::enable_if_t<std::is_same_v<Dummy, T> &&
                     ReduceFunctorHasInit<Dummy>::value>
    init(value_type& value) const {
      return this->get().init(value);
    }

    template <typename Dummy = T>
    std::enable_if_t<std::is_same_v<Dummy, T> && HasJoin<Dummy>::value> join(
        volatile value_type& dest, const volatile value_type& src) const {
      return this->get().join(dest, src);
    }

    template <typename Dummy = T>
    std::enable_if_t<std::is_same_v<Dummy, T> &&
                     ReduceFunctorHasFinal<Dummy>::value>
    final(value_type& value) const {
      return this->get().final(value);
    }
  };

  template <typename PolicyType, typename Functor, typename Reducer>
  void sycl_direct_launch(const PolicyType& policy, const Functor& functor,
                          const Reducer& reducer) const {
    static_assert(ReduceFunctorHasInit<Functor>::value ==
                  ReduceFunctorHasInit<FunctorType>::value);
    static_assert(ReduceFunctorHasFinal<Functor>::value ==
                  ReduceFunctorHasFinal<FunctorType>::value);
    static_assert(HasJoin<Functor>::value == HasJoin<FunctorType>::value);
    if constexpr (!std::is_same<Reducer, InvalidType>::value)
      static_assert(HasJoin<Reducer>::value == HasJoin<ReducerType>::value);

    using ReducerConditional =
        Kokkos::Impl::if_c<std::is_same<InvalidType, Reducer>::value, Functor,
                           Reducer>;
    using ReducerTypeFwd = typename ReducerConditional::type;
    using WorkTagFwd =
        std::conditional_t<std::is_same<InvalidType, Reducer>::value, WorkTag,
                           void>;
    using ValueInit =
        Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;
    using ValueJoin =
        Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, WorkTagFwd>;
    using ValueOps = Kokkos::Impl::FunctorValueOps<Functor, WorkTag>;

    auto selected_reducer = ReducerConditional::select(functor, reducer);

    // Convenience references
    const Kokkos::Experimental::SYCL& space = policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    sycl::queue& q = *instance.m_queue;

    // FIXME_SYCL optimize
    const size_t wgroup_size =
        space.impl_internal_space_instance()->m_maxThreadsPerSM;
    const std::size_t size = policy.end() - policy.begin();
    const auto init_size   = std::max<std::size_t>(
        ((size + 1) / 2 + wgroup_size - 1) / wgroup_size, 1);
    const auto value_count =
        FunctorValueTraits<ReducerTypeFwd, WorkTagFwd>::value_count(
            selected_reducer);
    const auto results_ptr =
        static_cast<pointer_type>(Experimental::SYCLSharedUSMSpace().allocate(
            "SYCL parallel_reduce result storage",
            sizeof(*m_result_ptr) * std::max(value_count, 1u) * init_size));
    const auto finished_work_groups_ptr =
        static_cast<std::size_t*>(Experimental::SYCLSharedUSMSpace().allocate(
            "SYCL finished_work_groups", sizeof(std::size_t)));
    *finished_work_groups_ptr = 0;

    // Initialize global memory
    if (size <= 1) {
      q.submit([&](sycl::handler& cgh) {
        cgh.single_task([=]() {
          reference_type update =
              ValueInit::init(selected_reducer, results_ptr);
          if (size == 1) {
            if constexpr (std::is_same<WorkTag, void>::value)
              functor(policy.begin(), update);
            else
              functor(WorkTag(), policy.begin(), update);
          }
          if constexpr (ReduceFunctorHasFinal<Functor>::value)
            FunctorFinal<Functor, WorkTag>::final(functor, results_ptr);
        });
      });
      space.fence();
    }

    const auto n_wgroups = ((size + 1) / 2 + wgroup_size - 1) / wgroup_size;
    q.submit([&](sycl::handler& cgh) {
      sycl::accessor<value_type, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          local_mem(sycl::range<1>(wgroup_size) * std::max(value_count, 1u),
                    cgh);

      cgh.parallel_for(
          sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
          [=](sycl::nd_item<1> item) {
            const auto local_id = item.get_local_linear_id();
            const auto global_id =
                wgroup_size * item.get_group_linear_id() * 2 + local_id;

            // Initialize local memory
            reference_type update = ValueInit::init(
                selected_reducer, &local_mem[local_id * value_count]);
            if (global_id < size) {
              const typename Policy::index_type id =
                  static_cast<typename Policy::index_type>(global_id) +
                  policy.begin();
              if constexpr (std::is_same<WorkTag, void>::value) {
                functor(id, update);
                if (global_id + wgroup_size < size)
                  functor(id + wgroup_size, update);
              } else {
                functor(WorkTag(), id, update);
                if (global_id + wgroup_size < size)
                  functor(WorkTag(), id + wgroup_size, update);
              }
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Perform workgroup reduction
            for (unsigned int stride = wgroup_size / 2; stride > 0;
                 stride >>= 1) {
              const auto idx = local_id;
              if (idx < stride) {
                ValueJoin::join(selected_reducer, &local_mem[idx * value_count],
                                &local_mem[(idx + stride) * value_count]);
              }
              item.barrier(sycl::access::fence_space::local_space);
            }

            if (local_id == 0) {
              ValueOps::copy(
                  functor,
                  &results_ptr[(item.get_group_linear_id()) * value_count],
                  &local_mem[0]);
            }

            item.barrier(sycl::access::fence_space::global_space);

            if (local_id == 0)
              if (atomic_fetch_add(finished_work_groups_ptr, 1) ==
                  n_wgroups - 1) {
                for (unsigned int i = 1; i < n_wgroups; ++i)
                  ValueJoin::join(selected_reducer, results_ptr,
                                  &results_ptr[i * value_count]);
                if constexpr (ReduceFunctorHasFinal<Functor>::value)
                  if (n_wgroups <= 1)
                    FunctorFinal<Functor, WorkTag>::final(
                        functor, &results_ptr[(item.get_group_linear_id()) *
                                              value_count]);
              }
          });
    });
    space.fence();

    if (m_result_ptr) {
      Kokkos::Impl::DeepCopy<Kokkos::Experimental::SYCLDeviceUSMSpace,
                             Kokkos::Experimental::SYCLDeviceUSMSpace>(
          space, m_result_ptr, results_ptr,
          sizeof(*m_result_ptr) * value_count);
      space.fence();
    }

    sycl::free(results_ptr, q);
  }

  template <typename Functor, typename Reducer>
  void sycl_indirect_launch(const Functor& functor,
                            const Reducer& reducer) const {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = m_policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMem& kernelMem =
        instance.m_indirectKernelMem;

    // Copy the functor into USM Shared Memory
    using KernelFunctorPtr = std::unique_ptr<
        Functor, Experimental::Impl::SYCLInternal::IndirectKernelMem::Deleter>;
    KernelFunctorPtr kernelFunctorPtr = kernelMem.copy_from(functor);
    auto kernelFunctor = ExtendedReferenceWrapper<Functor>(*kernelFunctorPtr);

    if constexpr (!std::is_same<Reducer, InvalidType>::value) {
      Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMem& reducerMem =
          instance.m_indirectReducerMem;

      // Copy the functor into USM Shared Memory
      std::unique_ptr<
          Reducer, Experimental::Impl::SYCLInternal::IndirectKernelMem::Deleter>
          kernelReducerPtr = reducerMem.copy_from(reducer);
      auto kernelReducer = ExtendedReferenceWrapper<Reducer>(*kernelReducerPtr);

      sycl_direct_launch(m_policy, kernelFunctor, kernelReducer);
    } else
      sycl_direct_launch(m_policy, kernelFunctor, reducer);
  }

 public:
  void execute() const {
    if constexpr (std::is_trivially_copyable_v<decltype(m_functor)> &&
                  std::is_trivially_copyable_v<decltype(m_reducer)>)
      sycl_direct_launch(m_policy, m_functor, m_reducer);
    else
      sycl_indirect_launch(m_functor, m_reducer);
  }

 private:
  FunctorType m_functor;
  Policy m_policy;
  ReducerType m_reducer;
  pointer_type m_result_ptr;
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif
#endif /* KOKKOS_SYCL_PARALLEL_REDUCE_HPP */
