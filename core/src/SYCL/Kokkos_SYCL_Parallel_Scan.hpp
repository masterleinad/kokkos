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

#ifndef KOKKO_SYCL_PARALLEL_SCAN_HPP
#define KOKKO_SYCL_PARALLEL_SCAN_HPP

#include <Kokkos_Macros.hpp>
#include <memory>
#if defined(KOKKOS_ENABLE_SYCL)

namespace Kokkos::Impl
{
	template <class ValueInit, class ValueOps, class Functor, class Policy, typename ValueType>
	struct FunctorWrapperRangePolicyParallelScanInitializeGlobal
	{
		using WorkTag = typename Policy::work_tag;

		void operator()(sycl::item<1> item) const {
        const typename Policy::index_type id =
            static_cast<typename Policy::index_type>(item.get_id()) + m_begin;
        ValueType update{};
        ValueInit::init(m_functor, &update);
        if constexpr (std::is_same<WorkTag, void>::value)
          m_functor(id, update, false);
        else
          m_functor(WorkTag(), id, update, false);
        ValueOps::copy(m_functor, &m_global_mem[id], &update);
      }

		#ifdef SYCL_DEVICE_COPYABLE
		 // We get ambiguous specialization if this class is trivially_copyable
          ~FunctorWrapperRangePolicyParallelScanInitializeGlobal() {}
#endif

typename Policy::index_type m_begin;
   Functor    m_functor;
    ValueType* m_global_mem;
		};


  template <class ValueOps, class Functor, class Policy, typename ValueType>
  struct FunctorWrapperRangePolicyParallelScanUpdateGlobalResults
  {
   using WorkTag = typename Policy::work_tag;

          void operator()(sycl::item<1> item) const {
        auto global_id = item.get_id();

        ValueType update = m_global_mem[global_id];
        if constexpr (std::is_same<WorkTag, void>::value)
          m_functor(global_id, update, true);
        else
          m_functor(WorkTag(), global_id, update, true);
        ValueOps::copy(m_functor, &m_global_mem[global_id], &update);
          }

	  #ifdef SYCL_DEVICE_COPYABLE
          // We get ambiguous specialization if this class is trivially_copyable
          ~FunctorWrapperRangePolicyParallelScanUpdateGlobalResults() {}
#endif

    Functor    m_functor;
    ValueType* m_global_mem;
  };
}

#ifdef SYCL_DEVICE_COPYABLE
template <class ValueInit, class ValueOps, class Functor, class Policy, typename ValueType>
struct sycl::is_device_copyable<
Kokkos::Impl::FunctorWrapperRangePolicyParallelScanInitializeGlobal<ValueInit, ValueOps, Functor, Policy, ValueType>> : std::true_type{};

template <class ValueOps, class Functor, class Policy, typename ValueType>
struct sycl::is_device_copyable<
Kokkos::Impl::FunctorWrapperRangePolicyParallelScanUpdateGlobalResults<ValueOps, Functor, Policy, ValueType>> : std::true_type{};
#endif

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelScanSYCLBase {
 public:
  using Policy = Kokkos::RangePolicy<Traits...>;

 protected:
  using Member       = typename Policy::member_type;
  using WorkTag      = typename Policy::work_tag;
  using WorkRange    = typename Policy::WorkRange;
  using LaunchBounds = typename Policy::launch_bounds;

  using ValueTraits = Kokkos::Impl::FunctorValueTraits<FunctorType, WorkTag>;
  using ValueInit   = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;
  using ValueJoin   = Kokkos::Impl::FunctorValueJoin<FunctorType, WorkTag>;
  using ValueOps    = Kokkos::Impl::FunctorValueOps<FunctorType, WorkTag>;

 public:
  using pointer_type   = typename ValueTraits::pointer_type;
  using value_type     = typename ValueTraits::value_type;
  using reference_type = typename ValueTraits::reference_type;
  using functor_type   = FunctorType;
  using size_type      = Kokkos::Experimental::SYCL::size_type;
  using index_type     = typename Policy::index_type;

 protected:
  const FunctorType m_functor;
  const Policy m_policy;
  pointer_type m_scratch_space = nullptr;

 private:
  template <typename Functor>
  void scan_internal(sycl::queue& q, const Functor& functor,
                     pointer_type global_mem, std::size_t size) const {
    // FIXME_SYCL optimize
    constexpr size_t wgroup_size = 32;
    auto n_wgroups               = (size + wgroup_size - 1) / wgroup_size;
    pointer_type group_results   = global_mem + size;

    auto local_scans = q.submit([&](sycl::handler& cgh) {
      sycl::accessor<value_type, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          local_mem(sycl::range<1>(wgroup_size), cgh);

      cgh.parallel_for(
          sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
          [=](sycl::nd_item<1> item) {
            const auto local_id  = item.get_local_linear_id();
            const auto global_id = item.get_global_linear_id();

            // Initialize local memory
            if (global_id < size)
              ValueOps::copy(functor, &local_mem[local_id],
                             &global_mem[global_id]);
            else
              ValueInit::init(functor, &local_mem[local_id]);
            item.barrier(sycl::access::fence_space::local_space);

            // Perform workgroup reduction
            for (size_t stride = 1; 2 * stride < wgroup_size + 1; stride *= 2) {
              auto idx = 2 * stride * (local_id + 1) - 1;
              if (idx < wgroup_size)
                ValueJoin::join(functor, &local_mem[idx],
                                &local_mem[idx - stride]);
              item.barrier(sycl::access::fence_space::local_space);
            }

            if (local_id == 0) {
              if (n_wgroups > 1)
                ValueOps::copy(functor,
                               &group_results[item.get_group_linear_id()],
                               &local_mem[wgroup_size - 1]);
              else
                ValueInit::init(functor,
                                &group_results[item.get_group_linear_id()]);
              ValueInit::init(functor, &local_mem[wgroup_size - 1]);
            }

            // Add results to all items
            for (size_t stride = wgroup_size / 2; stride > 0; stride /= 2) {
              auto idx = 2 * stride * (local_id + 1) - 1;
              if (idx < wgroup_size) {
                value_type dummy;
                ValueOps::copy(functor, &dummy, &local_mem[idx - stride]);
                ValueOps::copy(functor, &local_mem[idx - stride],
                               &local_mem[idx]);
                ValueJoin::join(functor, &local_mem[idx], &dummy);
              }
              item.barrier(sycl::access::fence_space::local_space);
            }

            // Write results to global memory
            if (global_id < size)
              ValueOps::copy(functor, &global_mem[global_id],
                             &local_mem[local_id]);
          });
    });
    // FIXME_SYCL remove guard once implemented for SYCL+CUDA
#ifdef KOKKOS_ARCH_INTEL_GEN
    q.submit_barrier(sycl::vector_class<sycl::event>{local_scans});
#else
    m_policy.space().fence();
#endif

    if (n_wgroups > 1) scan_internal(q, functor, group_results, n_wgroups);

    auto update_with_group_results = q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
                       [=](sycl::nd_item<1> item) {
                         const auto global_id = item.get_global_linear_id();
                         if (global_id < size)
                           ValueJoin::join(
                               functor, &global_mem[global_id],
                               &group_results[item.get_group_linear_id()]);
                       });
    });
    // FIXME_SYCL remove guard once implemented for SYCL+CUDA
#ifdef KOKKOS_ARCH_INTEL_GEN
    q.submit_barrier(
        sycl::vector_class<sycl::event>{update_with_group_results});
#else
    m_policy.space().fence();
#endif
  }

  template <typename Functor>
  sycl::event sycl_direct_launch(const Functor& functor) const {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = m_policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    sycl::queue& q = *instance.m_queue;

    const std::size_t len = m_policy.end() - m_policy.begin();

    // Initialize global memory
    auto initialize_global_memory = q.submit([&](sycl::handler& cgh) {
      auto global_mem = m_scratch_space;
      auto begin      = m_policy.begin();

      cgh.parallel_for(sycl::range<1>(len), 
                       FunctorWrapperRangePolicyParallelScanInitializeGlobal
		       <ValueInit, ValueOps, Functor, Policy, value_type>
		       {begin, functor, global_mem});
    });
    // FIXME_SYCL remove guard once implemented for SYCL+CUDA
#ifdef KOKKOS_ARCH_INTEL_GEN
    q.submit_barrier(sycl::vector_class<sycl::event>{initialize_global_memory});
#else
    space.fence();
#endif

    // Perform the actual exclusive scan
    scan_internal(q, functor, m_scratch_space, len);

    // Write results to global memory
    auto update_global_results = q.submit([&](sycl::handler& cgh) {
      auto global_mem = m_scratch_space;
      cgh.parallel_for(sycl::range<1>(len), 
		       FunctorWrapperRangePolicyParallelScanUpdateGlobalResults<
		         ValueOps, Functor, Policy, value_type> {functor, global_mem});
    });
// FIXME_SYCL remove guard once implemented for SYCL+CUDA
#ifdef KOKKOS_ARCH_INTEL_GEN
    q.submit_barrier(sycl::vector_class<sycl::event>{update_global_results});
#else
    space.fence();
#endif
    return update_global_results;
  }

 public:
  template <typename PostFunctor>
  void impl_execute(const PostFunctor& post_functor) {
    if (m_policy.begin() == m_policy.end()) return;

    auto& instance        = *m_policy.space().impl_internal_space_instance();
    const std::size_t len = m_policy.end() - m_policy.begin();

    // Compute the total amount of memory we will need. We emulate the recursive
    // structure that is used to do the actual scan. Essentially, we need to
    // allocate memory for the whole range and then recursively for the reduced
    // group results until only one group is left.
    std::size_t total_memory = 0;
    {
      size_t wgroup_size   = 32;
      size_t n_nested_size = len;
      size_t n_nested_wgroups;
      do {
        total_memory += sizeof(value_type) * n_nested_size;
        n_nested_wgroups = (n_nested_size + wgroup_size - 1) / wgroup_size;
        n_nested_size    = n_nested_wgroups;
      } while (n_nested_wgroups > 1);
      total_memory += sizeof(value_type) * wgroup_size;
    }

    // FIXME_SYCL consider only storing one value per block and recreate initial
    // results in the end before doing the final pass
    m_scratch_space =
        static_cast<pointer_type>(instance.scratch_space(total_memory));

    Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMem&
        indirectKernelMem = instance.m_indirectKernelMem;

    const auto functor_wrapper = Experimental::Impl::make_sycl_function_wrapper(
        m_functor, indirectKernelMem);

#ifdef SYCL_DEVICE_COPYABLE
    sycl_direct_launch(functor_wrapper.get_functor());
#else
    sycl::event event = sycl_direct_launch(functor_wrapper.get_functor());
    functor_wrapper.register_event(indirectKernelMem, event);
#endif
    post_functor();
  }

  ParallelScanSYCLBase(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

template <class FunctorType, class... Traits>
class ParallelScan<FunctorType, Kokkos::RangePolicy<Traits...>,
                   Kokkos::Experimental::SYCL>
    : private ParallelScanSYCLBase<FunctorType, Traits...> {
 public:
  using Base = ParallelScanSYCLBase<FunctorType, Traits...>;

  inline void execute() {
    Base::impl_execute([]() {});
  }

  ParallelScan(const FunctorType& arg_functor,
               const typename Base::Policy& arg_policy)
      : Base(arg_functor, arg_policy) {}
};

//----------------------------------------------------------------------------

template <class FunctorType, class ReturnType, class... Traits>
class ParallelScanWithTotal<FunctorType, Kokkos::RangePolicy<Traits...>,
                            ReturnType, Kokkos::Experimental::SYCL>
    : private ParallelScanSYCLBase<FunctorType, Traits...> {
 public:
  using Base = ParallelScanSYCLBase<FunctorType, Traits...>;

  ReturnType& m_returnvalue;

  inline void execute() {
    Base::impl_execute([&]() {
      const long long nwork = Base::m_policy.end() - Base::m_policy.begin();
      if (nwork > 0) {
        const int size = Base::ValueTraits::value_size(Base::m_functor);
        DeepCopy<HostSpace, Kokkos::Experimental::SYCLDeviceUSMSpace>(
            &m_returnvalue, Base::m_scratch_space + nwork - 1, size);
      }
    });
  }

  ParallelScanWithTotal(const FunctorType& arg_functor,
                        const typename Base::Policy& arg_policy,
                        ReturnType& arg_returnvalue)
      : Base(arg_functor, arg_policy), m_returnvalue(arg_returnvalue) {}
};

}  // namespace Impl
}  // namespace Kokkos

#endif

#endif
