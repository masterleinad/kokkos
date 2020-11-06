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
#if defined(KOKKOS_ENABLE_SYCL)

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
  size_type* m_scratch_flags = nullptr;
  size_type m_final          = false;
  int m_grid_x               = 0;

 private:

  void scan_internal(cl::sycl::queue& q, pointer_type global_mem, std::size_t size)
  {
    // FIXME_SYCL optimize
    constexpr size_t wgroup_size = 8;
    auto n_wgroups = (size + wgroup_size - 1) / wgroup_size;

    pointer_type group_results; 
      group_results = static_cast<pointer_type>(
        sycl::malloc(sizeof(value_type)*n_wgroups, q, sycl::usm::alloc::shared));

      static_assert(std::is_same<typename std::remove_reference<decltype(*group_results)>::type, value_type>::value, "");

        if (size <= 32)
       {
               std::cout << "before scan" << std::endl;
               for (unsigned int i=0; i<size; ++i)
                       std::cout << i << ": " << global_mem[i] << std::endl;
       }

       q.submit([&, *this] (cl::sycl::handler& cgh) {
          sycl::accessor <value_type, 1, sycl::access::mode::read_write, sycl::access::target::local>
                         local_mem(sycl::range<1>(wgroup_size), cgh);

          sycl::stream out(1024, 256, cgh);
          cgh.parallel_for<class reduction_kernel>(
               sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
               [=] (sycl::nd_item<1> item) {

            size_t local_id = item.get_local_linear_id();
            size_t global_id = item.get_global_linear_id();

            // Initialize local memory
            if (global_id < size)
               local_mem[local_id] = global_mem[global_id];
            else
               local_mem[local_id] = value_type{};
            item.barrier(sycl::access::fence_space::local_space);

            // Perform workgroup reduction
            for (size_t stride = 1; 2*stride < wgroup_size+1; stride *= 2) {
               auto idx = 2 * stride * (local_id+1)-1;
               if (idx < wgroup_size) {
                 local_mem[idx] = local_mem[idx] + local_mem[idx - stride];
               }
               item.barrier(sycl::access::fence_space::local_space);
            }

	    if (local_id == 0)
              group_results[item.get_group_linear_id()] = n_wgroups>1?local_mem[wgroup_size-1]:0;

	    // Store intermediate results in global memory
            /*if (global_id < size)
               global_mem[global_id] = local_mem[local_id];*/

	    if (local_id == 0)
              local_mem[wgroup_size-1] = 0;//group_results[item.get_group_linear_id()];


            // Add results to all items
            for (size_t stride = wgroup_size/2; stride > 0; stride /=2)
            {
               auto idx = 2*stride* (local_id+1)-1;
               if (idx < wgroup_size) {
                 auto dummy = local_mem[idx-stride];
                 local_mem[idx-stride] = local_mem[idx];
                 local_mem[idx] += dummy;
               }
               item.barrier(sycl::access::fence_space::local_space);
            }

	    // Write results to global memory
	    if (global_id < size)
              global_mem[global_id] = local_mem[local_id];
         });
       });

       if (n_wgroups>1)
          scan_internal(q, group_results, n_wgroups);
       q.wait();

  q.submit([&, *this] (sycl::handler& cgh) {
          cgh.parallel_for<class reduction_kernel>(
               sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
               [=] (sycl::nd_item<1> item) {
            const auto global_id = item.get_global_linear_id();
               if (global_id < size)
                global_mem[global_id] += group_results[item.get_group_linear_id()];
              });});
  q.wait();

       cl::sycl::free(group_results, q);
  }

  template <typename PolicyType, typename Functor>
  void sycl_direct_launch(const PolicyType& /*policy*/,
                          const Functor& functor) /*const*/ {
  // Convenience references
    const Kokkos::Experimental::SYCL& space = m_policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    cl::sycl::queue& q = *instance.m_queue;

    std::size_t len = m_policy.end()-m_policy.begin();
    std::cout << "length: " << len << std::endl;

    // FIXME_SYCL
    if (len==0)
      return;

    m_scratch_space = static_cast<pointer_type>(
        sycl::malloc(sizeof(value_type)*len, q, sycl::usm::alloc::shared));

    // Initialize global memory
    q.submit([&, *this] (sycl::handler& cgh) {
          auto global_mem = m_scratch_space;
          sycl::stream out(4096, 1024, cgh);
          cgh.parallel_for<class reduction_kernel>(
               sycl::range<1>(len),
               [=] (sycl::item<1> item) {

            auto global_id = item.get_id();

            typename FunctorType::value_type update = 0;
            functor(global_id, update, false);
            global_mem[global_id] = update;
	    });
	  });
    q.wait();

    // Perform the actual exlcusive scan
    scan_internal(q, m_scratch_space, len);

    // Write results to global memory
    q.submit([&, *this] (sycl::handler& cgh) {
		    auto global_mem = m_scratch_space;
          sycl::stream out(4096, 1024, cgh);
          cgh.parallel_for<class reduction_kernel>(
               sycl::range<1>(len),
               [=] (sycl::item<1> item) {

            auto global_id = item.get_id();

               typename FunctorType::value_type update = global_mem[global_id];
               functor(global_id, update, true);
	       global_mem[global_id] = update;
          });
       });
    q.wait();
//    cl::sycl::free(m_scratch_space, q);
//  std::abort();
  }

 template <typename Functor>
  void sycl_indirect_launch(const Functor& functor) /*const*/ {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = m_policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMemory& kernelMem =
        *instance.m_indirectKernel;

    // Allocate USM shared memory for the functor
    kernelMem.resize(std::max(kernelMem.size(), sizeof(functor)));

    // Placement new a copy of functor into USM shared memory
    //
    // Store it in a unique_ptr to call its destructor on scope exit
    std::unique_ptr<Functor, Kokkos::Impl::destruct_delete>
        kernelFunctorPtr(new (kernelMem.data()) Functor(functor));

    auto kernelFunctor = std::reference_wrapper(*kernelFunctorPtr);
    sycl_direct_launch(m_policy, kernelFunctor);
  }

public:

  void impl_execute() {
    if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
      sycl_direct_launch(m_policy, m_functor);
    else
      sycl_indirect_launch(m_functor);
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

  inline void execute() { Base::impl_execute(); }

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
    Base::impl_execute();

    const long long nwork = Base::m_policy.end() - Base::m_policy.begin();
    if (nwork>0) {
      const int size = Base::ValueTraits::value_size(Base::m_functor);
      DeepCopy<HostSpace, Kokkos::Experimental::SYCLDeviceUSMSpace>(
          &m_returnvalue,
          Base::m_scratch_space + nwork-1,
          size);
      if (m_returnvalue != (nwork*(nwork+1))/2)
      {
	      std::cout << m_returnvalue << " " << nwork << std::endl;
        std::abort();
      }
    }


    // FIXME_SYCL
    //std::abort();
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
