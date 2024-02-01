//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_SYCL_PARALLEL_FOR_RANGE_HPP_
#define KOKKOS_SYCL_PARALLEL_FOR_RANGE_HPP_

#ifndef KOKKOS_IMPL_SYCL_USE_IN_ORDER_QUEUES
#include <vector>
#endif

namespace Kokkos::Impl {
template <typename FunctorWrapper, typename Policy>
struct FunctorWrapperRangePolicyParallelFor {
  using WorkTag = typename Policy::work_tag;

  void operator()(sycl::nd_item<1> item) const {
    const typename Policy::index_type id = item.get_global_linear_id();
    if (id < m_work_size) {
      const auto shifted_id = id + m_begin;
      if constexpr (std::is_void_v<WorkTag>)
        m_functor_wrapper.get_functor()(shifted_id);
      else
        m_functor_wrapper.get_functor()(WorkTag(), shifted_id);
    }
  }

  typename Policy::index_type m_begin;
  FunctorWrapper m_functor_wrapper;
  typename Policy::index_type m_work_size;
};
}  // namespace Kokkos::Impl

template <class FunctorType, class... Traits>
class Kokkos::Impl::ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>,
                                Kokkos::Experimental::SYCL> {
 public:
  using Policy = Kokkos::RangePolicy<Traits...>;

 private:
  using Member  = typename Policy::member_type;
  using WorkTag = typename Policy::work_tag;

  const FunctorType m_functor;
  const Policy m_policy;

  template <typename Functor>
  static sycl::event sycl_direct_launch(const Policy& policy,
                                        const Functor& functor,
                                        const sycl::event& memcpy_event) {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = policy.space();
    sycl::queue& q                          = space.sycl_queue();

    desul::ensure_sycl_lock_arrays_on_device(q);

    auto parallel_for_event = q.submit([&](sycl::handler& cgh) {
#ifndef KOKKOS_IMPL_SYCL_USE_IN_ORDER_QUEUES
      cgh.depends_on(memcpy_event);
#else
      (void)memcpy_event;
#endif
      const auto actual_range = policy.end() - policy.begin();
      FunctorWrapperRangePolicyParallelFor<Functor, Policy> f{
          policy.begin(), functor, actual_range};
      auto wgroup_size = policy.chunk_size();
      if (wgroup_size <= 1) {
        static sycl::kernel kernel = [&] {
          sycl::kernel_id functor_kernel_id =
              sycl::get_kernel_id<decltype(f)>();
          auto kernel_bundle =
              sycl::get_kernel_bundle<sycl::bundle_state::executable>(
                  q.get_context(), std::vector{functor_kernel_id});
          return kernel_bundle.get_kernel(functor_kernel_id);
        }();
        wgroup_size = kernel.get_info<sycl::info::kernel_device_specific::
                                          preferred_work_group_size_multiple>(
            q.get_device());
      }
      // We need to make sure that the range the kernel is launched with is a
      // multiple of the workgroup size. Hence, we need to restrict the
      // execution of the functor in the kernel to the actual range.
      const auto launch_range =
          (actual_range + wgroup_size - 1) / wgroup_size * wgroup_size;
      sycl::nd_range<1> range(launch_range, wgroup_size);
      cgh.parallel_for<FunctorWrapperRangePolicyParallelFor<Functor, Policy>>(
          range, f);
    });
#ifndef KOKKOS_IMPL_SYCL_USE_IN_ORDER_QUEUES
    q.ext_oneapi_submit_barrier(std::vector<sycl::event>{parallel_for_event});
#endif

    return parallel_for_event;
  }

 public:
  using functor_type = FunctorType;

  void execute() const {
    if (m_policy.begin() == m_policy.end()) return;

    Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMem&
        indirectKernelMem = m_policy.space()
                                .impl_internal_space_instance()
                                ->get_indirect_kernel_mem();

    auto functor_wrapper = Experimental::Impl::make_sycl_function_wrapper(
        m_functor, indirectKernelMem);
    sycl::event event = sycl_direct_launch(m_policy, functor_wrapper,
                                           functor_wrapper.get_copy_event());
    functor_wrapper.register_event(event);
  }

  ParallelFor(const ParallelFor&) = delete;
  ParallelFor(ParallelFor&&)      = delete;
  ParallelFor& operator=(const ParallelFor&) = delete;
  ParallelFor& operator=(ParallelFor&&) = delete;
  ~ParallelFor()                        = default;

  ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

#endif  // KOKKOS_SYCL_PARALLEL_FOR_RANGE_HPP_
