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

#ifndef KOKKOS_SYCL_KERNELLAUNCH_HPP_
#define KOKKOS_SYCL_KERNELLAUNCH_HPP_

#include <Kokkos_SYCL.hpp>
#include <SYCL/Kokkos_SYCL_Instance.hpp>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <typename Policy, typename Functor>
void sycl_direct_launch(const Policy& policy, const Functor& functor) {
  // Convenience references
  const Kokkos::Experimental::SYCL& space = policy.space();
  Kokkos::Experimental::Impl::SYCLInternal& instance =
      *space.impl_internal_space_instance();
  cl::sycl::queue& q = *instance.m_queue;

  q.wait();

  auto end   = policy.end();
  auto begin = policy.begin();
  cl::sycl::range<1> range(end - begin);

  q.submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for(range, [=](cl::sycl::item<1> item) {
      const typename Policy::index_type id = item.get_linear_id();
      functor(id);
    });
  });

  q.wait();
}

// Indirectly launch a functor by explicitly creating it in USM shared memory
template <typename Policy, typename Functor>
void sycl_indirect_launch(const Policy& policy, const Functor& functor) {
  // Convenience references
  const Kokkos::Experimental::SYCL& space = policy.space();
  Kokkos::Experimental::Impl::SYCLInternal& instance =
      *space.impl_internal_space_instance();
  Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMemory& kernelMem =
      *instance.m_indirectKernel;

  // Allocate USM shared memory for the functor
  kernelMem.resize(std::max(kernelMem.size(), sizeof(Functor)));

  // Placement new a copy of functor into USM shared memory
  //
  // Store it in a unique_ptr to call its destructor on scope exit
  std::unique_ptr<Functor, Kokkos::Impl::destruct_delete> kernelFunctorPtr(
      new (kernelMem.data()) Functor(functor));

  // Use reference_wrapper (because it is both trivially copyable and invocable)
  // and launch it
  sycl_direct_launch(policy, std::reference_wrapper(*kernelFunctorPtr));
}

template <class Driver>
void sycl_launch(const Driver driver) {
  // if the functor is trivially copyable, we can launch it directly;
  // otherwise, we will launch it indirectly via explicitly creating
  // it in USM shared memory.
  if constexpr (std::is_trivially_copyable_v<decltype(driver.m_functor)>)
    sycl_direct_launch(driver.m_policy, driver.m_functor);
  else
    sycl_indirect_launch(driver.m_policy, driver.m_functor);
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_SYCL_KERNELLAUNCH_HPP_
