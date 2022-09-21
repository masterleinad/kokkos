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

#ifndef KOKKOS_EXPERIMENTAL_SYCL_VIEW_HPP
#define KOKKOS_EXPERIMENTAL_SYCL_VIEW_HPP

#include <impl/Kokkos_ViewMapping.hpp>

namespace Kokkos {
namespace Impl {

template <class Traits>
struct ViewDataHandle<
    Traits,
    std::enable_if_t<std::is_void<typename Traits::specialize>::value &&
                      !Traits::memory_traits::is_aligned &&
                      !Traits::memory_traits::is_restrict &&
		      !Traits::memory_traits::is_atomic && (is_sycl_type_space<typename Traits::memory_space>::value ||
                            std::is_same_v<typename Traits::memory_space,
                                         ScratchMemorySpace<Kokkos::Experimental::SYCL>>)>> {
  using value_type  = typename Traits::value_type;
  using memory_space = typename Traits::memory_space;
  using handle_type = std::conditional_t<is_sycl_type_space<memory_space>::value, sycl::global_ptr<value_type>, std::conditional_t<std::is_same_v<memory_space, ScratchMemorySpace<Kokkos::Experimental::SYCL>>, sycl::device_ptr<value_type>, sycl::host_ptr<value_type>>>;
  using return_type = typename Traits::value_type&;
  using track_type  = Kokkos::Impl::SharedAllocationTracker;

  KOKKOS_INLINE_FUNCTION
  static value_type* assign(value_type* arg_data_ptr,
                            track_type const& /*arg_tracker*/) {
    return (value_type*)(arg_data_ptr);
  }

  KOKKOS_INLINE_FUNCTION
  static value_type* assign(handle_type const arg_data_ptr, size_t offset) {
    return (value_type*)(arg_data_ptr + offset);
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif /* #ifndef KOKKOS_SYCL_VIEW_HPP */
