// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SYCL_ABORT_HPP
#define KOKKOS_SYCL_ABORT_HPP

#include <Kokkos_Printf.hpp>
#if defined(KOKKOS_ENABLE_SYCL)
#include <sycl/sycl.hpp>
#if defined(__ACPP__)
#include <hipSYCL/sycl/libkernel/sscp/builtins/assert.hpp>
#endif

namespace Kokkos {
namespace Impl {

inline void sycl_abort(char const* msg) {
#if defined(__ACPP__)
  // FIXME_ACPP Kokkos::printf output is dropped on AdaptiveCpp devices and
  // __assert_fail does not exist on macOS. Use AdaptiveCpp's device assert
  // builtin instead: on its host (OpenMP) device it prints the message and
  // calls std::abort; on Metal it compiles to MSL __builtin_trap(), which does
  // not stop the kernel (same as the "SYCL doesn't abort" semantics below).
  // __builtin_trap() itself must not be used: the Metal emitter prints it as
  // an invalid llvm.trap() call, breaking every kernel that contains it.
  __acpp_if_target_device(
      __acpp_sscp_assert_fail(msg, __FILE__, __LINE__, "Kokkos::abort");)
      __acpp_if_target_host(Kokkos::printf("Aborting with message %s.\n", msg);)
// FIXME_WINDOWS __assert_fail isn't defined
#elif defined(NDEBUG) || defined(_MSC_VER)
  Kokkos::printf("Aborting with message %s.\n", msg);
#else
  // Choosing "" here causes problems but a single whitespace character works.
  const char* empty = " ";
  __assert_fail(msg, empty, 0, empty);
#endif
}

}  // namespace Impl
}  // namespace Kokkos

#endif
#endif
