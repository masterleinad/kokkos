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

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#ifndef KOKKOS_COMPILER_NVHPC  // FIXME_NVHPC
template <class View, class ExecutionSpace>
struct TestViewMemoryAccessViolation {
  View v;
  static constexpr auto rank = View::rank;

  template <std::size_t... Is>
  KOKKOS_FUNCTION auto bad_access(std::index_sequence<Is...>) const {
    return 0;//v(Is...);
  }

  KOKKOS_FUNCTION void operator()(int) const {
	  v(0);
//    bad_access(std::make_index_sequence<rank>{});
  }

  TestViewMemoryAccessViolation(View w, ExecutionSpace const& s,
                                std::string const& matcher)
      : v(std::move(w)) {
   // EXPECT_DEATH(
        {
          Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(s, 0, 1),
                               *this);
          Kokkos::fence();
        }/*,
        matcher)*/;
  }
};

template <class View, class ExecutionSpace>
void test_view_memory_access_violation(View v, ExecutionSpace const& s,
                                       std::string const& m) {
  TestViewMemoryAccessViolation<View, ExecutionSpace>(std::move(v), s, m);
}

template <class View, class LblOrPtr, std::size_t... Is>
auto make_view_impl(LblOrPtr x, std::index_sequence<Is...>) {
  return View(x, (Is + 1)...);
}

template <class View, class LblOrPtr>
auto make_view(LblOrPtr x) {
  return make_view_impl<View>(std::move(x),
                              std::make_index_sequence<View::rank>());
}

template <class ExecutionSpace>
void test_view_memory_access_violations_from_host() {
  Kokkos::DefaultHostExecutionSpace const host_exec_space{};
  // clang-format off
  using V0 = Kokkos::View<int,         ExecutionSpace>;
  using V1 = Kokkos::View<int*,        ExecutionSpace>;
  using V2 = Kokkos::View<int**,       ExecutionSpace>;
  using V3 = Kokkos::View<int***,      ExecutionSpace>;
  using V4 = Kokkos::View<int****,     ExecutionSpace>;
  using V5 = Kokkos::View<int*****,    ExecutionSpace>;
  using V6 = Kokkos::View<int******,   ExecutionSpace>;
  using V7 = Kokkos::View<int*******,  ExecutionSpace>;
  using V8 = Kokkos::View<int********, ExecutionSpace>;
  std::string const prefix = "Kokkos::View ERROR: attempt to access inaccessible memory space";
  std::string const lbl = "my_label";
  //test_view_memory_access_violation(make_view<V0>(lbl), host_exec_space, prefix + ".*" + lbl);
  test_view_memory_access_violation(make_view<V1>(lbl), host_exec_space, prefix + ".*" + lbl);
  // clang-format on
}

TEST(TEST_CATEGORY_DEATH, view_memory_access_violations_from_host) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  using ExecutionSpace = TEST_EXECSPACE;

  if (Kokkos::SpaceAccessibility<
          /*AccessSpace=*/Kokkos::HostSpace,
          /*MemorySpace=*/typename ExecutionSpace::memory_space>::accessible) {
    GTEST_SKIP() << "skipping since no memory access violation would occur";
  }

  test_view_memory_access_violations_from_host<ExecutionSpace>();
}
#endif
