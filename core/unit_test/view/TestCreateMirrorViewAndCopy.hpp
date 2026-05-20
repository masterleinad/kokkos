// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <gtest/gtest.h>

namespace {

template <class MemorySpace>
void test_create_mirror_view_and_copy() {
  const int N = 10;
  Kokkos::View<int*, MemorySpace> device_view("device_view", N);
  Kokkos::deep_copy(device_view, 1);
  auto device_view_copy   = Kokkos::create_mirror_view_and_copy(device_view);
  using copy_memory_space = decltype(device_view_copy)::memory_space;
  static_assert(
      std::is_same_v<
          copy_memory_space,
          typename decltype(device_view)::host_mirror_type::memory_space>);
  if (std::is_same_v<copy_memory_space, MemorySpace>)
    ASSERT_EQ(device_view_copy, device_view);
  else
    ASSERT_NE(device_view_copy, device_view);
  for (int i = 0; i < N; ++i) ASSERT_EQ(device_view_copy(i), 1);
}

TEST(TEST_CATEGORY, view_create_mirror_view_and_copy) {
  test_create_mirror_view_and_copy<TEST_EXECSPACE::memory_space>();
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::DefaultExecutionSpace>) {
#if defined(KOKKOS_HAS_SHARED_SPACE)
    test_create_mirror_view_and_copy<Kokkos::SharedSpace>();
#endif
#if defined(KOKKOS_HAS_SHARED_HOST_PINNED_SPACE)
    test_create_mirror_view_and_copy<Kokkos::SharedHostPinnedSpace>();
#endif
  }
}
}  // namespace
