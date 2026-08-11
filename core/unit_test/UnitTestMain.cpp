// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

int main(int argc, char *argv[]) {
  // We want to use "threadsafe" by default while the default in GTest on Linux
  // is "fast"
  setenv("GTEST_DEATH_TEST_STYLE", "threadsafe", /*overwrite=*/0);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
