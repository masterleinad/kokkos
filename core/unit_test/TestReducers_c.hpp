// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <TestReducers.hpp>

namespace Test {
TEST(TEST_CATEGORY, reducers_double) {
#ifdef __ACPP__
  GTEST_SKIP() << "Fails at runtime with AdaptiveCpp (Metal backend)";
#endif
  TestReducers<double, TEST_EXECSPACE>::execute_float();
}
}  // namespace Test
