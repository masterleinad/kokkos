// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <TestDynViewAPI.hpp>
namespace Test {
TEST(TEST_CATEGORY, dyn_rank_view_api_generic) {
#ifdef __ACPP__
  GTEST_SKIP() << "Fails at runtime with AdaptiveCpp (Metal backend)";
#endif
  TestDynViewAPI<double, TEST_EXECSPACE>::run_tests();
}
}  // namespace Test
