// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include "TestDeepCopy.hpp"

TEST(TEST_CATEGORY, deep_copy_narrowing_rank_0) {
#ifdef __ACPP__
  GTEST_SKIP() << "Fails at runtime with AdaptiveCpp (Metal backend)";
#endif
  test_deep_copy_assignable_types<float, double>();
}

TEST(TEST_CATEGORY, deep_copy_narrowing_rank_1) {
#ifdef __ACPP__
  GTEST_SKIP() << "Fails at runtime with AdaptiveCpp (Metal backend)";
#endif
  test_deep_copy_assignable_types<float*, double*>(1);
}

TEST(TEST_CATEGORY, deep_copy_narrowing_rank_2) {
#ifdef __ACPP__
  GTEST_SKIP() << "Fails at runtime with AdaptiveCpp (Metal backend)";
#endif
  test_deep_copy_assignable_types<float**, double**>(1, 1);
}

TEST(TEST_CATEGORY, deep_copy_narrowing_rank_3) {
#ifdef __ACPP__
  GTEST_SKIP() << "Fails at runtime with AdaptiveCpp (Metal backend)";
#endif
  test_deep_copy_assignable_types<float***, double***>(1, 1, 1);
}

TEST(TEST_CATEGORY, deep_copy_narrowing_rank_4) {
#ifdef __ACPP__
  GTEST_SKIP() << "Fails at runtime with AdaptiveCpp (Metal backend)";
#endif
  test_deep_copy_assignable_types<float****, double****>(1, 1, 1, 1);
}

TEST(TEST_CATEGORY, deep_copy_narrowing_rank_5) {
#ifdef __ACPP__
  GTEST_SKIP() << "Fails at runtime with AdaptiveCpp (Metal backend)";
#endif
  test_deep_copy_assignable_types<float*****, double*****>(1, 1, 1, 1, 1);
}

TEST(TEST_CATEGORY, deep_copy_narrowing_rank_6) {
#ifdef __ACPP__
  GTEST_SKIP() << "Fails at runtime with AdaptiveCpp (Metal backend)";
#endif
  test_deep_copy_assignable_types<float******, double******>(1, 1, 1, 1, 1, 1);
}

TEST(TEST_CATEGORY, deep_copy_narrowing_rank_7) {
#ifdef __ACPP__
  GTEST_SKIP() << "Fails at runtime with AdaptiveCpp (Metal backend)";
#endif
  test_deep_copy_assignable_types<float*******, double*******>(1, 1, 1, 1, 1, 1,
                                                               1);
}

TEST(TEST_CATEGORY, deep_copy_narrowing_rank_8) {
#ifdef __ACPP__
  GTEST_SKIP() << "Fails at runtime with AdaptiveCpp (Metal backend)";
#endif
  test_deep_copy_assignable_types<float********, double********>(1, 1, 1, 1, 1,
                                                                 1, 1, 1);
}
