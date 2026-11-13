// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <TestAtomicOperations.hpp>

namespace Test {
TEST(TEST_CATEGORY, atomic_operations_double) {
#ifdef __ACPP__
  GTEST_SKIP() << "Fails at runtime with AdaptiveCpp (Metal backend)";
#endif
  const int start = -5;
  const int end   = 11;
  for (int i = start; i < end; ++i) {
    for (int t = 0; t < 7; t++) {
      if (t == 5 && TestAtomicOperations::division_may_differ_by_one_ulp<
                        TEST_EXECSPACE>) {
        ASSERT_TRUE((TestAtomicOperations::AtomicOperationsTestNonIntegralType<
                     double, TEST_EXECSPACE, true>(i, end - i + start, t)));
      } else {
        ASSERT_TRUE((TestAtomicOperations::AtomicOperationsTestNonIntegralType<
                     double, TEST_EXECSPACE>(i, end - i + start, t)));
      }
    }
  }
}
}  // namespace Test
