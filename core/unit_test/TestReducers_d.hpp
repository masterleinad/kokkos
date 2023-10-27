//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>
#include <TestReducers.hpp>
#include <TestNonTrivialScalarTypes.hpp>

namespace Test {
TEST(TEST_CATEGORY, reducers_int8_t) {
  using ThisTestType = int8_t;
 
  std::cout << 1 << std::endl;
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(1);
}

}  // namespace Test
