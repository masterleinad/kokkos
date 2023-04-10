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

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace {

template <class ExecutionSpace>
struct CheckClassWithExecutionSpaceAsDataMemberIsCopyable {
  Kokkos::DefaultExecutionSpace device;
  Kokkos::DefaultHostExecutionSpace host;

  KOKKOS_FUNCTION void operator()(int, int& e) const {
    // not actually doing anything useful, mostly checking that
    // ExecutionSpace::in_parallel() is callable
    if (static_cast<int>(device.in_parallel()) < 0) {
      ++e;
    }
  }

  CheckClassWithExecutionSpaceAsDataMemberIsCopyable() {
    int errors;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(0, 1), *this,
                            errors);
    EXPECT_EQ(errors, 0);
  }
};

// FIXME_OPENMPTARGET nvlink error: Undefined reference to
// '_ZSt25__throw_bad_function_callv' in
// '/tmp/TestOpenMPTarget_ExecutionSpace-434d81.cubin'
#ifndef KOKKOS_ENABLE_OPENMPTARGET
TEST(TEST_CATEGORY, execution_space_as_class_data_member) {
  CheckClassWithExecutionSpaceAsDataMemberIsCopyable<TEST_EXECSPACE>();
}
#endif


TEST(TEST_CATEGORY, execution_space_status)
{
  TEST_EXECSPACE exec;
  ASSERT_EQ(exec.get_status(), Kokkos::Experimental::ExecutionSpaceStatus::complete);  
  const int N = 10000;
  Kokkos::View<int, typename TEST_EXECSPACE::memory_space> result_view("result_view");
  Kokkos::parallel_reduce(Kokkos::RangePolicy<TEST_EXECSPACE>(exec, 0, N), KOKKOS_LAMBDA(int i, int& update)
		  {
		    update += i;
		  }, result_view);
  while (exec.get_status() != Kokkos::Experimental::ExecutionSpaceStatus::complete){}
  int result;
  Kokkos::deep_copy(exec, result, result_view);
  exec.fence();
  ASSERT_EQ(exec.get_status(), Kokkos::Experimental::ExecutionSpaceStatus::complete);
  ASSERT_EQ(result, N/2*(N-1));
}

}  // namespace
