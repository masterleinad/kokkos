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

#include <gtest/gtest.h>
#include <cstdlib>

#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_CUDA
#include <TestCuda_Category.hpp>
#endif
#ifdef KOKKOS_ENABLE_HIP
#include <TestHIP_Category.hpp>
#endif
#ifdef KOKKOS_ENABLE_SYCL
#include <TestSYCL_Category.hpp>
#endif
#ifdef KOKKOS_ENABLE_OPENMP
#include <TestOpenMP_Category.hpp>
#endif
#ifdef KOKKOS_ENABLE_THREADS
#include <TestThreads_Category.hpp>
#endif
#ifdef KOKKOS_ENABLE_HPX
#include <TestHPX_Category.hpp>
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
#include <TestOpenMPTarget_Category.hpp>
#endif
#ifndef TEST_EXECSPACE
#ifdef KOKKOS_ENABLE_SERIAL
#include <TestSerial_Category.hpp>
#endif
#endif
#include <TestReducers_d.hpp>

class MinimalistPrinter : public testing::EmptyTestEventListener {
  // Called before a test starts.
  void OnTestStart(const testing::TestInfo&) override {
    buffer.str(std::string());  // clears the buffer.
    sbuf = std::cout.rdbuf();
    std::cout.rdbuf(buffer.rdbuf());
  }

  // Called after a failed assertion or a SUCCESS().
  void OnTestPartResult(
      const testing::TestPartResult& test_part_result) override {
    switch (test_part_result.type()) {
      // If the test part succeeded, we don't need to do anything.
      case TestPartResult::kSuccess: break;
      default: std::cout.rdbuf(sbuf); std::cout << buffer.str() << std::endl;
    }
    buffer.str(std::string());  // clears the buffer.
    sbuf = std::cout.rdbuf();
    std::cout.rdbuf(buffer.rdbuf());
  }

  // Called after a test ends.
  void OnTestEnd(const testing::TestInfo&) override { std::cout.rdbuf(sbuf); }

  std::stringstream buffer;
  std::streambuf* sbuf;
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);

  ::testing::TestEventListeners& listeners =
      ::testing::UnitTest::GetInstance()->listeners();
  auto default_listener = listeners.Release(listeners.default_result_printer());
  listeners.Append(new MinimalistPrinter);
  listeners.Append(default_listener);

  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
