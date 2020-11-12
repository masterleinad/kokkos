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

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <sstream>
#include <iostream>

namespace Test {
struct SomeTag {};

template <class ExecutionSpace>
class TestRangePolicyConstruction {
 public:
  TestRangePolicyConstruction() {
    test_compile_time_parameters();
    test_runtime_parameters();
  }

 private:
  void test_compile_time_parameters() {
    {
      using policy_t        = Kokkos::RangePolicy<>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type,
                                typename execution_space::size_type>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Static>>::value));
      ASSERT_TRUE((std::is_same<work_tag, void>::value));
    }

    {
      using policy_t        = Kokkos::RangePolicy<ExecutionSpace>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type,
                                typename execution_space::size_type>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Static>>::value));
      ASSERT_TRUE((std::is_same<work_tag, void>::value));
    }

    {
      using policy_t        = Kokkos::RangePolicy<ExecutionSpace,
                                           Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type,
                                typename execution_space::size_type>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, void>::value));
    }

    {
      using policy_t =
          Kokkos::RangePolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>,
                              Kokkos::IndexType<long>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, void>::value));
    }

    {
      using policy_t =
          Kokkos::RangePolicy<Kokkos::IndexType<long>, ExecutionSpace,
                              Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, void>::value));
    }

    {
      using policy_t =
          Kokkos::RangePolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>,
                              Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t =
          Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, ExecutionSpace,
                              Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t =
          Kokkos::RangePolicy<SomeTag, Kokkos::Schedule<Kokkos::Dynamic>,
                              Kokkos::IndexType<long>, ExecutionSpace>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type,
                                typename execution_space::size_type>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, void>::value));
    }

    {
      using policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                           Kokkos::IndexType<long>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, void>::value));
    }

    {
      using policy_t        = Kokkos::RangePolicy<Kokkos::IndexType<long>,
                                           Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, void>::value));
    }

    {
      using policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                           Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                           Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t =
          Kokkos::RangePolicy<SomeTag, Kokkos::Schedule<Kokkos::Dynamic>,
                              Kokkos::IndexType<long>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }
  }
  void test_runtime_parameters() {
    using policy_t = Kokkos::RangePolicy<>;
    {
      policy_t p(5, 15);
      ASSERT_TRUE((p.begin() == 5));
      ASSERT_TRUE((p.end() == 15));
    }
    {
      policy_t p(Kokkos::DefaultExecutionSpace(), 5, 15);
      ASSERT_TRUE((p.begin() == 5));
      ASSERT_TRUE((p.end() == 15));
    }
    {
      policy_t p(5, 15, Kokkos::ChunkSize(10));
      ASSERT_TRUE((p.begin() == 5));
      ASSERT_TRUE((p.end() == 15));
      ASSERT_TRUE((p.chunk_size() == 10));
    }
    {
      policy_t p(Kokkos::DefaultExecutionSpace(), 5, 15, Kokkos::ChunkSize(10));
      ASSERT_TRUE((p.begin() == 5));
      ASSERT_TRUE((p.end() == 15));
      ASSERT_TRUE((p.chunk_size() == 10));
    }
    {
      policy_t p;
      ASSERT_TRUE((p.begin() == 0));
      ASSERT_TRUE((p.end() == 0));
      p = policy_t(5, 15, Kokkos::ChunkSize(10));
      ASSERT_TRUE((p.begin() == 5));
      ASSERT_TRUE((p.end() == 15));
      ASSERT_TRUE((p.chunk_size() == 10));
    }
  }
};

template <class ExecutionSpace>
class TestTeamPolicyConstruction {
 public:
  TestTeamPolicyConstruction() {
    test_compile_time_parameters();
    test_run_time_parameters();
  }

 private:
  void test_compile_time_parameters() {
    {
      using policy_t        = Kokkos::TeamPolicy<>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type,
                                typename execution_space::size_type>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Static>>::value));
      ASSERT_TRUE((std::is_same<work_tag, void>::value));
    }

    {
      using policy_t        = Kokkos::TeamPolicy<ExecutionSpace>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type,
                                typename execution_space::size_type>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Static>>::value));
      ASSERT_TRUE((std::is_same<work_tag, void>::value));
    }

    {
      using policy_t =
          Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type,
                                typename execution_space::size_type>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, void>::value));
    }

    {
      using policy_t =
          Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>,
                             Kokkos::IndexType<long>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, void>::value));
    }

    {
      using policy_t =
          Kokkos::TeamPolicy<Kokkos::IndexType<long>, ExecutionSpace,
                             Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, void>::value));
    }

    {
      using policy_t =
          Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>,
                             Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t =
          Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>, ExecutionSpace,
                             Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t =
          Kokkos::TeamPolicy<SomeTag, Kokkos::Schedule<Kokkos::Dynamic>,
                             Kokkos::IndexType<long>, ExecutionSpace>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type,
                                typename execution_space::size_type>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, void>::value));
    }

    {
      using policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                          Kokkos::IndexType<long>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, void>::value));
    }

    {
      using policy_t        = Kokkos::TeamPolicy<Kokkos::IndexType<long>,
                                          Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, void>::value));
    }

    {
      using policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                          Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                          Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t =
          Kokkos::TeamPolicy<SomeTag, Kokkos::Schedule<Kokkos::Dynamic>,
                             Kokkos::IndexType<long>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }
  }

  template <class policy_t>
  void test_run_time_parameters_type() {
    int league_size = 131;
    int team_size   = 4 < policy_t::execution_space::concurrency()
                        ? 4
                        : policy_t::execution_space::concurrency();
#ifdef KOKKOS_ENABLE_HPX
    team_size = 1;
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    if (std::is_same<typename policy_t::execution_space,
                     Kokkos::Experimental::OpenMPTarget>::value)
      team_size = 32;
#endif
    int chunk_size         = 4;
    int per_team_scratch   = 1024;
    int per_thread_scratch = 16;
    int scratch_size       = per_team_scratch + per_thread_scratch * team_size;

    policy_t p1(league_size, team_size);
    ASSERT_EQ(p1.league_size(), league_size);
    ASSERT_EQ(p1.team_size(), team_size);
    ASSERT_TRUE(p1.chunk_size() > 0);
    ASSERT_EQ(p1.scratch_size(0), 0);

    policy_t p2 = p1.set_chunk_size(chunk_size);
    ASSERT_EQ(p1.league_size(), league_size);
    ASSERT_EQ(p1.team_size(), team_size);
    ASSERT_EQ(p1.chunk_size(), chunk_size);
    ASSERT_EQ(p1.scratch_size(0), 0);

    ASSERT_EQ(p2.league_size(), league_size);
    ASSERT_EQ(p2.team_size(), team_size);
    ASSERT_EQ(p2.chunk_size(), chunk_size);
    ASSERT_EQ(p2.scratch_size(0), 0);

    policy_t p3 = p2.set_scratch_size(0, Kokkos::PerTeam(per_team_scratch));
    ASSERT_EQ(p2.league_size(), league_size);
    ASSERT_EQ(p2.team_size(), team_size);
    ASSERT_EQ(p2.chunk_size(), chunk_size);
    ASSERT_EQ(p2.scratch_size(0), per_team_scratch);
    ASSERT_EQ(p3.league_size(), league_size);
    ASSERT_EQ(p3.team_size(), team_size);
    ASSERT_EQ(p3.chunk_size(), chunk_size);
    ASSERT_EQ(p3.scratch_size(0), per_team_scratch);

    policy_t p4 = p2.set_scratch_size(0, Kokkos::PerThread(per_thread_scratch));
    ASSERT_EQ(p2.league_size(), league_size);
    ASSERT_EQ(p2.team_size(), team_size);
    ASSERT_EQ(p2.chunk_size(), chunk_size);
    ASSERT_EQ(p2.scratch_size(0), scratch_size);
    ASSERT_EQ(p4.league_size(), league_size);
    ASSERT_EQ(p4.team_size(), team_size);
    ASSERT_EQ(p4.chunk_size(), chunk_size);
    ASSERT_EQ(p4.scratch_size(0), scratch_size);

    policy_t p5 = p2.set_scratch_size(0, Kokkos::PerThread(per_thread_scratch),
                                      Kokkos::PerTeam(per_team_scratch));
    ASSERT_EQ(p2.league_size(), league_size);
    ASSERT_EQ(p2.team_size(), team_size);
    ASSERT_EQ(p2.chunk_size(), chunk_size);
    ASSERT_EQ(p2.scratch_size(0), scratch_size);
    ASSERT_EQ(p5.league_size(), league_size);
    ASSERT_EQ(p5.team_size(), team_size);
    ASSERT_EQ(p5.chunk_size(), chunk_size);
    ASSERT_EQ(p5.scratch_size(0), scratch_size);

    policy_t p6 = p2.set_scratch_size(0, Kokkos::PerTeam(per_team_scratch),
                                      Kokkos::PerThread(per_thread_scratch));
    ASSERT_EQ(p2.league_size(), league_size);
    ASSERT_EQ(p2.team_size(), team_size);
    ASSERT_EQ(p2.chunk_size(), chunk_size);
    ASSERT_EQ(p2.scratch_size(0), scratch_size);
    ASSERT_EQ(p6.league_size(), league_size);
    ASSERT_EQ(p6.team_size(), team_size);
    ASSERT_EQ(p6.chunk_size(), chunk_size);
    ASSERT_EQ(p6.scratch_size(0), scratch_size);

    policy_t p7 = p3.set_scratch_size(0, Kokkos::PerTeam(per_team_scratch),
                                      Kokkos::PerThread(per_thread_scratch));
    ASSERT_EQ(p3.league_size(), league_size);
    ASSERT_EQ(p3.team_size(), team_size);
    ASSERT_EQ(p3.chunk_size(), chunk_size);
    ASSERT_EQ(p3.scratch_size(0), scratch_size);
    ASSERT_EQ(p7.league_size(), league_size);
    ASSERT_EQ(p7.team_size(), team_size);
    ASSERT_EQ(p7.chunk_size(), chunk_size);
    ASSERT_EQ(p7.scratch_size(0), scratch_size);

    policy_t p8;  // default constructed
    ASSERT_EQ(p8.league_size(), 0);
    ASSERT_EQ(p8.scratch_size(0), 0);
    p8 = p3;  // call assignment operator
    ASSERT_EQ(p3.league_size(), league_size);
    ASSERT_EQ(p3.team_size(), team_size);
    ASSERT_EQ(p3.chunk_size(), chunk_size);
    ASSERT_EQ(p3.scratch_size(0), scratch_size);
    ASSERT_EQ(p8.league_size(), league_size);
    ASSERT_EQ(p8.team_size(), team_size);
    ASSERT_EQ(p8.chunk_size(), chunk_size);
    ASSERT_EQ(p8.scratch_size(0), scratch_size);
  }

  void test_run_time_parameters() {
    test_run_time_parameters_type<Kokkos::TeamPolicy<ExecutionSpace>>();
    test_run_time_parameters_type<
        Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>,
                           Kokkos::IndexType<long>>>();
    test_run_time_parameters_type<
        Kokkos::TeamPolicy<Kokkos::IndexType<long>, ExecutionSpace,
                           Kokkos::Schedule<Kokkos::Dynamic>>>();
    test_run_time_parameters_type<
        Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                           Kokkos::IndexType<long>, ExecutionSpace, SomeTag>>();
  }
};

// semiregular is copyable and default initializable
// (regular requires equality comparable)
template <class Policy>
void check_semiregular() {
  static_assert(std::is_default_constructible<Policy>::value, "");
  static_assert(std::is_copy_constructible<Policy>::value, "");
  static_assert(std::is_move_constructible<Policy>::value, "");
  static_assert(std::is_copy_assignable<Policy>::value, "");
  static_assert(std::is_move_assignable<Policy>::value, "");
  static_assert(std::is_destructible<Policy>::value, "");
}

TEST(TEST_CATEGORY, policy_construction) {
  check_semiregular<Kokkos::RangePolicy<TEST_EXECSPACE>>();
  check_semiregular<Kokkos::TeamPolicy<TEST_EXECSPACE>>();
  check_semiregular<Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>>();

  TestRangePolicyConstruction<TEST_EXECSPACE>();
  // FIXME_SYCL requires Team policy
#ifndef KOKKOS_ENABLE_SYCL
  TestTeamPolicyConstruction<TEST_EXECSPACE>();
#endif
}

template <template <class...> class Policy, class... Args>
void check_converting_constructor_add_work_tag(Policy<Args...> const& policy) {
  // Not the greatest but at least checking it compiles
  struct WorkTag {};
  Policy<Args..., WorkTag> policy_with_tag = policy;
  (void)policy_with_tag;
}

TEST(TEST_CATEGORY, policy_converting_constructor_from_other_policy) {
  check_converting_constructor_add_work_tag(
      Kokkos::RangePolicy<TEST_EXECSPACE>{});
  // FIXME_SYCL requires Team policy
#ifndef KOKKOS_ENABLE_SYCL
  check_converting_constructor_add_work_tag(
      Kokkos::TeamPolicy<TEST_EXECSPACE>{});
#endif
  check_converting_constructor_add_work_tag(
      Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{});
}

#ifndef KOKKOS_ENABLE_OPENMPTARGET  // FIXME_OPENMPTARGET
TEST(TEST_CATEGORY_DEATH, policy_bounds_unsafe_narrowing_conversions) {
  using Policy = Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                                       Kokkos::IndexType<unsigned>>;

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH(
      {
        (void)Policy({-1, 0}, {2, 3});
      },
      "unsafe narrowing conversion");
}
#endif

template <class Policy>
void test_prefer_desired_occupancy(Policy const& policy) {
  static_assert(!Policy::experimental_contains_desired_occupancy, "");

  // MaximizeOccupancy -> MaximizeOccupancy
  auto const policy_still_no_occ = Kokkos::Experimental::prefer(
      policy, Kokkos::Experimental::MaximizeOccupancy{});
  static_assert(
      !decltype(policy_still_no_occ)::experimental_contains_desired_occupancy,
      "");

  // MaximizeOccupancy -> DesiredOccupancy
  auto const policy_with_occ = Kokkos::Experimental::prefer(
      policy, Kokkos::Experimental::DesiredOccupancy{33});
  static_assert(
      decltype(policy_with_occ)::experimental_contains_desired_occupancy, "");
  EXPECT_EQ(policy_with_occ.impl_get_desired_occupancy().value(), 33);

  // DesiredOccupancy -> DesiredOccupancy
  auto const policy_change_occ = Kokkos::Experimental::prefer(
      policy_with_occ, Kokkos::Experimental::DesiredOccupancy{24});
  static_assert(
      decltype(policy_change_occ)::experimental_contains_desired_occupancy, "");
  EXPECT_EQ(policy_change_occ.impl_get_desired_occupancy().value(), 24);

  // DesiredOccupancy -> MaximizeOccupancy
  auto const policy_drop_occ = Kokkos::Experimental::prefer(
      policy_with_occ, Kokkos::Experimental::MaximizeOccupancy{});
  static_assert(
      !decltype(policy_drop_occ)::experimental_contains_desired_occupancy, "");
}

template <class... Args>
struct DummyPolicy : Kokkos::Impl::PolicyTraits<Args...> {
  using execution_policy = DummyPolicy;
  using traits           = Kokkos::Impl::PolicyTraits<Args...>;
  template <class... OtherArgs>
  DummyPolicy(DummyPolicy<OtherArgs...> const& p) : traits(p) {}
  DummyPolicy() = default;
};

TEST(TEST_CATEGORY, desired_occupancy_prefer) {
  test_prefer_desired_occupancy(DummyPolicy<TEST_EXECSPACE>{});
  test_prefer_desired_occupancy(Kokkos::RangePolicy<TEST_EXECSPACE>{});
  test_prefer_desired_occupancy(
      Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{});
  // FIXME_SYCL requires Team policy
#ifndef KOKKOS_ENABLE_SYCL
  test_prefer_desired_occupancy(Kokkos::TeamPolicy<TEST_EXECSPACE>{});
#endif
}

TEST(TEST_CATEGORY, desired_occupancy_empty_base_optimization) {
  DummyPolicy<TEST_EXECSPACE> const policy{};
  static_assert(sizeof(decltype(policy)) == 1, "");

  using Kokkos::Experimental::DesiredOccupancy;
  auto policy_with_occ =
      Kokkos::Experimental::prefer(policy, DesiredOccupancy{50});
  static_assert(sizeof(decltype(policy_with_occ)) == sizeof(DesiredOccupancy),
                "");
}

template <typename Policy>
void test_desired_occupancy_converting_constructors(Policy const& policy) {
  auto policy_with_occ = Kokkos::Experimental::prefer(
      policy, Kokkos::Experimental::DesiredOccupancy{50});
  EXPECT_EQ(policy_with_occ.impl_get_desired_occupancy().value(), 50);

  auto policy_with_hint = Kokkos::Experimental::require(
      policy_with_occ, Kokkos::Experimental::WorkItemProperty::HintLightWeight);
  EXPECT_EQ(policy_with_hint.impl_get_desired_occupancy().value(), 50);
}

TEST(TEST_CATEGORY, desired_occupancy_converting_constructors) {
  test_desired_occupancy_converting_constructors(
      Kokkos::RangePolicy<TEST_EXECSPACE>{});
  test_desired_occupancy_converting_constructors(
      Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{});
    // FIXME_SYCL requires Team policy
#ifndef KOKKOS_ENABLE_SYCL
  test_desired_occupancy_converting_constructors(
      Kokkos::TeamPolicy<TEST_EXECSPACE>{});
#endif
}

}  // namespace Test
