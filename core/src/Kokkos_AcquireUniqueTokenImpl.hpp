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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_ACQUIRE_UNIQUE_TOKEN_IMPL_HPP
#define KOKKOS_ACQUIRE_UNIQUE_TOKEN_IMPL_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_UniqueToken.hpp>
namespace Kokkos {
namespace Experimental {

template <typename TeamPolicy>
KOKKOS_FUNCTION AcquireTeamUniqueToken<TeamPolicy>::AcquireTeamUniqueToken(
    AcquireTeamUniqueToken<TeamPolicy>::token_type t, team_member_type team)
    : my_token(t), my_team_acquired_val(team.team_scratch(0)), my_team(team) {
  Kokkos::single(Kokkos::PerTeam(my_team),
                 [&]() { my_team_acquired_val() = my_token.acquire(); });
  my_team.team_barrier();

  my_acquired_val = my_team_acquired_val();
}

template <typename TeamPolicy>
KOKKOS_FUNCTION AcquireTeamUniqueToken<TeamPolicy>::~AcquireTeamUniqueToken() {
  my_team.team_barrier();
  Kokkos::single(Kokkos::PerTeam(my_team),
                 [&]() { my_token.release(my_acquired_val); });
  my_team.team_barrier();
}

}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_UNIQUE_TOKEN_HPP
