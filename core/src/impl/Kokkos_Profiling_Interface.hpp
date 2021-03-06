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

#ifndef KOKKOSP_INTERFACE_HPP
#define KOKKOSP_INTERFACE_HPP

#include <cinttypes>
#include <cstddef>
#include <string>

#include <iostream>
#include <cstdlib>

// NOTE: in this Kokkos::Profiling block, do not define anything that shouldn't
// exist should Profiling be disabled

namespace Kokkos {
namespace Profiling {
namespace Experimental {
enum struct DeviceType {
  Serial,
  OpenMP,
  Cuda,
  HIP,
  OpenMPTarget,
  HPX,
  Threads
};
template <typename ExecutionSpace>
struct DeviceTypeTraits;

constexpr const size_t device_type_bits = 8;
constexpr const size_t instance_bits    = 24;
template <typename ExecutionSpace>
inline uint32_t device_id(ExecutionSpace const& space) noexcept {
  auto device_id = static_cast<uint32_t>(DeviceTypeTraits<ExecutionSpace>::id);
  return (device_id << instance_bits) + space.impl_instance_id();
}
}  // namespace Experimental
}  // namespace Profiling
}  // end namespace Kokkos

#if defined(KOKKOS_ENABLE_PROFILING)
// We check at configure time that libdl is available.
#include <dlfcn.h>
#endif

#include <impl/Kokkos_Profiling_DeviceInfo.hpp>
#include <impl/Kokkos_Profiling_C_Interface.h>

namespace Kokkos {
namespace Profiling {

using SpaceHandle = Kokkos_Profiling_SpaceHandle;

}  // end namespace Profiling

namespace Profiling {

namespace Experimental {
using EventSet = Kokkos_Profiling_EventSet;
}  // namespace Experimental
using initFunction           = Kokkos_Profiling_initFunction;
using finalizeFunction       = Kokkos_Profiling_finalizeFunction;
using beginFunction          = Kokkos_Profiling_beginFunction;
using endFunction            = Kokkos_Profiling_endFunction;
using pushFunction           = Kokkos_Profiling_pushFunction;
using popFunction            = Kokkos_Profiling_popFunction;
using allocateDataFunction   = Kokkos_Profiling_allocateDataFunction;
using deallocateDataFunction = Kokkos_Profiling_deallocateDataFunction;
using createProfileSectionFunction =
    Kokkos_Profiling_createProfileSectionFunction;
using startProfileSectionFunction =
    Kokkos_Profiling_startProfileSectionFunction;
using stopProfileSectionFunction = Kokkos_Profiling_stopProfileSectionFunction;
using destroyProfileSectionFunction =
    Kokkos_Profiling_destroyProfileSectionFunction;
using profileEventFunction  = Kokkos_Profiling_profileEventFunction;
using beginDeepCopyFunction = Kokkos_Profiling_beginDeepCopyFunction;
using endDeepCopyFunction   = Kokkos_Profiling_endDeepCopyFunction;

}  // end namespace Profiling

}  // namespace Kokkos

#endif
