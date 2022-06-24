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

#ifndef KOKKOS_EXEC_SPACE_MANAGER_HPP
#define KOKKOS_EXEC_SPACE_MANAGER_HPP

#include <impl/Kokkos_InitArguments.hpp>

#include <iosfwd>
#include <map>
#include <string>

namespace Kokkos {
namespace Impl {

struct ExecSpaceFooBase {
  virtual void initialize(InitArguments const&)                    = 0;
  virtual void finalize()                                          = 0;
  virtual void fence()                                             = 0;
  virtual void fence(std::string)                                  = 0;
  virtual void print_configuration(std::ostream& msg, bool detail) = 0;
  virtual ~ExecSpaceFooBase()                                      = default;
};

template <class ExecutionSpace>
struct ExecSpaceFooDerived : ExecSpaceFooBase {
  void initialize(InitArguments const& args) final {
    ExecutionSpace::impl_initialize(args);
  }
  void finalize() final { ExecutionSpace::impl_finalize(); }
  void fence() final { ExecutionSpace().fence(); }
  void fence(std::string label) final {
    ExecutionSpace().fence(std::move(label));
  }
  void print_configuration(std::ostream& msg, bool detail) final {
    ExecutionSpace().print_configuration(msg, detail);
  }
};

/* ExecSpaceManager - Responsible for initializing all the registered
 * backends. Backends are registered using the register_space_initializer()
 * function which should be called from a global context so that it is called
 * prior to initialize_spaces() which is called from Kokkos::initialize()
 */
class ExecSpaceManager {
  std::map<std::string, std::unique_ptr<ExecSpaceFooBase>>
      exec_space_factory_list;

 public:
  ExecSpaceManager() = default;

  void register_space_factory(std::string name,
                              std::unique_ptr<ExecSpaceFooBase> ptr);
  void initialize_spaces(const Kokkos::InitArguments& args);
  void finalize_spaces();
  void static_fence();
  void static_fence(const std::string&);
  void print_configuration(std::ostream& msg, const bool detail);
  static ExecSpaceManager& get_instance();
};

template <class ExecutionSpace>
int initialize_space_factory(std::string name) {
  auto space_ptr = std::make_unique<ExecSpaceFooDerived<ExecutionSpace>>();
  ExecSpaceManager::get_instance().register_space_factory(name,
                                                          std::move(space_ptr));
  return 1;
}

}  // namespace Impl
}  // namespace Kokkos

#endif
