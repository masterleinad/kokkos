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

struct Interface
{
    KOKKOS_DEFAULTED_FUNCTION
    virtual ~Interface() = default;
    KOKKOS_FUNCTION
    virtual void operator()( const size_t) const = 0;
};
struct Implementation final : public Interface
{
    KOKKOS_FUNCTION
    void operator()(const size_t i) const final
    { }
    void apply(){
        Kokkos::parallel_for("myLoop",10,
            KOKKOS_CLASS_LAMBDA (const size_t i) { this->Implementation::operator()(i); }
        );
    }
};
int main ()
{
  Kokkos::ScopeGuard scope_guard;
  auto implementationPtr = std::make_shared<Implementation>();
  implementationPtr->apply();
}
