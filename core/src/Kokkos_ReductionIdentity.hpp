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

#ifndef KOKKOS_REDUCTION_IDENTITY_HPP
#define KOKKOS_REDUCTION_IDENTITY_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_REDUCTION_IDENTITY
#endif

#include <Kokkos_Macros.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_NumericTraits.hpp>
#include <cfloat>
#include <climits>
#include <cmath>

namespace Kokkos {

template <class T>
struct reduction_identity; /*{
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T sum() { return T(); }  // 0
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T prod()  // 1
    { static_assert( false, "Missing specialization of
Kokkos::reduction_identity for custom prod reduction type"); return T(); }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T max()   // minimum value
    { static_assert( false, "Missing specialization of
Kokkos::reduction_identity for custom max reduction type"); return T(); }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T min()   // maximum value
    { static_assert( false, "Missing specialization of
Kokkos::reduction_identity for custom min reduction type"); return T(); }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T bor()   // 0, only for integer
type { static_assert( false, "Missing specialization of
Kokkos::reduction_identity for custom bor reduction type"); return T(); }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T band()  // !0, only for integer
type { static_assert( false, "Missing specialization of
Kokkos::reduction_identity for custom band reduction type"); return T(); }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T lor()   // 0, only for integer
type { static_assert( false, "Missing specialization of
Kokkos::reduction_identity for custom lor reduction type"); return T(); }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T land()  // !0, only for integer
type { static_assert( false, "Missing specialization of
Kokkos::reduction_identity for custom land reduction type"); return T(); }
};*/

template <Kokkos::Impl::IntegralType T>
struct reduction_identity<T> {
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T sum() {
    return static_cast<T>(0);
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T prod() {
    return static_cast<T>(1);
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T max() {
    return Kokkos::Experimental::finite_min_v<T>;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T min() {
    return Kokkos::Experimental::finite_max_v<T>;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T bor() {
    return static_cast<T>(0x0);
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T band() {
    return ~static_cast<T>(0x0);
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T lor() {
    return static_cast<T>(0);
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T land() {
    return static_cast<T>(1);
  }
};

template <Kokkos::Impl::FloatingPointType T>
struct reduction_identity<T> {
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T sum() {
    return static_cast<T>(0);
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T prod() {
    return static_cast<T>(1);
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T max() {
    using namespace Kokkos::Experimental;
#if __FINITE_MATH_ONLY__
    return finite_min_v<T>;
#else
    return -infinity_v<T>;
#endif
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T min() {
    using namespace Kokkos::Experimental;
#if __FINITE_MATH_ONLY__
    return finite_max_v<T>;
#else
    return infinity_v<T>;
#endif
  }
};

// No __host__ __device__ annotation because long double treated as double in
// device code.  May be revisited later if that is not true any more.
template <>
struct reduction_identity<long double> {
  constexpr static long double sum() { return static_cast<long double>(0.0); }
  constexpr static long double prod() { return static_cast<long double>(1.0); }

  constexpr static long double max() {
#if __FINITE_MATH_ONLY__
    return -LDBL_MAX;
#else
    return -HUGE_VALL;
#endif
  }
  constexpr static long double min() {
#if __FINITE_MATH_ONLY__
    return LDBL_MAX;
#else
    return HUGE_VALL;
#endif
  }
};

}  // namespace Kokkos

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_REDUCTION_IDENTITY
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_REDUCTION_IDENTITY
#endif
#endif
