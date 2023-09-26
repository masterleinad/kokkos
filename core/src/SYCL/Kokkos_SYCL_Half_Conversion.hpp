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

#ifndef KOKKOS_SYCL_HALF_HPP_
#define KOKKOS_SYCL_HALF_HPP_

#ifdef KOKKOS_IMPL_SYCL_HALF_TYPE_DEFINED

#include <Kokkos_Half.hpp>
#include <Kokkos_ReductionIdentity.hpp>

namespace Kokkos {
namespace Experimental {

/************************** half conversions **********************************/
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(half_t val) { return val; }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(float val) { return half_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(double val) { return half_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(short val) { return half_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned short val) { return half_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(int val) { return half_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned int val) { return half_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(long long val) { return half_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned long long val) { return half_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(long val) { return half_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned long val) { return half_t::impl_type(val); }

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, float>::value, T>
cast_from_half(half_t val) {
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, double>::value, T>
cast_from_half(half_t val) {
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, short>::value, T>
cast_from_half(half_t val) {
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned short>::value, T>
    cast_from_half(half_t val) {
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, int>::value, T>
cast_from_half(half_t val) {
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, unsigned int>::value, T>
cast_from_half(half_t val) {
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long long>::value, T>
cast_from_half(half_t val) {
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long long>::value, T>
    cast_from_half(half_t val) {
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long>::value, T>
cast_from_half(half_t val) {
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long>::value, T>
    cast_from_half(half_t val) {
  return static_cast<T>(half_t::impl_type(val));
}
}  // namespace Experimental

template <>
struct reduction_identity<Kokkos::Experimental::half_t> {
  KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::Experimental::half_t
  sum() noexcept {
    return Kokkos::Experimental::half_t::impl_type(0.0F);
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::Experimental::half_t
  prod() noexcept {
    return Kokkos::Experimental::half_t::impl_type(1.0F);
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::Experimental::half_t
  max() noexcept {
    return std::numeric_limits<
        Kokkos::Experimental::half_t::impl_type>::lowest();
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::Experimental::half_t
  min() noexcept {
    return std::numeric_limits<Kokkos::Experimental::half_t::impl_type>::max();
  }
};

}  // namespace Kokkos
#endif  // KOKKOS_IMPL_SYCL_HALF_TYPE_DEFINED

#ifdef KOKKOS_IMPL_SYCL_BHALF_TYPE_DEFINED

namespace Kokkos {
namespace Experimental {

/************************** bhalf conversions *********************************/
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(bhalf_t val) { return val; }

KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(float val) { return bhalf_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(double val) { return bhalf_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(short val) { return bhalf_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(unsigned short val) { return bhalf_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(int val) { return bhalf_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(unsigned int val) { return bhalf_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(long long val) { return bhalf_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(unsigned long long val) {
  return bhalf_t::impl_type(val);
}
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(long val) { return bhalf_t::impl_type(val); }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(unsigned long val) { return bhalf_t::impl_type(val); }

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, float>::value, T>
cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, double>::value, T>
cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, short>::value, T>
cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned short>::value, T>
    cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, int>::value, T>
cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, unsigned int>::value, T>
cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long long>::value, T>
cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long long>::value, T>
    cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long>::value, T>
cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long>::value, T>
    cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(bhalf_t::impl_type(val));
}
}  // namespace Experimental

// sycl::bfloat16 doesn't have constexpr constructors so we return float
template <>
struct reduction_identity<Kokkos::Experimental::bhalf_t> {
  KOKKOS_FORCEINLINE_FUNCTION constexpr static float sum() noexcept {
    return 0.f;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static float prod() noexcept {
    return 1.0f;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static float max() noexcept {
    return -0x7f7f;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static float min() noexcept {
    return 0x7f7f;
  }
};

#define KOKKOS_SYCL_HALF_FUNCTION(op)                                 \
  KOKKOS_INLINE_FUNCTION Kokkos::Experimental::half_t op(Kokkos::Experimental::half_t x) {                  \
    return sycl::op(Kokkos::Experimental::half_t::impl_type(x));  \
  }

#define KOKKOS_SYCL_BHALF_FUNCTION(op)                                 \
  KOKKOS_INLINE_FUNCTION Kokkos::Experimental::bhalf_t op(Kokkos::Experimental::bhalf_t x) {                  \
    return sycl::ext::oneapi::experimental::op(Kokkos::Experimental::bhalf_t::impl_type(x));  \
  }

// FIXME_SYCL
#define KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(op)                                 \
  KOKKOS_INLINE_FUNCTION Kokkos::Experimental::bhalf_t op(Kokkos::Experimental::bhalf_t x) {                  \
    return Kokkos::op(static_cast<float>(x));  \
  }

KOKKOS_SYCL_HALF_FUNCTION(abs)
KOKKOS_SYCL_HALF_FUNCTION(asin)
	KOKKOS_SYCL_HALF_FUNCTION(asinh)
KOKKOS_SYCL_HALF_FUNCTION(acos)
	KOKKOS_SYCL_HALF_FUNCTION(acosh)
	KOKKOS_SYCL_HALF_FUNCTION(atan)
	        KOKKOS_SYCL_HALF_FUNCTION(atanh)
	                        KOKKOS_SYCL_HALF_FUNCTION(cbrt)
	KOKKOS_SYCL_HALF_FUNCTION(ceil)
KOKKOS_SYCL_HALF_FUNCTION(cos)
	KOKKOS_SYCL_HALF_FUNCTION(cosh)
	KOKKOS_SYCL_HALF_FUNCTION(exp)
KOKKOS_SYCL_HALF_FUNCTION(exp2)
KOKKOS_SYCL_HALF_FUNCTION(expm1)
KOKKOS_SYCL_HALF_FUNCTION(erf)
KOKKOS_SYCL_HALF_FUNCTION(erfc)
	KOKKOS_SYCL_HALF_FUNCTION(fabs)
	KOKKOS_SYCL_HALF_FUNCTION(floor)
KOKKOS_SYCL_HALF_FUNCTION(log)
	KOKKOS_SYCL_HALF_FUNCTION(logb)
	        KOKKOS_SYCL_HALF_FUNCTION(log1p)
	KOKKOS_SYCL_HALF_FUNCTION(log2)
KOKKOS_SYCL_HALF_FUNCTION(log10)
KOKKOS_SYCL_HALF_FUNCTION(lgamma)
	        KOKKOS_SYCL_HALF_FUNCTION(round)
KOKKOS_SYCL_HALF_FUNCTION(sin)
	KOKKOS_SYCL_HALF_FUNCTION(sinh)
	KOKKOS_SYCL_HALF_FUNCTION(sqrt)
KOKKOS_SYCL_HALF_FUNCTION(tan)
	KOKKOS_SYCL_HALF_FUNCTION(tanh)
	KOKKOS_SYCL_HALF_FUNCTION(trunc)
KOKKOS_SYCL_HALF_FUNCTION(tgamma)

	KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(abs)
KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(asin)
	        KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(asinh)
KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(acos)
	KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(acosh)
	KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(atan)
	        KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(atanh)
	                KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(cbrt)
	        KOKKOS_SYCL_BHALF_FUNCTION(ceil)
KOKKOS_SYCL_BHALF_FUNCTION(cos)
	KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(cosh)
KOKKOS_SYCL_BHALF_FUNCTION(exp)
KOKKOS_SYCL_BHALF_FUNCTION(exp2)
KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(expm1)
KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(erf)
KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(erfc)
	        KOKKOS_SYCL_BHALF_FUNCTION(fabs)
	        KOKKOS_SYCL_BHALF_FUNCTION(floor)
KOKKOS_SYCL_BHALF_FUNCTION(log)
	KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(logb)
	KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(log1p)
	KOKKOS_SYCL_BHALF_FUNCTION(log2)
KOKKOS_SYCL_BHALF_FUNCTION(log10)
KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(lgamma)
	KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(round)
KOKKOS_SYCL_BHALF_FUNCTION(sin)
	KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(sinh)
	KOKKOS_SYCL_BHALF_FUNCTION(sqrt)
KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(tan)
	KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(tanh)
	KOKKOS_SYCL_BHALF_FUNCTION(trunc)
KOKKOS_SYCL_BHALF_FUNCTION_FALLBACK(tgamma)

	// FIXME_SYCL
#define KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_MIXED_FALLBACK(FUNC, HALF_TYPE, MIXED_TYPE) \
  KOKKOS_INLINE_FUNCTION double FUNC(HALF_TYPE x, MIXED_TYPE y) {  \
    return Kokkos::FUNC(static_cast<double>(x), static_cast<double>(y)); \
  } \
  KOKKOS_INLINE_FUNCTION double FUNC(MIXED_TYPE x, HALF_TYPE y) {  \
    return Kokkos::FUNC(static_cast<double>(x), static_cast<double>(y)); \
  }

	// FIXME_SYCL
#define KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_FALLBACK(FUNC, HALF_TYPE)       \
  KOKKOS_INLINE_FUNCTION HALF_TYPE FUNC(HALF_TYPE x, HALF_TYPE y) {  \
    return static_cast<HALF_TYPE>(                                   \
        Kokkos::FUNC(static_cast<float>(x), static_cast<float>(y))); \
  } \
  KOKKOS_INLINE_FUNCTION float FUNC(float x, HALF_TYPE y) {  \
    return Kokkos::FUNC(static_cast<float>(x), static_cast<float>(y)); \
  } \
  KOKKOS_INLINE_FUNCTION float FUNC(HALF_TYPE x, float y) {  \
    return Kokkos::FUNC(static_cast<float>(x), static_cast<float>(y)); \
  } \
  KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_MIXED_FALLBACK(FUNC, HALF_TYPE, double) \
  KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_MIXED_FALLBACK(FUNC, HALF_TYPE, short) \
  KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_MIXED_FALLBACK(FUNC, HALF_TYPE, unsigned short) \
  KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_MIXED_FALLBACK(FUNC, HALF_TYPE, int) \
  KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_MIXED_FALLBACK(FUNC, HALF_TYPE, unsigned int) \
  KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_MIXED_FALLBACK(FUNC, HALF_TYPE, long) \
  KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_MIXED_FALLBACK(FUNC, HALF_TYPE, unsigned long) \
  KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_MIXED_FALLBACK(FUNC, HALF_TYPE, long long) \
  KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_MIXED_FALLBACK(FUNC, HALF_TYPE, unsigned long long)

KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_FALLBACK(pow, Kokkos::Experimental::half_t);
KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_FALLBACK(pow, Kokkos::Experimental::bhalf_t);
KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_FALLBACK(nextafter, Kokkos::Experimental::half_t);
KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_FALLBACK(nextafter, Kokkos::Experimental::bhalf_t);
KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_FALLBACK(hypot, Kokkos::Experimental::half_t);
KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_FALLBACK(hypot, Kokkos::Experimental::bhalf_t);
KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_FALLBACK(copysign, Kokkos::Experimental::half_t);
KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_FALLBACK(copysign, Kokkos::Experimental::bhalf_t);
KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_FALLBACK(fmod, Kokkos::Experimental::half_t);
KOKKOS_SYCL_MATH_BINARY_FUNCTION_HALF_FALLBACK(fmod, Kokkos::Experimental::bhalf_t);

// FIXME_SYCL
KOKKOS_INLINE_FUNCTION bool
isnan(Kokkos::Experimental::half_t x) {
  return Kokkos::isnan(static_cast<float>(Kokkos::Experimental::half_t::impl_type(x)));
}

KOKKOS_INLINE_FUNCTION bool
isnan(Kokkos::Experimental::bhalf_t x) {
  return sycl::ext::oneapi::experimental::isnan(Kokkos::Experimental::bhalf_t::impl_type(x));
}

KOKKOS_INLINE_FUNCTION Kokkos::Experimental::bhalf_t
fmin(Kokkos::Experimental::bhalf_t x, Kokkos::Experimental::bhalf_t y) {
  return sycl::ext::oneapi::experimental::fmin(Kokkos::Experimental::bhalf_t::impl_type(x), Kokkos::Experimental::bhalf_t::impl_type(y));
}

KOKKOS_INLINE_FUNCTION Kokkos::Experimental::bhalf_t
fmax(Kokkos::Experimental::bhalf_t x, Kokkos::Experimental::bhalf_t y) {
  return sycl::ext::oneapi::experimental::fmax(Kokkos::Experimental::bhalf_t::impl_type(x), Kokkos::Experimental::bhalf_t::impl_type(y));
}

KOKKOS_INLINE_FUNCTION Kokkos::Experimental::bhalf_t
fma(Kokkos::Experimental::bhalf_t x, Kokkos::Experimental::bhalf_t y, Kokkos::Experimental::bhalf_t z) {
  return sycl::ext::oneapi::experimental::fma(Kokkos::Experimental::bhalf_t::impl_type(x), Kokkos::Experimental::bhalf_t::impl_type(y), Kokkos::Experimental::bhalf_t::impl_type(z));
}

}  // namespace Kokkos
#endif  // KOKKOS_IMPL_SYCL_BHALF_TYPE_DEFINED

#endif
