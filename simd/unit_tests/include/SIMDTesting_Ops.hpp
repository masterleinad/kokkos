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

#ifndef KOKKOS_SIMD_TESTING_OPS_HPP
#define KOKKOS_SIMD_TESTING_OPS_HPP

#include <Kokkos_SIMD.hpp>

class plus {
 public:
  template <class T>
  auto on_host(T const& a, T const& b) const {
    return a + b;
  }
  template <class T>
  KOKKOS_INLINE_FUNCTION auto on_device(T const& a, T const& b) const {
    return a + b;
  }
};

class minus {
 public:
  template <class T>
  auto on_host(T const& a, T const& b) const {
    return a - b;
  }
  template <class T>
  KOKKOS_INLINE_FUNCTION auto on_device(T const& a, T const& b) const {
    return a - b;
  }
};

class multiplies {
 public:
  template <class T>
  auto on_host(T const& a, T const& b) const {
    return a * b;
  }
  template <class T>
  KOKKOS_INLINE_FUNCTION auto on_device(T const& a, T const& b) const {
    return a * b;
  }
};

class divides {
 public:
  template <class T>
  auto on_host(T const& a, T const& b) const {
    return a / b;
  }
  template <class T>
  KOKKOS_INLINE_FUNCTION auto on_device(T const& a, T const& b) const {
    return a / b;
  }
};

class absolutes {
  template <typename T>
  static KOKKOS_FUNCTION auto abs_impl(T const& x) {
    if constexpr (std::is_signed_v<T>) {
      return Kokkos::abs(x);
    }
    return x;
  }

 public:
  template <typename T>
  auto on_host(T const& a) const {
    return Kokkos::Experimental::abs(a);
  }
  template <typename T>
  auto on_host_serial(T const& a) const {
    return abs_impl(a);
  }
  template <typename T>
  KOKKOS_INLINE_FUNCTION auto on_device(T const& a) const {
    return Kokkos::Experimental::abs(a);
  }
  template <typename T>
  KOKKOS_INLINE_FUNCTION auto on_device_serial(T const& a) const {
    return abs_impl(a);
  }
};

class shift_right {
 public:
  template <typename T, typename U>
  auto on_host(T&& a, U&& b) const {
    return a >> b;
  }
  template <typename T, typename U>
  KOKKOS_INLINE_FUNCTION auto on_device(T&& a, U&& b) const {
    return a >> b;
  }
};

class shift_left {
 public:
  template <typename T, typename U>
  auto on_host(T&& a, U&& b) const {
    return a << b;
  }
  template <typename T, typename U>
  KOKKOS_INLINE_FUNCTION auto on_device(T&& a, U&& b) const {
    return a << b;
  }
};

class cbrt_op {
 public:
  template <typename T>
  auto on_host(T const& a) const {
#if defined(KOKKOS_COMPILER_INTEL) || defined(KOKKOS_COMPILER_INTEL_LLVM)
    using abi_type = typename T::abi_type;
    if constexpr (!std::is_same_v<abi_type,
                                  Kokkos::Experimental::simd_abi::scalar>)
      return Kokkos::Experimental::cbrt(a);
#endif
    T result(a);
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = Kokkos::cbrt(result[i]);
    }
    return result;
  }
  template <typename T>
  auto on_host_serial(T const& a) const {
    return Kokkos::cbrt(a);
  }
};

class exp_op {
 public:
  template <typename T>
  auto on_host(T const& a) const {
#if defined(KOKKOS_COMPILER_INTEL) || defined(KOKKOS_COMPILER_INTEL_LLVM)
    using abi_type = typename T::abi_type;
    if constexpr (!std::is_same_v<abi_type,
                                  Kokkos::Experimental::simd_abi::scalar>)
      return Kokkos::Experimental::exp(a);
#endif
    T result(a);
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = Kokkos::exp(result[i]);
    }
    return result;
  }
  template <typename T>
  auto on_host_serial(T const& a) const {
    return Kokkos::exp(a);
  }
};

class log_op {
 public:
  template <typename T>
  auto on_host(T const& a) const {
#if defined(KOKKOS_COMPILER_INTEL) || defined(KOKKOS_COMPILER_INTEL_LLVM)
    using abi_type = typename T::abi_type;
    if constexpr (!std::is_same_v<abi_type,
                                  Kokkos::Experimental::simd_abi::scalar>)
      return Kokkos::Experimental::log(a);
#endif
    T result(a);
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = Kokkos::log(result[i]);
    }
    return result;
  }
  template <typename T>
  auto on_host_serial(T const& a) const {
    return Kokkos::log(a);
  }
};

class hmin {
 public:
  template <typename T>
  auto on_host(T const& a) const {
    return Kokkos::Experimental::hmin(a);
  }
  template <typename T>
  auto on_host_serial(T const& a) const {
    using DataType = typename T::value_type::value_type;

    auto const& v = a.impl_get_value();
    auto const& m = a.impl_get_mask();
    auto result   = Kokkos::reduction_identity<DataType>::min();
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (m[i]) result = Kokkos::min(result, v[i]);
    }
    return result;
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION auto on_device(T const& a) const {
    return Kokkos::Experimental::hmin(a);
  }
  template <typename T>
  KOKKOS_INLINE_FUNCTION auto on_device_serial(T const& a) const {
    using DataType = typename T::value_type::value_type;

    auto const& v = a.impl_get_value();
    auto const& m = a.impl_get_mask();
    auto result   = Kokkos::reduction_identity<DataType>::min();
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (m[i]) result = Kokkos::min(result, v[i]);
    }
    return result;
  }
};

class hmax {
 public:
  template <typename T>
  auto on_host(T const& a) const {
    return Kokkos::Experimental::hmax(a);
  }
  template <typename T>
  auto on_host_serial(T const& a) const {
    using DataType = typename T::value_type::value_type;

    auto const& v = a.impl_get_value();
    auto const& m = a.impl_get_mask();
    auto result   = Kokkos::reduction_identity<DataType>::max();
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (m[i]) result = Kokkos::max(result, v[i]);
    }
    return result;
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION auto on_device(T const& a) const {
    return Kokkos::Experimental::hmax(a);
  }
  template <typename T>
  KOKKOS_INLINE_FUNCTION auto on_device_serial(T const& a) const {
    using DataType = typename T::value_type::value_type;

    auto const& v = a.impl_get_value();
    auto const& m = a.impl_get_mask();
    auto result   = Kokkos::reduction_identity<DataType>::max();
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (m[i]) result = Kokkos::max(result, v[i]);
    }
    return result;
  }
};

class reduce {
 public:
  template <typename T>
  auto on_host(T const& a) const {
    using DataType = typename T::value_type::value_type;
    return Kokkos::Experimental::reduce(a, DataType(0), std::plus<>());
  }
  template <typename T>
  auto on_host_serial(T const& a) const {
    using DataType = typename T::value_type::value_type;

    auto const& v = a.impl_get_value();
    auto const& m = a.impl_get_mask();
    auto result   = Kokkos::reduction_identity<DataType>::sum();
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (m[i]) result += v[i];
    }
    return result;
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION auto on_device(T const& a) const {
    using DataType = typename T::value_type::value_type;
    return Kokkos::Experimental::reduce(a, DataType(0), std::plus<>());
  }
  template <typename T>
  KOKKOS_INLINE_FUNCTION auto on_device_serial(T const& a) const {
    using DataType = typename T::value_type::value_type;

    auto const& v = a.impl_get_value();
    auto const& m = a.impl_get_mask();
    auto result   = Kokkos::reduction_identity<DataType>::sum();
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (m[i]) result += v[i];
    }
    return result;
  }
};

#endif
