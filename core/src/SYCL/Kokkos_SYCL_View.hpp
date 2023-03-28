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

#ifndef KOKKOS_EXPERIMENTAL_SYCL_VIEW_HPP
#define KOKKOS_EXPERIMENTAL_SYCL_VIEW_HPP

#include <impl/Kokkos_ViewMapping.hpp>

// Prior to oneAPI 2023.1.0, this gives
//
// InvalidModule: Invalid SPIR-V module: Casts from private/local/global address
// space are allowed only to generic
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20230100

namespace Kokkos {
namespace Impl {


// shared space -> global
// host space -> host
// device space -> device, private
// scratch space -> local, global

template <typename ValueType, typename MemorySpace>
struct SYCLUSMHandle;

template <typename ValueType>
struct SYCLUSMHandle<ValueType, Experimental::SYCLSharedUSMSpace>{
  sycl::global_ptr<ValueType, sycl::access::decorated::yes> m_ptr;

  template <typename iType>
  KOKKOS_FUNCTION ValueType& operator[](const iType& i) {
    return m_ptr[i];
  }

  KOKKOS_FUNCTION
  operator ValueType*() const { return m_ptr.get(); }

  KOKKOS_DEFAULTED_FUNCTION
  SYCLUSMHandle() = default;

  KOKKOS_FUNCTION
  explicit SYCLUSMHandle(ValueType* const arg_ptr) : m_ptr(arg_ptr) {
  }

     KOKKOS_FUNCTION
  SYCLUSMHandle(const SYCLUSMHandle& arg_handle, size_t offset) : m_ptr(arg_handle.m_ptr + offset) {}
};

template <typename ValueType>
struct SYCLUSMHandle<ValueType, Experimental::SYCLHostUSMSpace>{
  sycl::host_ptr<ValueType> m_ptr;

  template <typename iType>
  KOKKOS_FUNCTION ValueType& operator[](const iType& i) {
    return m_ptr[i];
  }

  KOKKOS_FUNCTION
  operator ValueType*() const { return m_ptr.get(); }

  KOKKOS_DEFAULTED_FUNCTION
  SYCLUSMHandle() = default;

  KOKKOS_FUNCTION
  explicit SYCLUSMHandle(ValueType* const arg_ptr) : m_ptr(arg_ptr) {
  }

   KOKKOS_FUNCTION
  SYCLUSMHandle(const SYCLUSMHandle& arg_handle, size_t offset) : m_ptr(arg_handle.m_ptr + offset) {}
};

template <typename ValueType>
struct SYCLUSMHandle<ValueType, Experimental::SYCLDeviceUSMSpace>{
  sycl::device_ptr<ValueType, sycl::access::decorated::yes> m_device_ptr;
  sycl::private_ptr<ValueType, sycl::access::decorated::yes> m_private_ptr;

  template <typename iType>
  KOKKOS_FUNCTION ValueType& operator[](const iType& i) {
    return m_device_ptr?m_device_ptr[i]:m_private_ptr[i];
  }

  KOKKOS_FUNCTION
  operator ValueType*() const { return m_device_ptr?static_cast<ValueType*>(m_device_ptr.get()): static_cast<ValueType*>(m_private_ptr.get()); }

  KOKKOS_DEFAULTED_FUNCTION
  SYCLUSMHandle() = default;

  KOKKOS_FUNCTION
  explicit SYCLUSMHandle(ValueType* const arg_ptr) :
  m_device_ptr (sycl::address_space_cast<sycl::access::address_space::ext_intel_global_device_space,  sycl::access::decorated::yes>(arg_ptr)) ,
  m_private_ptr(sycl::address_space_cast<sycl::access::address_space::private_space, sycl::access::decorated::yes>(arg_ptr))
  {
	  auto dummy = sycl::address_space_cast<sycl::access::address_space::ext_intel_global_device_space,  sycl::access::decorated::yes>(arg_ptr);
    static_assert(std::is_same_v<decltype(dummy), sycl::device_ptr<ValueType, sycl::access::decorated::yes>>);

  }

   KOKKOS_FUNCTION
  SYCLUSMHandle(const SYCLUSMHandle& arg_handle, size_t offset) : m_device_ptr(arg_handle.m_device_ptr?arg_handle.m_device_ptr + offset:nullptr),
                                                m_private_ptr(arg_handle.m_private_ptr?arg_handle.m_private_ptr + offset:nullptr){
        }
};

template <typename ValueType>
struct SYCLUSMHandle<ValueType, ScratchMemorySpace<Kokkos::Experimental::SYCL>>{
  sycl::device_ptr<ValueType, sycl::access::decorated::yes> m_device_ptr;
  sycl::local_ptr<ValueType, sycl::access::decorated::yes> m_local_ptr;

  template <typename iType>
  KOKKOS_FUNCTION ValueType& operator[](const iType& i) {
    return m_device_ptr?m_device_ptr[i]:m_local_ptr[i];
  }

  KOKKOS_FUNCTION
  operator ValueType*() const { return m_device_ptr?static_cast<ValueType*>(m_device_ptr.get()): static_cast<ValueType*>(m_local_ptr.get()); }

  KOKKOS_DEFAULTED_FUNCTION
  SYCLUSMHandle() = default;

  KOKKOS_FUNCTION
  explicit SYCLUSMHandle(ValueType* const arg_ptr) :
    m_device_ptr (sycl::address_space_cast<sycl::access::address_space::ext_intel_global_device_space,  sycl::access::decorated::yes>(arg_ptr)) ,
    m_local_ptr(sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(arg_ptr))
	{}

  KOKKOS_FUNCTION
  SYCLUSMHandle(const SYCLUSMHandle& arg_handle, size_t offset) : m_device_ptr(arg_handle.m_device_ptr?arg_handle.m_device_ptr + offset:nullptr), 
                                                m_local_ptr(arg_handle.m_local_ptr?arg_handle.m_device_ptr + offset:nullptr){
        }
};


template <class Traits>
struct ViewDataHandle<
    Traits,
    std::enable_if_t<
        std::is_void<typename Traits::specialize>::value &&
        !Traits::memory_traits::is_aligned &&
        !Traits::memory_traits::is_restrict &&
        !Traits::memory_traits::is_atomic &&
        (is_sycl_type_space<typename Traits::memory_space>::value ||
         std::is_same_v<typename Traits::memory_space,
                        ScratchMemorySpace<Kokkos::Experimental::SYCL>>)>> {
		//private:
  using value_type   = typename Traits::value_type;
  using memory_space = typename Traits::memory_space;
  using handle_type = SYCLUSMHandle<value_type,memory_space>;
  using return_type = typename Traits::value_type&;
  using track_type  = Kokkos::Impl::SharedAllocationTracker;

		public:
  KOKKOS_INLINE_FUNCTION
  static handle_type assign(value_type* const arg_data_ptr,
                            track_type const& /*arg_tracker*/) {
    return handle_type(arg_data_ptr);
  }

 KOKKOS_INLINE_FUNCTION
  static handle_type assign(value_type* const arg_data_ptr, size_t offset) {
    return handle_type(arg_data_ptr+offset);
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type assign(const handle_type arg_data_ptr, size_t offset) {
    return handle_type(arg_data_ptr, offset);
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // #if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >=
        // 20230100

#endif /* #ifndef KOKKOS_SYCL_VIEW_HPP */
