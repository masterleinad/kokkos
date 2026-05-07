// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

module;

#include <impl/Kokkos_UnorderedMap_impl.hpp>

export module kokkos.unordered_map_impl;

export {
  namespace Kokkos::Impl {
  using ::Kokkos::Impl::append_to_label;
  using ::Kokkos::Impl::find_hash_size;
  using ::Kokkos::Impl::UnorderedMapCanAssign;
  using ::Kokkos::Impl::UnorderedMapErase;
  using ::Kokkos::Impl::UnorderedMapHistogram;
  using ::Kokkos::Impl::UnorderedMapPrint;
  using ::Kokkos::Impl::UnorderedMapRehash;
  }  // namespace Kokkos::Impl
}
