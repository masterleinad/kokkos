// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_INITIALIZE_FINALIZE_HPP
#define KOKKOS_INITIALIZE_FINALIZE_HPP

#include <Kokkos_Macros.hpp>

#include <functional>
#include <iosfwd>
#include <string>
#include <optional>

namespace Kokkos {

class InitializationSettings {
#define KOKKOS_IMPL_DECLARE(TYPE, NAME)                                      \
 private:                                                                    \
  std::optional<TYPE> m_##NAME;                                              \
                                                                             \
 public:                                                                     \
  InitializationSettings& set_##NAME(TYPE NAME) {                            \
    m_##NAME = NAME;                                                         \
    return *this;                                                            \
  }                                                                          \
  bool has_##NAME() const noexcept { return static_cast<bool>(m_##NAME); }   \
  TYPE get_##NAME() const noexcept /* NOLINT(bugprone-exception-escape) */ { \
    return *m_##NAME; /* NOLINT(bugprone-unchecked-optional-access) */       \
  }                                                                          \
  static_assert(true, "no-op to require trailing semicolon")

 public:
  KOKKOS_IMPL_DECLARE(int, num_threads);
  KOKKOS_IMPL_DECLARE(int, device_id);
  KOKKOS_IMPL_DECLARE(std::string, map_device_id_by);
  KOKKOS_IMPL_DECLARE(bool, disable_warnings);
  KOKKOS_IMPL_DECLARE(bool, print_configuration);
  KOKKOS_IMPL_DECLARE(bool, tune_internals);
  KOKKOS_IMPL_DECLARE(bool, tools_help);
  KOKKOS_IMPL_DECLARE(std::string, tools_libs);
  KOKKOS_IMPL_DECLARE(std::string, tools_args);

#undef KOKKOS_IMPL_DECLARE
};

void initialize(int& argc, char* argv[]);

void initialize(
    InitializationSettings const& settings = InitializationSettings());

void finalize();

/**
 * \brief Push a user-defined function to be called in
 *   Kokkos::finalize, before any Kokkos state is finalized.
 *
 * \warning Only call this after Kokkos::initialize, but before
 *   Kokkos::finalize.
 *
 * This function is the Kokkos analog to std::atexit.  If you call
 * this with a function f, then your function will get called when
 * Kokkos::finalize is called.  Specifically, it will be called BEFORE
 * Kokkos does any finalization.  This means that all execution
 * spaces, memory spaces, etc. that were initialized will still be
 * initialized when your function is called.
 *
 * Just like std::atexit, if you call push_finalize_hook in sequence
 * with multiple functions (f, g, h), Kokkos::finalize will call them
 * in reverse order (h, g, f), as if popping a stack.  Furthermore,
 * just like std::atexit, if any of your functions throws but does not
 * catch an exception, Kokkos::finalize will call std::terminate.
 */
void push_finalize_hook(std::function<void()> f);

/** \brief Print "Bill of Materials" */
void print_configuration(std::ostream& os, bool verbose = false);

[[nodiscard]] bool is_initialized() noexcept;
[[nodiscard]] bool is_finalized() noexcept;

[[nodiscard]] int device_id() noexcept;
[[nodiscard]] int num_devices() noexcept;
[[nodiscard]] int num_threads() noexcept;

bool show_warnings() noexcept;
bool tune_internals() noexcept;

void fence(const std::string& name /*= "Kokkos::fence: Unnamed Global Fence"*/);
}  // namespace Kokkos

namespace Kokkos::Impl {

void pre_initialize(const InitializationSettings& settings);

void post_initialize(const InitializationSettings& settings);

void pre_finalize();

void post_finalize();

void declare_configuration_metadata(const std::string& category,
                                    const std::string& key,
                                    const std::string& value);

}  // namespace Kokkos::Impl

#endif
