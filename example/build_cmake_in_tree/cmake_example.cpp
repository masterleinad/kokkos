#include <Kokkos_Core.hpp>

int main() {
  Kokkos::ScopeGuard scope_guard;

  std::vector<double> vector(1);
  Kokkos::View<double, Kokkos::SYCLHostUSMSpace> host_view("host");
  Kokkos::View<double, Kokkos::SYCLSharedUSMSpace> shared_view("shared");
  Kokkos::View<double, Kokkos::SYCLDeviceUSMSpace> device_view("device");

  auto properties = Kokkos::view_wrap(shared_view.data());
  Kokkos::View<double, Kokkos::SYCLHostUSMSpace> host_view_unmanaged(properties);

}

