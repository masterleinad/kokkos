#include <Kokkos_Core.hpp>
#include <cmath>

int main() {
  Kokkos::ScopeGuard scope_guard;

  int N = 4;
  int R = 10;
  Kokkos::View<int*> a("PuppyWeights", N);

  for (int r = 0; r < R; r++)
    Kokkos::parallel_for(
        "PuppyOnCouch", N, KOKKOS_LAMBDA(int i) { a(i) = i * r; });
}
