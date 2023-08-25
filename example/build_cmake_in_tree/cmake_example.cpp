#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int n = 3;
    Kokkos::View<int * [4]> view("view", n);
    Kokkos::Experimental::ScatterView<int * [4]> sview(view);

    for (int i = 0; i < 2; i++) {
      if (i == 1) {
        n--;
        sview.realloc(n);
        Kokkos::resize(view, n);
      }
      sview.reset();
      Kokkos::parallel_for(
          n, KOKKOS_LAMBDA(const int ii) {
            auto j = sview.access();
            for (int k = 0; k < n; k++) {
              j(k, 0) += 1;
              j(k, 1) += 1;
              j(k, 2) += 1;
              j(k, 3) += 1;
            }
          });
      Kokkos::Experimental::contribute(view, sview);

      // output
      printf("i = %d\n", i);
      Kokkos::parallel_for(
          n, KOKKOS_LAMBDA(const int i) {
            printf("view(%d, *) = (%d, %d, %d, %d)\n", i, view(i, 0),
                   view(i, 1), view(i, 2), view(i, 3));
          });
      // Kokkos::fence();
    }
  }
  Kokkos::finalize();
}
