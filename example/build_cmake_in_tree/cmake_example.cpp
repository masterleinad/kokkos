#include <Kokkos_Core.hpp>

struct AXPBY {
  using view_t = Kokkos::View<double*>;
  int N;
  view_t x, y, z;

  bool fence_all;
  AXPBY(int N_, bool fence_all_)
      : N(N_),
        x(view_t("X", N)),
        y(view_t("Y", N)),
        z(view_t("Z", N)),
        fence_all(fence_all_) {}

  KOKKOS_FUNCTION
  void operator()(int i) const { z(i) = x(i) + y(i); }

  double kk_axpby(int R) {
    // Warmup
    Kokkos::parallel_for("kk_axpby_wup", N, *this);
    Kokkos::fence();

    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      Kokkos::parallel_for("kk_axpby", N, *this);
      if (fence_all) Kokkos::fence();
    }
    Kokkos::fence();
    double time = timer.seconds();
    return time;
  }

   void run_test(int R) {
    double bytes_moved = 1. * sizeof(double) * N * 3 * R;
    double GB          = bytes_moved / 1024 / 1024 / 1024;
    double time_kk     = kk_axpby(R);
    printf("AXPBY KK: %e s %e GB/s\n", time_kk, GB / time_kk);
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
//    unsigned int N = argc > 1 ? atoi(argv[1]) : 100000000;
    int R          = argc > 2 ? atoi(argv[2]) : 1000;
    for (unsigned int n = 43691; n <= 43693; ++n) {
      std::cout << "n = " << n << std::endl;
      AXPBY axpby(n, true);
      axpby.run_test(R);
    }
  }
  Kokkos::finalize();
}


