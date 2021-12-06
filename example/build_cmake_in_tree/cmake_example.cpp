#include <Kokkos_Core.hpp>


class DummyDummy;


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

    double* x_=x.data();
    double* y_=y.data();
    double* z_=z.data();

    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      Kokkos::parallel_for("kk_axpby", N, KOKKOS_LAMBDA(int i) {z_[i]=x_[i]+y_[i];});
      if (fence_all) Kokkos::fence();
    }
    Kokkos::fence();
    double time = timer.seconds();
    return time;
  }

#ifdef KOKKOS_ENABLE_SYCL
  double sycl_axpby(int R) {
    auto sycl_queue = cl::sycl::queue(cl::sycl::gpu_selector());

    double* x_ = sycl::malloc_device<double>(N, sycl_queue);
    double* y_ = sycl::malloc_device<double>(N, sycl_queue);
    double* z_ = sycl::malloc_device<double>(N, sycl_queue);

    // Initialization
    sycl_queue.submit([&](cl::sycl::handler& cgh) {
      cgh.parallel_for<DummyDummy>(cl::sycl::range<1>(N), [=](cl::sycl::item<1> itemId) {
        const int i = itemId.get_id();
        z_[i] = x_[i] = y_[i] = 0;
      });
    });
    sycl_queue.wait();

    // Warmup
    sycl_queue.submit([&](cl::sycl::handler& cgh) {
      cgh.parallel_for(cl::sycl::range<1>(N), [=](cl::sycl::item<1> itemId) {
        const int i = itemId.get_id();
        z_[i]       = x_[i] + y_[i];
      });
    });
    sycl_queue.wait();

    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      sycl_queue.submit([&](cl::sycl::handler& cgh) {
        cgh.parallel_for(cl::sycl::range<1>(N),
                                        [=](cl::sycl::item<1> itemId) {
                                          const int i = itemId.get_id();
                                          z_[i]       = x_[i] + y_[i];
                                        });
      });
      if (fence_all) sycl_queue.wait();
    }
    sycl_queue.wait();
    double time = timer.seconds();
    return time;
  }
#endif

   void run_test(int R) {
    double bytes_moved = 1. * sizeof(double) * N * 3 * R;
    double GB          = bytes_moved / 1024 / 1024 / 1024;
    double time_kk     = kk_axpby(R);
    printf("AXPBY KK: %e s %e GB/s\n", time_kk, GB / time_kk);
#ifdef KOKKOS_ENABLE_SYCL
    double time_sycl = sycl_axpby(R);
    printf("AXPBY SYCL: %e s %e GB/s\n", time_sycl, GB / time_sycl);
#endif
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    unsigned int N = argc > 1 ? atoi(argv[1]) : 100000000;
    int R          = argc > 2 ? atoi(argv[2]) : 1000;
    for (unsigned int n = 524287; n <= 524289; ++n) {
      std::cout << "n = " << n << std::endl;
      AXPBY axpby(n, true);
      axpby.run_test(R);
    }
  }
  Kokkos::finalize();
}


