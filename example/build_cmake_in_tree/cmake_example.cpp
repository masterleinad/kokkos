#include <Kokkos_Core.hpp>
#include<iostream>
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;
using Device = Kokkos::Device<ExecutionSpace,MemorySpace>;
using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using HostMemorySpace = HostExecutionSpace::memory_space;
using HostDevice = Kokkos::Device<HostExecutionSpace, HostMemorySpace>;
using Real = float;

template < class ... VS >
Real sumView2d(const Kokkos::View<Real**, VS...> m) {
  using ViewInput = std::decay_t<decltype(m)>;
  using D = typename ViewInput::device_type;
  using E = typename D::execution_space;
  Real s;
  Kokkos::parallel_reduce(
    Kokkos::MDRangePolicy<E, Kokkos::Rank<2>>(
      {0,0}, {m.extent(0), m.extent(1)}),
    KOKKOS_LAMBDA (const int i, const int j, Real& update) {
      update += m(i, j); 
    }, s);
  E{}.fence();
  return s;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv); {
  const int s = 45;
  const int stepSizes[2] = {1, 10000};
  Kokkos::View<Real**, Device> v("v", s, (s + 1) * stepSizes[1]);
  std::cout << "original range: " << s << " " << (s+1)*stepSizes[1] << std::endl;
  Kokkos::deep_copy(ExecutionSpace{}, v, 1);
  for(int iStep = 40; iStep < v.extent_int(0); ++iStep) {
    auto vSub = Kokkos::subview(v,
      std::make_pair(0, (iStep + 1) * stepSizes[0]),
      std::make_pair(0, (iStep + 2) * stepSizes[1]));
    std::cout << "subview range: " << (iStep+1)*stepSizes[0] << " " << (iStep+2)*stepSizes[1] << std::endl;
    Kokkos::View<Real**, Device> vSubCopy("vSubCopy", vSub.extent(0), vSub.extent(1));
    Kokkos::deep_copy(ExecutionSpace{}, vSubCopy, vSub);
    ExecutionSpace{}.fence();
    auto vSubCopyHost = Kokkos::create_mirror_view_and_copy(HostExecutionSpace{}, vSubCopy);
    auto vSubCopyHostCopy = Kokkos::create_mirror_view_and_copy(Kokkos::CudaUVMSpace{}, vSubCopyHost);
    auto sum_vSub = sumView2d(vSub);
    auto sum_vSubCopy = sumView2d(vSubCopy);
    auto sum_vSubCopyHost = sumView2d(vSubCopyHost);
    auto sum_vSubCopyHostCopy = sumView2d(vSubCopyHostCopy);
    std::cout << iStep << ": " << sum_vSub << " " << sum_vSub - sum_vSubCopy << " " << sum_vSub - sum_vSubCopyHost << " " << sum_vSubCopyHostCopy - sum_vSubCopyHost << " " << sum_vSub - sum_vSubCopyHostCopy << "\n";
    std::cout << "range " << vSub.extent(0) << ' ' << vSub.extent(1) << std::endl;
    Kokkos::parallel_for(
    Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
      {0,0}, {vSub.extent(0), vSub.extent(1)}),
    KOKKOS_LAMBDA (const int i, const int j) {
      if (vSub(i,j) != vSubCopyHostCopy(i,j))
        KOKKOS_IMPL_DO_NOT_USE_PRINTF("%d,%d: %f %f\n", i, j, vSub(i,j), vSubCopyHostCopy(i,j));
    });
  }


  } Kokkos::finalize();

}



