/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_SYCL_PARALLEL_RANGE_HPP_
#define KOKKOS_SYCL_PARALLEL_RANGE_HPP_

template <class FunctorType, class ExecPolicy>
class Kokkos::Impl::ParallelFor<FunctorType, ExecPolicy,
                                Kokkos::Experimental::SYCL> {
 public:
  using Policy = ExecPolicy;

 private:
  using Member       = typename Policy::member_type;
  using WorkTag      = typename Policy::work_tag;
  using LaunchBounds = typename Policy::launch_bounds;

  const FunctorType m_functor;
  const Policy m_policy;

 private:
  ParallelFor()        = delete;
  ParallelFor& operator=(const ParallelFor&) = delete;

  template <typename Functor>
  static void sycl_direct_launch(const Policy& policy, const Functor& functor) {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    cl::sycl::queue& q = *instance.m_queue;

    q.wait();

    q.submit([functor, policy](cl::sycl::handler& cgh) {
      cl::sycl::range<1> range(policy.end() - policy.begin());

      cgh.parallel_for(range, [=](cl::sycl::item<1> item) {
        const typename Policy::index_type id =
            static_cast<typename Policy::index_type>(item.get_linear_id()) +
            policy.begin();
        if constexpr (std::is_same<WorkTag, void>::value)
          functor(id);
        else
          functor(WorkTag(), id);
      });
    });

    q.wait();
  }

  // Indirectly launch a functor by explicitly creating it in USM shared memory
  void sycl_indirect_launch() const {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = m_policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMemory& kernelMem =
        *instance.m_indirectKernel;

    // Allocate USM shared memory for the functor
    kernelMem.resize(std::max(kernelMem.size(), sizeof(m_functor)));

    // Placement new a copy of functor into USM shared memory
    //
    // Store it in a unique_ptr to call its destructor on scope exit
    std::unique_ptr<FunctorType, Kokkos::Impl::destruct_delete>
        kernelFunctorPtr(new (kernelMem.data()) FunctorType(m_functor));

    // Use reference_wrapper (because it is both trivially copyable and
    // invocable) and launch it
    sycl_direct_launch(m_policy, std::reference_wrapper(*kernelFunctorPtr));
  }

 public:
  using functor_type = FunctorType;

  void execute() const {
    // if the functor is trivially copyable, we can launch it directly;
    // otherwise, we will launch it indirectly via explicitly creating
    // it in USM shared memory.
    if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
      sycl_direct_launch(m_policy, m_functor);
    else
      sycl_indirect_launch();
  }

  ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

// ParallelFor
template <class FunctorType, class... Traits>
class Kokkos::Impl::ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
                  Kokkos::Experimental::SYCL> {
 public:
  using Policy = Kokkos::MDRangePolicy<Traits...>;

 private:
  using array_index_type = typename Policy::array_index_type;
  using index_type       = typename Policy::index_type;
  using LaunchBounds     = typename Policy::launch_bounds;
  using WorkTag          = typename Policy::work_tag;

  const FunctorType m_functor;
  const Policy m_policy;

  ParallelFor()        = delete;
  ParallelFor& operator=(ParallelFor const&) = delete;

  template <typename Functor>
  static void sycl_direct_launch(const Policy& policy, const Functor& functor) {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    cl::sycl::queue& q = *instance.m_queue;

    q.wait();

    if (policy.m_num_tiles == 0) return;
    // FIXME_SYCL optimize
    const int maxblocks = 32;
    if constexpr (Policy::rank == 2) {
      cl::sycl::range<2> local_sizes(policy.m_tile[0], policy.m_tile[1]);
      cl::sycl::range<2> global_sizes(std::min<index_type>((policy.m_upper[0] - policy.m_lower[0] + local_sizes[0] - 1) /
                       local_sizes[0],
                   maxblocks),
          std::min<index_type>((policy.m_upper[1] - policy.m_lower[1] + local_sizes[1] - 1) /
                       local_sizes[1],
                   maxblocks));
      cl::sycl::nd_range<2> range(global_sizes, local_sizes);

    } else if (Policy::rank == 3) {
      cl::sycl::range<3> local_sizes(policy.m_tile[0], policy.m_tile[1], policy.m_tile[2]);
      cl::sycl::range<3> global_sizes(std::min<index_type>((policy.m_upper[0] - policy.m_lower[0] + local_sizes[0] - 1) /
                       local_sizes[0],
                   maxblocks),
          std::min<index_type>((policy.m_upper[1] - policy.m_lower[1] + local_sizes[1] - 1) /
                       local_sizes[1],
                   maxblocks),
          std::min<index_type>((policy.m_upper[2] - policy.m_lower[2] + local_sizes[2] - 1) /
                       local_sizes[2],
                   maxblocks));
      cl::sycl::nd_range<3> range(global_sizes, local_sizes);
    } else if (Policy::rank == 4) {
      // id0,id1 encoded within threadIdx.x; id2 to threadIdx.y; id3 to
      // threadIdx.z
      cl::sycl::range<3> local_sizes(policy.m_tile[0] * policy.m_tile[1],
                                     policy.m_tile[2], policy.m_tile[3]);
      cl::sycl::range<3> global_sizes(std::min<index_type>(policy.m_tile_end[0] *
                                         policy.m_tile_end[1],
                   maxblocks),
          std::min<index_type>((policy.m_upper[2] - policy.m_lower[2] + local_sizes[1] - 1) /
                       local_sizes[1],
                   maxblocks),
          std::min<index_type>((policy.m_upper[3] - policy.m_lower[3] + local_sizes[2] - 1) /
                       local_sizes[2],
                   maxblocks));
      cl::sycl::nd_range<3> range(global_sizes, local_sizes);
    } else if (Policy::rank == 5) {
      // id0,id1 encoded within threadIdx.x; id2,id3 to threadIdx.y; id4
      // to threadIdx.z
      cl::sycl::range<3> local_sizes(policy.m_tile[0] * policy.m_tile[1],
                       policy.m_tile[2] * policy.m_tile[3],
                       policy.m_tile[4]);
      cl::sycl::range<3> global_sizes(
          std::min<index_type>(policy.m_tile_end[0] * policy.m_tile_end[1], maxblocks),
          std::min<index_type>(policy.m_tile_end[2] * policy.m_tile_end[3], maxblocks),
          std::min<index_type>((policy.m_upper[4] - policy.m_lower[4] + local_sizes[2] - 1) /
                       local_sizes[2],
                   maxblocks));
    } else if (Policy::rank == 6) {
      // id0,id1 encoded within threadIdx.x; id2,id3 to threadIdx.y;
      // id4,id5 to threadIdx.z
      cl::sycl::range<3> local_sizes(policy.m_tile[0] * policy.m_tile[1],
                       policy.m_tile[2] * policy.m_tile[3],
                       policy.m_tile[4] * policy.m_tile[5]);
      cl::sycl::range<3> global_sizes(std::min<index_type>(policy.m_tile_end[0] *
                                                       policy.m_tile_end[1],
                               maxblocks),
                      std::min<index_type>(policy.m_tile_end[2] *
                                                       policy.m_tile_end[3],
                               maxblocks),
                      std::min<index_type>(policy.m_tile_end[4] *
                                                       policy.m_tile_end[5],
                               maxblocks));
    } else {
      Kokkos::abort("Kokkos::MDRange Error: Exceeded rank bounds with HIP\n");
    }

    /*q.submit([functor, policy](cl::sycl::handler& cgh) {
      cl::sycl::range<1> range(policy.end() - policy.begin());

      cgh.parallel_for(range, [=](cl::sycl::item<1> item) {
        const typename Policy::index_type id =
            static_cast<typename Policy::index_type>(item.get_linear_id()) +
            policy.begin();
        if constexpr (std::is_same<WorkTag, void>::value)
          functor(id);
        else
          functor(WorkTag(), id);
      });
    });*/

    q.wait();
  }

  // Indirectly launch a functor by explicitly creating it in USM shared memory
  void sycl_indirect_launch() const {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = m_policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMemory& kernelMem =
        *instance.m_indirectKernel;

    // Allocate USM shared memory for the functor
    kernelMem.resize(std::max(kernelMem.size(), sizeof(m_functor)));

    // Placement new a copy of functor into USM shared memory
    //
    // Store it in a unique_ptr to call its destructor on scope exit
    std::unique_ptr<FunctorType, Kokkos::Impl::destruct_delete>
        kernelFunctorPtr(new (kernelMem.data()) FunctorType(m_functor));

    // Use reference_wrapper (because it is both trivially copyable and
    // invocable) and launch it
    sycl_direct_launch(m_policy, std::reference_wrapper(*kernelFunctorPtr));
  }

 public:
  using functor_type = FunctorType;

  void execute() const {
    // if the functor is trivially copyable, we can launch it directly;
    // otherwise, we will launch it indirectly via explicitly creating
    // it in USM shared memory.
    if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
      sycl_direct_launch(m_policy, m_functor);
    else
      sycl_indirect_launch();
  }

  ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};


#endif  // KOKKOS_SYCL_PARALLEL_RANGE_HPP_
