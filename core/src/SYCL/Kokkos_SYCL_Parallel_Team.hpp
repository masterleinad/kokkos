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

#ifndef KOKKOS_SYCL_PARALLEL_TEAM_HPP
#define KOKKOS_SYCL_PARALLEL_TEAM_HPP

#include <Kokkos_Parallel.hpp>

#include <SYCL/Kokkos_SYCL_Parallel_Reduce.hpp>  // workgroup_reduction
#include <SYCL/Kokkos_SYCL_Team.hpp>

#include <vector>

namespace Kokkos {
namespace Impl {
template <typename... Properties>
class TeamPolicyInternal<Kokkos::Experimental::SYCL, Properties...>
    : public PolicyTraits<Properties...> {
 public:
  using execution_policy = TeamPolicyInternal;

  using traits = PolicyTraits<Properties...>;

  template <typename ExecSpace, typename... OtherProperties>
  friend class TeamPolicyInternal;

 private:
  typename traits::execution_space m_space;
  int m_league_size;
  int m_team_size;
  int m_vector_length;
  int m_team_scratch_size[2];
  int m_thread_scratch_size[2];
  int m_chunk_size;
  bool m_tune_team_size;
  bool m_tune_vector_length;

 public:
  using execution_space = Kokkos::Experimental::SYCL;

  template <class... OtherProperties>
  TeamPolicyInternal(TeamPolicyInternal<OtherProperties...> const& p) {
    m_league_size            = p.m_league_size;
    m_team_size              = p.m_team_size;
    m_vector_length          = p.m_vector_length;
    m_team_scratch_size[0]   = p.m_team_scratch_size[0];
    m_team_scratch_size[1]   = p.m_team_scratch_size[1];
    m_thread_scratch_size[0] = p.m_thread_scratch_size[0];
    m_thread_scratch_size[1] = p.m_thread_scratch_size[1];
    m_chunk_size             = p.m_chunk_size;
    m_space                  = p.m_space;
    m_tune_team_size         = p.m_tune_team_size;
    m_tune_vector_length     = p.m_tune_vector_length;
  }

  template <typename FunctorType>
  int team_size_max(FunctorType const& f, ParallelForTag const&) const {
    return internal_team_size_max_for(f);
  }

  template <class FunctorType>
  inline int team_size_max(const FunctorType& f,
                           const ParallelReduceTag&) const {
    return internal_team_size_max_reduce(f);
  }

  template <class FunctorType, class ReducerType>
  inline int team_size_max(const FunctorType& f, const ReducerType& /*r*/,
                           const ParallelReduceTag&) const {
    return internal_team_size_max_reduce(f);
  }

  template <typename FunctorType>
  int team_size_recommended(FunctorType const& f, ParallelForTag const&) const {
    return internal_team_size_max_for(f);
  }

  template <typename FunctorType>
  inline int team_size_recommended(FunctorType const& f,
                                   ParallelReduceTag const&) const {
    return internal_team_size_recommended_reduce(f);
  }

  template <class FunctorType, class ReducerType>
  int team_size_recommended(FunctorType const& f, ReducerType const&,
                            ParallelReduceTag const&) const {
    return internal_team_size_recommended_reduce(f);
  }
  inline bool impl_auto_vector_length() const { return m_tune_vector_length; }
  inline bool impl_auto_team_size() const { return m_tune_team_size; }
  // FIXME_SYCL This is correct in most cases, but not necessarily in case a
  // custom sycl::queue is used to initialize the execution space.
  static int vector_length_max() {
    std::vector<size_t> sub_group_sizes =
        execution_space{}
            .impl_internal_space_instance()
            ->m_queue->get_device()
            .template get_info<sycl::info::device::sub_group_sizes>();
    return *std::max_element(sub_group_sizes.begin(), sub_group_sizes.end());
  }

 private:
  static int verify_requested_vector_length(int requested_vector_length) {
    int test_vector_length =
        std::min(requested_vector_length, vector_length_max());

    // Allow only power-of-two vector_length
    if (!(is_integral_power_of_two(test_vector_length))) {
      int test_pow2 = 1;
      while (test_pow2 < test_vector_length) test_pow2 <<= 1;
      test_vector_length = test_pow2 >> 1;
    }

    return test_vector_length;
  }

 public:
  static int scratch_size_max(int level) {
    return level == 0 ? 1024 * 32
                      :           // FIXME_SYCL arbitrarily setting this to 32kB
               20 * 1024 * 1024;  // FIXME_SYCL arbitrarily setting this to 20MB
  }
  inline void impl_set_vector_length(size_t size) { m_vector_length = size; }
  inline void impl_set_team_size(size_t size) { m_team_size = size; }
  int impl_vector_length() const { return m_vector_length; }
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_3
  KOKKOS_DEPRECATED int vector_length() const { return impl_vector_length(); }
#endif

  int team_size() const { return m_team_size; }

  int league_size() const { return m_league_size; }

  int scratch_size(int level, int team_size_ = -1) const {
    if (team_size_ < 0) team_size_ = m_team_size;
    return m_team_scratch_size[level] +
           team_size_ * m_thread_scratch_size[level];
  }

  int team_scratch_size(int level) const { return m_team_scratch_size[level]; }

  int thread_scratch_size(int level) const {
    return m_thread_scratch_size[level];
  }

  typename traits::execution_space space() const { return m_space; }

  TeamPolicyInternal()
      : m_space(typename traits::execution_space()),
        m_league_size(0),
        m_team_size(-1),
        m_vector_length(0),
        m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0),
        m_tune_team_size(false),
        m_tune_vector_length(false) {}

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal(const execution_space space_, int league_size_,
                     int team_size_request, int vector_length_request = 1)
      : m_space(space_),
        m_league_size(league_size_),
        m_team_size(team_size_request),
        m_vector_length(
            (vector_length_request > 0)
                ? verify_requested_vector_length(vector_length_request)
                : (verify_requested_vector_length(1))),
        m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0),
        m_tune_team_size(bool(team_size_request <= 0)),
        m_tune_vector_length(bool(vector_length_request <= 0)) {
    // FIXME_SYCL Check that league size is permissible,
    // https://github.com/intel/llvm/pull/4064

    // Make sure total block size is permissible
    if (m_team_size * m_vector_length >
        static_cast<int>(
            m_space.impl_internal_space_instance()->m_maxWorkgroupSize)) {
      Impl::throw_runtime_exception(
          std::string("Kokkos::TeamPolicy<SYCL> the team size is too large. "
                      "Team size x vector length is " +
                      std::to_string(m_team_size * m_vector_length) +
                      " but must be smaller than ") +
          std::to_string(
              m_space.impl_internal_space_instance()->m_maxWorkgroupSize));
    }
  }

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal(const execution_space space_, int league_size_,
                     const Kokkos::AUTO_t& /* team_size_request */,
                     int vector_length_request = 1)
      : TeamPolicyInternal(space_, league_size_, -1, vector_length_request) {}
  // FLAG
  /** \brief  Specify league size and team size, request vector length*/
  TeamPolicyInternal(const execution_space space_, int league_size_,
                     int team_size_request,
                     const Kokkos::AUTO_t& /* vector_length_request */
                     )
      : TeamPolicyInternal(space_, league_size_, team_size_request, -1)

  {}

  /** \brief  Specify league size, request team size and vector length*/
  TeamPolicyInternal(const execution_space space_, int league_size_,
                     const Kokkos::AUTO_t& /* team_size_request */,
                     const Kokkos::AUTO_t& /* vector_length_request */

                     )
      : TeamPolicyInternal(space_, league_size_, -1, -1)

  {}

  TeamPolicyInternal(int league_size_, int team_size_request,
                     int vector_length_request = 1)
      : TeamPolicyInternal(typename traits::execution_space(), league_size_,
                           team_size_request, vector_length_request) {}

  TeamPolicyInternal(int league_size_,
                     const Kokkos::AUTO_t& /* team_size_request */,
                     int vector_length_request = 1)
      : TeamPolicyInternal(typename traits::execution_space(), league_size_, -1,
                           vector_length_request) {}

  /** \brief  Specify league size and team size, request vector length*/
  TeamPolicyInternal(int league_size_, int team_size_request,
                     const Kokkos::AUTO_t& /* vector_length_request */

                     )
      : TeamPolicyInternal(typename traits::execution_space(), league_size_,
                           team_size_request, -1)

  {}

  /** \brief  Specify league size, request team size and vector length*/
  TeamPolicyInternal(int league_size_,
                     const Kokkos::AUTO_t& /* team_size_request */,
                     const Kokkos::AUTO_t& /* vector_length_request */

                     )
      : TeamPolicyInternal(typename traits::execution_space(), league_size_, -1,
                           -1) {}

  int chunk_size() const { return m_chunk_size; }

  TeamPolicyInternal& set_chunk_size(typename traits::index_type chunk_size_) {
    m_chunk_size = chunk_size_;
    return *this;
  }

  /** \brief set per team scratch size for a specific level of the scratch
   * hierarchy */
  TeamPolicyInternal& set_scratch_size(int level,
                                       PerTeamValue const& per_team) {
    m_team_scratch_size[level] = per_team.value;
    return *this;
  }

  /** \brief set per thread scratch size for a specific level of the scratch
   * hierarchy */
  TeamPolicyInternal& set_scratch_size(int level,
                                       PerThreadValue const& per_thread) {
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

  /** \brief set per thread and per team scratch size for a specific level of
   * the scratch hierarchy */
  TeamPolicyInternal& set_scratch_size(int level, PerTeamValue const& per_team,
                                       PerThreadValue const& per_thread) {
    m_team_scratch_size[level]   = per_team.value;
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

  using member_type = Kokkos::Impl::SYCLTeamMember;

 protected:
  template <class FunctorType>
  int internal_team_size_max_for(const FunctorType& /*f*/) const {
    // nested_reducer_memsize = (sizeof(double) * (m_team_size + 2)
    // custom: m_team_scratch_size[0] + m_thread_scratch_size[0] * m_team_size
    // total:
    // 2*sizeof(double)+m_team_scratch_size[0]
    // + m_team_size(sizeof(double)+m_thread_scratch_size[0])
    const int max_threads_for_memory =
        (space().impl_internal_space_instance()->m_maxShmemPerBlock -
         2 * sizeof(double) - m_team_scratch_size[0]) /
        (sizeof(double) + m_thread_scratch_size[0]);
    return std::min<int>(
               m_space.impl_internal_space_instance()->m_maxWorkgroupSize,
               max_threads_for_memory) /
           impl_vector_length();
  }

  template <class FunctorType>
  int internal_team_size_max_reduce(const FunctorType& f) const {
    using Analysis        = FunctorAnalysis<FunctorPatternInterface::REDUCE,
                                     TeamPolicyInternal, FunctorType>;
    using value_type      = typename Analysis::value_type;
    const int value_count = Analysis::value_count(f);

    // nested_reducer_memsize = (sizeof(double) * (m_team_size + 2)
    // reducer_memsize = sizeof(value_type) * m_team_size * value_count
    // custom: m_team_scratch_size[0] + m_thread_scratch_size[0] * m_team_size
    // total:
    // 2*sizeof(double)+m_team_scratch_size[0]
    // + m_team_size(sizeof(double)+sizeof(value_type)*value_count
    //               +m_thread_scratch_size[0])
    const int max_threads_for_memory =
        (space().impl_internal_space_instance()->m_maxShmemPerBlock -
         2 * sizeof(double) - m_team_scratch_size[0]) /
        (sizeof(double) + sizeof(value_type) * value_count +
         m_thread_scratch_size[0]);
    return std::min<int>(
               m_space.impl_internal_space_instance()->m_maxWorkgroupSize,
               max_threads_for_memory) /
           impl_vector_length();
  }

  template <class FunctorType>
  int internal_team_size_recommended_for(const FunctorType& f) const {
    // FIXME_SYCL improve
    return internal_team_size_max_for(f);
  }

  template <class FunctorType>
  int internal_team_size_recommended_reduce(const FunctorType& f) const {
    // FIXME_SYCL improve
    return internal_team_size_max_reduce(f);
  }
};

template <typename FunctorType, typename... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                  Kokkos::Experimental::SYCL> {
 public:
  using Policy = TeamPolicyInternal<Kokkos::Experimental::SYCL, Properties...>;
  using functor_type = FunctorType;
  using size_type    = ::Kokkos::Experimental::SYCL::size_type;

 private:
  using member_type   = typename Policy::member_type;
  using work_tag      = typename Policy::work_tag;
  using launch_bounds = typename Policy::launch_bounds;

  FunctorType const m_functor;
  Policy const m_policy;
  size_type const m_league_size;
  int m_team_size;
  size_type const m_vector_size;
  int m_shmem_begin;
  int m_shmem_size;
  void* m_scratch_ptr[2];
  int m_scratch_size[2];
  // Only let one ParallelFor/Reduce modify the team scratch memory. The
  // constructor acquires the mutex which is released in the destructor.
  std::scoped_lock<std::mutex> m_scratch_lock;

  template <typename Functor>
  sycl::event sycl_direct_launch(const Policy& policy,
                                 const Functor& functor) const {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    sycl::queue& q = *instance.m_queue;

    auto parallel_for_event = q.submit([&](sycl::handler& cgh) {
      // FIXME_SYCL accessors seem to need a size greater than zero at least for
      // host queues
      sycl::accessor<char, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          team_scratch_memory_L0(
              sycl::range<1>(std::max(m_scratch_size[0] + m_shmem_begin, 1)),
              cgh);

      // Avoid capturing *this since it might not be trivially copyable
      const auto shmem_begin     = m_shmem_begin;
      const int scratch_size[2]  = {m_scratch_size[0], m_scratch_size[1]};
      void* const scratch_ptr[2] = {m_scratch_ptr[0], m_scratch_ptr[1]};

      cgh.parallel_for(
          sycl::nd_range<2>(
              sycl::range<2>(m_team_size, m_league_size * m_vector_size),
              sycl::range<2>(m_team_size, m_vector_size)),
          [=](sycl::nd_item<2> item) {
#ifdef KOKKOS_ENABLE_DEBUG
            if (item.get_sub_group().get_local_range() %
                    item.get_local_range(1) !=
                0)
              Kokkos::abort(
                  "The sub_group size is not divisible by the vector_size. "
                  "Choose a smaller vector_size!");
#endif
            const member_type team_member(
                team_scratch_memory_L0.get_pointer(), shmem_begin,
                scratch_size[0],
                static_cast<char*>(scratch_ptr[1]) +
                    item.get_group(1) * scratch_size[1],
                scratch_size[1], item);
            if constexpr (std::is_same<work_tag, void>::value)
              functor(team_member);
            else
              functor(work_tag(), team_member);
          });
    });
    q.submit_barrier(std::vector<sycl::event>{parallel_for_event});
    return parallel_for_event;
  }

 public:
  inline void execute() const {
    if (m_league_size == 0) return;

    Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMem&
        indirectKernelMem = m_policy.space()
                                .impl_internal_space_instance()
                                ->m_indirectKernelMem;

    const auto functor_wrapper = Experimental::Impl::make_sycl_function_wrapper(
        m_functor, indirectKernelMem);

    sycl::event event =
        sycl_direct_launch(m_policy, functor_wrapper.get_functor());
    functor_wrapper.register_event(indirectKernelMem, event);
  }

  ParallelFor(FunctorType const& arg_functor, Policy const& arg_policy)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_league_size(arg_policy.league_size()),
        m_team_size(arg_policy.team_size()),
        m_vector_size(arg_policy.impl_vector_length()),
        m_scratch_lock(arg_policy.space()
                           .impl_internal_space_instance()
                           ->m_team_scratch_mutex) {
    // FIXME_SYCL optimize
    if (m_team_size < 0) m_team_size = 32;

    m_shmem_begin = (sizeof(double) * (m_team_size + 2));
    m_shmem_size =
        (m_policy.scratch_size(0, m_team_size) +
         FunctorTeamShmemSize<FunctorType>::value(m_functor, m_team_size));
    m_scratch_size[0] = m_shmem_size;
    m_scratch_size[1] = m_policy.scratch_size(1, m_team_size);

    // FIXME_SYCL so far accessors used instead of these pointers
    // Functor's reduce memory, team scan memory, and team shared memory depend
    // upon team size.
    auto& space      = *m_policy.space().impl_internal_space_instance();
    m_scratch_ptr[0] = nullptr;
    m_scratch_ptr[1] = space.resize_team_scratch_space(
        static_cast<ptrdiff_t>(m_scratch_size[1]) * m_league_size);

    if (static_cast<int>(space.m_maxShmemPerBlock) <
        m_shmem_size - m_shmem_begin) {
      std::stringstream out;
      out << "Kokkos::Impl::ParallelFor<SYCL> insufficient shared memory! "
             "Requested "
          << m_shmem_size - m_shmem_begin << " bytes but maximum is "
          << space.m_maxShmemPerBlock << '\n';
      Kokkos::Impl::throw_runtime_exception(out.str());
    }

    const auto max_team_size =
        m_policy.team_size_max(arg_functor, ParallelForTag{});
    if (m_team_size > m_policy.team_size_max(arg_functor, ParallelForTag{}))
      Kokkos::Impl::throw_runtime_exception(
          "Kokkos::Impl::ParallelFor<SYCL> requested too large team size. The "
          "maximal team_size is " +
          std::to_string(max_team_size) + '!');
  }
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <class FunctorType, class ReducerType, class... Properties>
class ParallelReduce<FunctorType, Kokkos::TeamPolicy<Properties...>,
                     ReducerType, Kokkos::Experimental::SYCL> {
 public:
  using Policy = TeamPolicyInternal<Kokkos::Experimental::SYCL, Properties...>;

 private:
  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;
  using member_type   = typename Policy::member_type;
  using WorkTag       = typename Policy::work_tag;
  using launch_bounds = typename Policy::launch_bounds;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;
  using value_type     = typename Analysis::value_type;

 public:
  using functor_type = FunctorType;
  using size_type    = Kokkos::Experimental::SYCL::size_type;

 private:
  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;
  const bool m_result_ptr_device_accessible;
  // FIXME_SYCL avoid reallocating memory for reductions
  /*  size_type* m_scratch_space;
    size_type* m_scratch_flags;
    size_type m_team_begin;*/
  size_type m_shmem_begin;
  size_type m_shmem_size;
  void* m_scratch_ptr[2];
  int m_scratch_size[2];
  const size_type m_league_size;
  int m_team_size;
  const size_type m_vector_size;
  // Only let one ParallelFor/Reduce modify the team scratch memory. The
  // constructor acquires the mutex which is released in the destructor.
  std::scoped_lock<std::mutex> m_scratch_lock;

  template <typename PolicyType, typename Functor, typename Reducer>
  sycl::event sycl_direct_launch(const PolicyType& policy,
                                 const Functor& functor,
                                 const Reducer& reducer) const {
    using ReducerConditional =
        Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                           FunctorType, ReducerType>;
    using ReducerTypeFwd = typename ReducerConditional::type;
    using WorkTagFwd =
        std::conditional_t<std::is_same<InvalidType, ReducerType>::value,
                           WorkTag, void>;
    using ValueInit =
        Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;
    using ValueJoin =
        Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, WorkTagFwd>;
    using ValueOps = Kokkos::Impl::FunctorValueOps<FunctorType, WorkTag>;

    auto selected_reducer = ReducerConditional::select(functor, reducer);

    // Convenience references
    const Kokkos::Experimental::SYCL& space = policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    sycl::queue& q = *instance.m_queue;

    // FIXME_SYCL optimize
    const size_t wgroup_size = m_team_size * m_vector_size;
    std::size_t size = std::size_t(m_league_size) * m_team_size * m_vector_size;
    const auto init_size =
        std::max<std::size_t>((size + wgroup_size - 1) / wgroup_size, 1);
    const unsigned int value_count =
        FunctorValueTraits<ReducerTypeFwd, WorkTagFwd>::value_count(
            selected_reducer);
    const auto results_ptr = static_cast<pointer_type>(instance.scratch_space(
        sizeof(value_type) * std::max(value_count, 1u) * init_size));
    value_type* device_accessible_result_ptr =
        m_result_ptr_device_accessible ? m_result_ptr : nullptr;
    auto scratch_flags = static_cast<unsigned int*>(
        instance.scratch_flags(sizeof(unsigned int)));

    sycl::event last_reduction_event;

    // If size<=1 we only call init(), the functor and possibly final once
    // working with the global scratch memory but don't copy back to
    // m_result_ptr yet.
    if (size <= 1) {
      auto parallel_reduce_event = q.submit([&](sycl::handler& cgh) {
        // FIXME_SYCL accessors seem to need a size greater than zero at least
        // for host queues
        sycl::accessor<char, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            team_scratch_memory_L0(
                sycl::range<1>(std::max(m_scratch_size[0] + m_shmem_begin, 1)),
                cgh);

        // Avoid capturing *this since it might not be trivially copyable
        const auto shmem_begin     = m_shmem_begin;
        const int scratch_size[2]  = {m_scratch_size[0], m_scratch_size[1]};
        void* const scratch_ptr[2] = {m_scratch_ptr[0], m_scratch_ptr[1]};

        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(1, 1), sycl::range<2>(1, 1)),
            [=](sycl::nd_item<2> item) {
              const auto& selected_reducer = ReducerConditional::select(
                  static_cast<const FunctorType&>(functor),
                  static_cast<const ReducerType&>(reducer));
              reference_type update =
                  ValueInit::init(selected_reducer, results_ptr);
              if (size == 1) {
                const member_type team_member(
                    team_scratch_memory_L0.get_pointer(), shmem_begin,
                    scratch_size[0], static_cast<char*>(scratch_ptr[1]),
                    scratch_size[1], item);
                if constexpr (std::is_same<WorkTag, void>::value)
                  functor(team_member, update);
                else
                  functor(WorkTag(), team_member, update);
              }
              if constexpr (ReduceFunctorHasFinal<FunctorType>::value)
                FunctorFinal<FunctorType, WorkTag>::final(
                    static_cast<const FunctorType&>(functor), results_ptr);
              if (device_accessible_result_ptr)
                ValueOps::copy(functor, device_accessible_result_ptr,
                               &results_ptr[0]);
            });
      });
      q.submit_barrier(std::vector<sycl::event>{parallel_reduce_event});
      last_reduction_event = parallel_reduce_event;
    }

    // Otherwise, we perform a reduction on the values in all workgroups
    // separately, write the workgroup results back to global memory and recurse
    // until only one workgroup does the reduction and thus gets the final
    // value.
    if (size > 1) {
      auto n_wgroups             = (size + wgroup_size - 1) / wgroup_size;
      auto parallel_reduce_event = q.submit([&](sycl::handler& cgh) {
        // FIXME_SYCL eor some reason, we get "misaligned memory" errors when
        // using a separate local memory allocation using SYCL+CUDA
        sycl::accessor<value_type, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            local_mem(sycl::range<1>(wgroup_size) * std::max(value_count, 1u) +
                          (sizeof(unsigned int) + sizeof(value_type) - 1) /
                              sizeof(value_type),
                      cgh);

        // FIXME_SYCL accessors seem to need a size greater than zero at least
        // for host queues
        sycl::accessor<char, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            team_scratch_memory_L0(
                sycl::range<1>(std::max(m_scratch_size[0] + m_shmem_begin, 1)),
                cgh);

        // Avoid capturing *this since it might not be trivially copyable
        const auto shmem_begin     = m_shmem_begin;
        const int scratch_size[2]  = {m_scratch_size[0], m_scratch_size[1]};
        void* const scratch_ptr[2] = {m_scratch_ptr[0], m_scratch_ptr[1]};

        cgh.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(m_team_size, m_league_size * m_vector_size),
                sycl::range<2>(m_team_size, m_vector_size)),
            [=](sycl::nd_item<2> item) {
#ifdef KOKKOS_ENABLE_DEBUG
              if (item.get_sub_group().get_local_range() %
                      item.get_local_range(1) !=
                  0)
                Kokkos::abort(
                    "The sub_group size is not divisible by the vector_size. "
                    "Choose a smaller vector_size!");
#endif
              auto& num_teams_done = reinterpret_cast<unsigned int&>(
                  local_mem[wgroup_size * std::max(value_count, 1u)]);
              const auto local_id          = item.get_local_linear_id();
              const auto& selected_reducer = ReducerConditional::select(
                  static_cast<const FunctorType&>(functor),
                  static_cast<const ReducerType&>(reducer));

              // In the first iteration, we call functor to initialize the local
              // memory. Otherwise, the local memory is initialized with the
              // results from the previous iteration that are stored in global
              // memory.
              reference_type update = ValueInit::init(
                  selected_reducer, &local_mem[local_id * value_count]);
              const member_type team_member(
                  team_scratch_memory_L0.get_pointer(), shmem_begin,
                  scratch_size[0],
                  static_cast<char*>(scratch_ptr[1]) +
                      item.get_group(1) * scratch_size[1],
                  scratch_size[1], item);
              if constexpr (std::is_same<WorkTag, void>::value)
                functor(team_member, update);
              else
                functor(WorkTag(), team_member, update);
              item.barrier(sycl::access::fence_space::local_space);

              SYCLReduction::workgroup_reduction<ValueJoin, ValueOps, WorkTag>(
                  item, local_mem.get_pointer(), results_ptr,
                  device_accessible_result_ptr, value_count, selected_reducer,
                  static_cast<const FunctorType&>(functor), false,
                  item.get_local_range()[0]);

              if (local_id == 0)
                num_teams_done = Kokkos::atomic_fetch_add(scratch_flags, 1) + 1;
              item.barrier(sycl::access::fence_space::local_space);
              if (num_teams_done == n_wgroups) {
                if (local_id >= n_wgroups)
                  ValueInit::init(selected_reducer,
                                  &local_mem[local_id * value_count]);
                else {
                  ValueOps::copy(functor, &local_mem[local_id * value_count],
                                 &results_ptr[local_id * value_count]);
                  for (unsigned int id = local_id + wgroup_size; id < n_wgroups;
                       id += wgroup_size) {
                    ValueJoin::join(selected_reducer,
                                    &local_mem[local_id * value_count],
                                    &results_ptr[id * value_count]);
                  }
                }

                SYCLReduction::workgroup_reduction<ValueJoin, ValueOps,
                                                   WorkTag>(
                    item, local_mem.get_pointer(), results_ptr,
                    device_accessible_result_ptr, value_count, selected_reducer,
                    static_cast<const FunctorType&>(functor), true, n_wgroups);
              }
            });
      });
      last_reduction_event =
          q.submit_barrier(std::vector<sycl::event>{parallel_reduce_event});
    }

    // At this point, the reduced value is written to the entry in results_ptr
    // and all that is left is to copy it back to the given result pointer if
    // necessary.
    if (m_result_ptr && !m_result_ptr_device_accessible) {
      Kokkos::Impl::DeepCopy<Kokkos::Experimental::SYCLDeviceUSMSpace,
                             Kokkos::Experimental::SYCLDeviceUSMSpace>(
          space, m_result_ptr, results_ptr,
          sizeof(*m_result_ptr) * value_count);
      space.fence(
          "Kokkos::Impl::ParallelReduce<TeamPolicy,SYCL>: fence because "
          "reduction can't access result storage location");
    }

    return last_reduction_event;
  }

 public:
  inline void execute() {
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *m_policy.space().impl_internal_space_instance();
    using IndirectKernelMem =
        Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMem;
    IndirectKernelMem& indirectKernelMem  = instance.m_indirectKernelMem;
    IndirectKernelMem& indirectReducerMem = instance.m_indirectReducerMem;

    const auto functor_wrapper = Experimental::Impl::make_sycl_function_wrapper(
        m_functor, indirectKernelMem);
    const auto reducer_wrapper = Experimental::Impl::make_sycl_function_wrapper(
        m_reducer, indirectReducerMem);

    sycl::event event = sycl_direct_launch(
        m_policy, functor_wrapper.get_functor(), reducer_wrapper.get_functor());
    functor_wrapper.register_event(indirectKernelMem, event);
    reducer_wrapper.register_event(indirectReducerMem, event);
  }

 private:
  void initialize() {
    // FIXME_SYCL optimize
    if (m_team_size < 0) m_team_size = 32;
    // Must be a power of two greater than two, get the one not bigger than the
    // requested one.
    if ((m_team_size & m_team_size - 1) || m_team_size < 2) {
      int temp_team_size = 2;
      while ((temp_team_size << 1) < m_team_size) temp_team_size <<= 1;
      m_team_size = temp_team_size;
    }

    m_shmem_begin = (sizeof(double) * (m_team_size + 2));
    m_shmem_size =
        (m_policy.scratch_size(0, m_team_size) +
         FunctorTeamShmemSize<FunctorType>::value(m_functor, m_team_size));
    m_scratch_size[0] = m_shmem_size;
    m_scratch_size[1] = m_policy.scratch_size(1, m_team_size);

    // FIXME_SYCL so far accessors used instead of these pointers
    // Functor's reduce memory, team scan memory, and team shared memory depend
    // upon team size.
    auto& space      = *m_policy.space().impl_internal_space_instance();
    m_scratch_ptr[0] = nullptr;
    m_scratch_ptr[1] = space.resize_team_scratch_space(
        static_cast<ptrdiff_t>(m_scratch_size[1]) * m_league_size);

    if (static_cast<int>(space.m_maxShmemPerBlock) <
        m_shmem_size - m_shmem_begin) {
      std::stringstream out;
      out << "Kokkos::Impl::ParallelFor<SYCL> insufficient shared memory! "
             "Requested "
          << m_shmem_size - m_shmem_begin << " bytes but maximum is "
          << space.m_maxShmemPerBlock << '\n';
      Kokkos::Impl::throw_runtime_exception(out.str());
    }

    if (m_team_size > m_policy.team_size_max(m_functor, ParallelForTag{}))
      Kokkos::Impl::throw_runtime_exception(
          "Kokkos::Impl::ParallelFor<SYCL> requested too large team size.");
  }

 public:
  template <class ViewType>
  ParallelReduce(FunctorType const& arg_functor, Policy const& arg_policy,
                 ViewType const& arg_result,
                 typename std::enable_if<Kokkos::is_view<ViewType>::value,
                                         void*>::type = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result.data()),
        m_result_ptr_device_accessible(
            MemorySpaceAccess<Kokkos::Experimental::SYCLDeviceUSMSpace,
                              typename ViewType::memory_space>::accessible),
        m_league_size(arg_policy.league_size()),
        m_team_size(arg_policy.team_size()),
        m_vector_size(arg_policy.impl_vector_length()),
        m_scratch_lock(arg_policy.space()
                           .impl_internal_space_instance()
                           ->m_team_scratch_mutex) {
    initialize();
  }

  ParallelReduce(FunctorType const& arg_functor, Policy const& arg_policy,
                 ReducerType const& reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()),
        m_result_ptr_device_accessible(
            MemorySpaceAccess<Kokkos::Experimental::SYCLDeviceUSMSpace,
                              typename ReducerType::result_view_type::
                                  memory_space>::accessible),
        m_league_size(arg_policy.league_size()),
        m_team_size(arg_policy.team_size()),
        m_vector_size(arg_policy.impl_vector_length()),
        m_scratch_lock(arg_policy.space()
                           .impl_internal_space_instance()
                           ->m_team_scratch_mutex) {
    initialize();
  }
};
}  // namespace Impl
}  // namespace Kokkos

#endif
