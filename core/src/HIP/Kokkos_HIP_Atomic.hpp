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

#ifndef KOKKOS_HIP_ATOMIC_HPP
#define KOKKOS_HIP_ATOMIC_HPP

#ifdef KOKKOS_ENABLE_HIP_ATOMICS
namespace Kokkos {
// HIP can do:
// Types int/unsigned int
// variants:
// atomic_exchange/compare_exchange/fetch_add/fetch_sub/fetch_max/fetch_min/fetch_and/fetch_or/fetch_xor/fetch_inc/fetch_dec

__inline__ __device__ int atomic_exchange(volatile int *const dest,
                                          const int val) {
  return atomicExch(const_cast<int *>(dest), val);
}

__inline__ __device__ unsigned int atomic_exchange(
    volatile unsigned int *const dest, const unsigned int val) {
  return atomicExch(const_cast<unsigned int *>(dest), val);
}

__inline__ __device__ unsigned long long int atomic_exchange(
    volatile unsigned long long int *const dest,
    const unsigned long long int val) {
  return atomicExch(const_cast<unsigned long long *>(dest), val);
}

/** \brief  Atomic exchange for any type with compatible size */
template <typename T>
__inline__ __device__ T atomic_exchange(
    volatile T *const dest,
    typename std::enable_if<sizeof(T) == sizeof(int), const T &>::type val) {
  int tmp = atomicExch(reinterpret_cast<int *>(const_cast<T *>(dest)),
                       *reinterpret_cast<int *>(const_cast<T *>(&val)));
  return reinterpret_cast<T &>(tmp);
}

template <typename T>
__inline__ __device__ T atomic_exchange(
    volatile T *const dest,
    typename std::enable_if<sizeof(T) != sizeof(int) &&
                                sizeof(T) == sizeof(unsigned long long int),
                            const T &>::type val) {
  typedef unsigned long long int type;

  type tmp = atomicExch(reinterpret_cast<type *>(const_cast<T *>(dest)),
                        *reinterpret_cast<type *>(const_cast<T *>(&val)));
  return reinterpret_cast<T &>(tmp);
}

/** \brief  Atomic exchange for any type with compatible size */
template <typename T>
__inline__ __device__ void atomic_assign(
    volatile T *const dest,
    typename std::enable_if<sizeof(T) == sizeof(int), const T &>::type val) {
  atomicExch(reinterpret_cast<int *>(const_cast<T *>(dest)),
             *reinterpret_cast<int *>(const_cast<T *>(&val)));
}

template <typename T>
__inline__ __device__ void atomic_assign(
    volatile T *const dest,
    typename std::enable_if<sizeof(T) != sizeof(int) &&
                                sizeof(T) == sizeof(unsigned long long int),
                            const T &>::type val) {
  typedef unsigned long long int type;
  atomicExch(reinterpret_cast<type *>(const_cast<T *>(dest)),
             *reinterpret_cast<type *>(const_cast<T *>(&val)));
}

template <typename T>
__inline__ __device__ void atomic_assign(
    volatile T *const dest,
    typename std::enable_if<sizeof(T) != sizeof(int) &&
                                sizeof(T) != sizeof(unsigned long long int),
                            const T &>::type val) {
  atomic_exchange(dest, val);
}

inline __device__ int atomic_exchange(int *dest, const int &val) {
  return atomicExch(dest, val);
}

inline __device__ unsigned int atomic_exchange(unsigned int *dest,
                                               const unsigned int &val) {
  return atomicExch(dest, val);
}

inline __device__ unsigned long long int atomic_exchange(
    unsigned long long int *dest, const unsigned long long int &val) {
  return atomicExch(dest, val);
}

inline __device__ float atomic_exchange(float *dest, const float &val) {
  return atomicExch(dest, val);
}

template <class T>
inline __device__ T atomic_exchange(T * /*dest*/, const T &val) {
  // FIXME
  printf("Not implemented atomic_exchange\n");
  return val;
}

inline __device__ int atomic_compare_exchange(int *dest, int compare,
                                              const int &val) {
  return atomicCAS(dest, compare, val);
}

inline __device__ unsigned int atomic_compare_exchange(
    unsigned int *dest, unsigned int compare, const unsigned int &val) {
  return atomicCAS(dest, compare, val);
}

inline __device__ unsigned long long int atomic_compare_exchange(
    unsigned long long int *dest, unsigned long long int compare,
    const unsigned long long int &val) {
  return atomicCAS(dest, compare, val);
}
template <typename T>
inline __device__ T atomic_compare_exchange(T * /*dest*/, T /*compare*/,
                                            const T &val) {
  // FIXME
  printf("Not implemented atomic_compare_exchange\n");
  return val;
}

template <typename T>
inline __device__ T atomic_compare_exchange(volatile T * /*dest*/,
                                            T /*compare*/, const T &val) {
  // FIXME
  printf("Not implemented volatile atomic_compare_exchange\n");
  return val;
}

/*
  KOKKOS_INLINE_FUNCTION
  int64_t atomic_compare_exchange(int64_t* dest, int64_t compare, const int64_t&
  val) { return (int64_t) hc::atomic_compare_exchange_uint64((uint64_t*)dest,
  (uint64_t)compare, (const uint64_t&)val);
  }

  KOKKOS_INLINE_FUNCTION
  uint64_t atomic_compare_exchange(uint64_t* dest, uint64_t compare, const
  uint64_t& val) { return hc::atomic_compare_exchange_uint64(dest, compare,
  val);
  }

  KOKKOS_INLINE_FUNCTION
  long long atomic_compare_exchange(long long* dest, long long compare, const
  long long& val) { return (long
  long)hc::atomic_compare_exchange_uint64((uint64_t*)(dest),
  (uint64_t)(compare), (const uint64_t&)(val));
  }

  KOKKOS_INLINE_FUNCTION
  float atomic_compare_exchange(float* dest, float compare, const float& val) {
    union U {
      int i ;
      float f ;
      KOKKOS_INLINE_FUNCTION U() {}
    } idest,icompare,ival;
    idest.f = *dest;
    icompare.f = compare;
    ival.f = val;
    idest.i = hc::atomic_compare_exchange_int(reinterpret_cast<int*>(dest),
  icompare.i, ival.i); return idest.f;
  }

  KOKKOS_INLINE_FUNCTION
  double atomic_compare_exchange(double* dest, double compare, const double&
  val) { union U { uint64_t i ; double d ; KOKKOS_INLINE_FUNCTION U() {}; }
  idest,icompare,ival; idest.d = *dest; icompare.d = compare; ival.d = val;
    idest.i =
  hc::atomic_compare_exchange_uint64(reinterpret_cast<uint64_t*>(dest),
  icompare.i, ival.i); return idest.d;
  }

  template<class T>
  KOKKOS_INLINE_FUNCTION
  T atomic_compare_exchange(volatile T* dest, T compare, typename
  std::enable_if<sizeof(T) == sizeof(int), const T&>::type val) { union U { int
  i ; T f ; KOKKOS_INLINE_FUNCTION U() {}; } idest,icompare,ival; idest.f =
  *dest; icompare.f = compare; ival.f = val; idest.i =
  hc::atomic_compare_exchange_int((int*)(dest), icompare.i, ival.i); return
  idest.f;
  }

  template<class T>
  KOKKOS_INLINE_FUNCTION
  T atomic_compare_exchange(volatile T* dest, T compare, typename
  std::enable_if<sizeof(T) == sizeof(int64_t), const T&>::type val) { union U {
      uint64_t i ;
      T f ;
      KOKKOS_INLINE_FUNCTION U() {}
    } idest,icompare,ival;
    idest.f = *dest;
    icompare.f = compare;
    ival.f = val;
    idest.i = hc::atomic_compare_exchange_uint64((uint64_t*)(dest), icompare.i,
  ival.i); return idest.f;
  }

  template<class T>
  KOKKOS_INLINE_FUNCTION
  T atomic_compare_exchange(volatile T* dest, T compare, typename
  std::enable_if<(sizeof(T) != sizeof(int32_t)) && (sizeof(T) !=
  sizeof(int64_t)), const T&>::type val) { return val;
  }
*/
inline __device__ int atomic_fetch_add(volatile int *dest, const int &val) {
  return atomicAdd(const_cast<int *>(dest), val);
}

inline __device__ unsigned int atomic_fetch_add(volatile unsigned int *dest,
                                                const unsigned int &val) {
  return atomicAdd(const_cast<unsigned int *>(dest), val);
}

inline __device__ unsigned long long atomic_fetch_add(
    volatile unsigned long long *dest, const unsigned long long &val) {
  return atomicAdd(const_cast<unsigned long long *>(dest), val);
}

//template <typename T>
//inline __device__ T atomic_fetch_add(volatile T * /*dest*/, const T &val) {
  // FIXME
//  Kokkos::abort("Not implemented\n");
//  return val;
//}

template <typename T>
inline __device__ T atomic_fetch_add(
    volatile T* const dest,
    typename std::enable_if<sizeof(T) == sizeof(int), const T>::type val) {
  union U {
    int i;
    T t;
    KOKKOS_INLINE_FUNCTION U() {}
  } assume, oldval, newval;

  oldval.t = *dest;

  do {
    assume.i = oldval.i;
    newval.t = assume.t + val;
    oldval.i = atomicCAS((int*)dest, assume.i, newval.i);
  } while (assume.i != oldval.i);

  return oldval.t;
}

template <typename T>
inline __device__ T atomic_fetch_add(
    volatile T* const dest,
    typename std::enable_if<sizeof(T) == sizeof(unsigned long long), const T>::type val) {
printf("Using long long\n");
  union U {
    unsigned long long i;
    T t;
    KOKKOS_INLINE_FUNCTION U() {}
  } assume, oldval, newval;

  oldval.t = *dest;
  printf("before Using long long\n");

  do {
    assume.i = oldval.i;
    newval.t = assume.t + val;
    oldval.i = atomic_compare_exchange((unsigned long long*)dest, assume.i, newval.i);
  } while (assume.i != oldval.i);
  
printf("after Using long long\n");
  return oldval.t;
}


// FIXME Not implemented in HIP

// KOKKOS_INLINE_FUNCTION
// int64_t atomic_fetch_add(volatile int64_t* dest, const int64_t& val) {
//   return atomicAdd(const_cast<int64_t*>(dest),val);
// }
KOKKOS_INLINE_FUNCTION
char atomic_fetch_add(volatile char *dest, const char &val) {
  unsigned int oldval, newval, assume;
  oldval = *(int *)dest;

  do {
    assume = oldval;
    newval = assume & 0x7fffff00 + ((assume & 0xff) + val) & 0xff;
    oldval = atomicCAS((unsigned int *)dest, assume, newval);
  } while (assume != oldval);

  return oldval;
}

KOKKOS_INLINE_FUNCTION
short atomic_fetch_add(volatile short *dest, const short &val) {
  unsigned int oldval, newval, assume;
  oldval = *(int *)dest;

  do {
    assume = oldval;
    newval = assume & 0x7fff0000 + ((assume & 0xffff) + val) & 0xffff;
    oldval = atomicCAS((unsigned int *)dest, assume, newval);
  } while (assume != oldval);

  return oldval;
}

KOKKOS_INLINE_FUNCTION
long long atomic_fetch_add(volatile long long *dest, const long long &val) {
  return atomicAdd((unsigned long long *)(dest), val);
}

/*
template <class T>
KOKKOS_INLINE_FUNCTION T atomic_fetch_add(
    volatile T *dest,
    typename std::enable_if<sizeof(T) == sizeof(int), const T &>::type val)
{
  union U {
    unsigned int i;
    T t;
    KOKKOS_INLINE_FUNCTION U() {}
  } assume, oldval, newval;

  oldval.t = *dest;

  do
  {
    assume.i = oldval.i;
    newval.t = assume.t + val;
    oldval.i =
        atomic_compare_exchange((unsigned int *)(dest), assume.i, newval.i);
  } while (assume.i != oldval.i);

  return oldval.t;
}

template <class T>
KOKKOS_INLINE_FUNCTION T atomic_fetch_add(
    volatile T *dest, typename std::enable_if<sizeof(T) != sizeof(int) &&
                                                  sizeof(T) == sizeof(int64_t),
                                              const T &>::type val)
{
  union U {
    uint64_t i;
    T t;
    KOKKOS_INLINE_FUNCTION U() {}
  } assume, oldval, newval;

  oldval.t = *dest;

  do
  {
    assume.i = oldval.i;
    newval.t = assume.t + val;
    oldval.i = atomic_compare_exchange((uint64_t *)dest, assume.i, newval.i);
  } while (assume.i != oldval.i);

  return oldval.t;
}

// WORKAROUND
template <class T>
KOKKOS_INLINE_FUNCTION T atomic_fetch_add(
    volatile T *dest, typename std::enable_if<sizeof(T) != sizeof(int) &&
                                                  sizeof(T) != sizeof(int64_t),
                                              const T &>::type val)
{
  T return_val;
  // Do we need to (like in CUDA) handle potential wavefront branching?
  int done = 0;
  // unsigned int active = KOKKOS_IMPL_CUDA_BALLOT(1);
  // unsigned int done_active = 0;
  // while (active!=done_active) {
  if (!done)
  {
    bool locked = ::Kokkos::Impl::lock_address_hip_space((void *)dest);
    if (locked)
    {
      return_val = *dest;
      *dest = return_val + val;
      ::Kokkos::Impl::unlock_address_hip_space((void *)dest);
      done = 1;
    }
  }
  // done_active = KOKKOS_IMPL_CUDA_BALLOT(done);
  //}
  return return_val;
}

*/

KOKKOS_INLINE_FUNCTION
int atomic_fetch_sub(volatile int *dest, int const &val) {
  return atomicSub(const_cast<int *>(dest), val);
}

KOKKOS_INLINE_FUNCTION
unsigned int atomic_fetch_sub(volatile unsigned int *dest,
                              unsigned int const &val) {
  return atomicSub(const_cast<unsigned int *>(dest), val);
}

KOKKOS_INLINE_FUNCTION
int64_t atomic_fetch_sub(int64_t *dest, int64_t const &val) {
  return static_cast<int64_t>(
      atomicAdd(reinterpret_cast<unsigned long long *>(dest),
                -reinterpret_cast<unsigned long long const &>(val)));
}

KOKKOS_INLINE_FUNCTION
char atomic_fetch_sub(volatile char *dest, const char &val) {
  unsigned int oldval, newval, assume;
  oldval = *(int *)dest;

  do {
    assume = oldval;
    newval = assume & 0x7fffff00 + ((assume & 0xff) - val) & 0xff;
    oldval =
        atomicCAS(reinterpret_cast<unsigned int *>(const_cast<char *>(dest)),
                  assume, newval);
  } while (assume != oldval);

  return oldval;
}

KOKKOS_INLINE_FUNCTION
short atomic_fetch_sub(volatile short *dest, const short &val) {
  unsigned int oldval, newval, assume;
  oldval = *(int *)dest;

  do {
    assume = oldval;
    newval = assume & 0x7fff0000 + ((assume & 0xffff) - val) & 0xffff;
    oldval =
        atomicCAS(reinterpret_cast<unsigned int *>(const_cast<short *>(dest)),
                  assume, newval);
  } while (assume != oldval);

  return oldval;
}

KOKKOS_INLINE_FUNCTION
long long atomic_fetch_sub(volatile long long *dest, const long long &val) {
  return static_cast<long long>(atomicAdd(
      reinterpret_cast<unsigned long long int *>(const_cast<long long *>(dest)),
      -reinterpret_cast<unsigned long long int const &>(val)));
}

template <class T>
KOKKOS_INLINE_FUNCTION T atomic_fetch_sub(
    volatile T *dest,
    typename std::enable_if<sizeof(T) == sizeof(int), T>::type val) {
  union U {
    int i;
    T t;
    KOKKOS_INLINE_FUNCTION U() {}
  } assume, oldval, newval;

  oldval.t = *dest;

  do {
    assume.i = oldval.i;
    newval.t = assume.t - val;
    oldval.i = atomic_compare_exchange((int *)dest, assume.i, newval.i);
  } while (assume.i != oldval.i);

  return oldval.t;
}

template <class T>
KOKKOS_INLINE_FUNCTION T atomic_fetch_sub(
    volatile T *dest, typename std::enable_if<sizeof(T) != sizeof(int) &&
                                                  sizeof(T) == sizeof(int64_t),
                                              const T &>::type val) {
  union U {
    int64_t i;
    T t;
    KOKKOS_INLINE_FUNCTION U() {}
  } assume, oldval, newval;

  oldval.t = *dest;

  do {
    assume.i = oldval.i;
    newval.t = assume.t - val;
    oldval.i = atomic_compare_exchange((int64_t *)dest, assume.i, newval.i);
  } while (assume.i != oldval.i);

  return oldval.t;
}
template <class T>
KOKKOS_INLINE_FUNCTION T atomic_fetch_sub(
    volatile T *dest,
    typename std::enable_if<sizeof(T) == sizeof(char), T>::type val) {
  unsigned int oldval, newval, assume;
  oldval = *(int *)dest;

  do {
    assume = oldval;
    newval = assume & 0x7fffff00 + ((assume & 0xff) - val) & 0xff;
    oldval = atomicCAS(reinterpret_cast<unsigned int *>(dest), assume, newval);
  } while (assume != oldval);

  return (T)oldval & 0xff;
}

template <class T>
KOKKOS_INLINE_FUNCTION T atomic_fetch_sub(
    volatile T *dest,
    typename std::enable_if<sizeof(T) == sizeof(short), T>::type val) {
  unsigned int oldval, newval, assume;
  oldval = *(int *)dest;

  do {
    assume = oldval;
    newval = assume & 0x7fff0000 + ((assume & 0xffff) - val) & 0xffff;
    oldval = atomicCAS(reinterpret_cast<unsigned int *>(dest), assume, newval);
  } while (assume != oldval);

  return (T)oldval & 0xffff;
}

// KOKKOS_INLINE_FUNCTION
// int atomic_fetch_or(volatile int *const dest, int const val) {
//  return atomicOr(const_cast<int *>(dest), val);
//}

// KOKKOS_INLINE_FUNCTION
// unsigned int atomic_fetch_or(volatile unsigned int *const dest,
//                             unsigned int const val) {
//  return atomicOr(const_cast<unsigned int *>(dest), val);
//}

KOKKOS_INLINE_FUNCTION
unsigned long long int atomic_fetch_or(
    volatile unsigned long long int *const dest,
    unsigned long long int const val) {
  return atomicOr(const_cast<unsigned long long int *>(dest), val);
}

// KOKKOS_INLINE_FUNCTION
// int atomic_fetch_and(volatile int *const dest, int const val) {
//  return atomicAnd(const_cast<int *>(dest), val);
//}

// KOKKOS_INLINE_FUNCTION
// unsigned int atomic_fetch_and(volatile unsigned int *const dest,
//                              unsigned int const val) {
//  return atomicAnd(const_cast<unsigned int *>(dest), val);
//}

KOKKOS_INLINE_FUNCTION
unsigned long long int atomic_fetch_and(
    volatile unsigned long long int *const dest,
    unsigned long long int const val) {
  return atomicAnd(const_cast<unsigned long long int *>(dest), val);
}
}  // namespace Kokkos
#endif

#endif
