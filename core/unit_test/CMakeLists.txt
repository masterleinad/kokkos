#
# Add test-only library for gtest to be reused by all the subpackages
#


SET(GTEST_SOURCE_DIR ${${PARENT_PACKAGE_NAME}_SOURCE_DIR}/tpls/gtest)

#need here for tribits
KOKKOS_INCLUDE_DIRECTORIES(${GTEST_SOURCE_DIR})
KOKKOS_ADD_TEST_LIBRARY(
  kokkos_gtest
  HEADERS ${GTEST_SOURCE_DIR}/gtest/gtest.h
  SOURCES ${GTEST_SOURCE_DIR}/gtest/gtest-all.cc
)
#These can be direct, no need for Tribits or Kokkos wrappers

# WORKAROUND FOR HIPCC
IF(Kokkos_ENABLE_HIP)
  TARGET_COMPILE_DEFINITIONS(kokkos_gtest PUBLIC "-DGTEST_HAS_PTHREAD=0 --amdgpu-target=gfx906")
ELSE()
  TARGET_COMPILE_DEFINITIONS(kokkos_gtest PUBLIC "-DGTEST_HAS_PTHREAD=0")
ENDIF()

TARGET_INCLUDE_DIRECTORIES(kokkos_gtest PUBLIC ${GTEST_SOURCE_DIR})
#Gtest minimally requires C++11
TARGET_COMPILE_FEATURES(kokkos_gtest PUBLIC cxx_std_11)

#
# Define Incremental Testing Feature Levels
# Define Device name mappings (i.e. what comes after Kokkos:: for the ExecSpace)
#

SET(KOKKOS_CUDA_FEATURE_LEVEL 999)
SET(KOKKOS_CUDA_NAME Cuda)
SET(KOKKOS_HIP_FEATURE_LEVEL 12)
SET(KOKKOS_HIP_NAME Experimental::HIP)
SET(KOKKOS_HPX_FEATURE_LEVEL 999)
SET(KOKKOS_HPX_NAME Experimental::HPX)
SET(KOKKOS_OPENMP_FEATURE_LEVEL 999)
SET(KOKKOS_OPENMP_NAME OpenMP)
SET(KOKKOS_OPENMPTARGET_FEATURE_LEVEL 10)
SET(KOKKOS_OPENMPTARGET_NAME Experimental::OpenMPTarget)
SET(KOKKOS_SERIAL_FEATURE_LEVEL 999)
SET(KOKKOS_SERIAL_NAME Serial)
SET(KOKKOS_THREADS_FEATURE_LEVEL 999)
SET(KOKKOS_THREADS_NAME Threads)


#
# Define the tests
#

#I will leave these alone for now because I don't need transitive dependencies on tests
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})

foreach(Tag Threads;Serial;OpenMP;Cuda;HPX;OpenMPTarget;HIP)
  # Because there is always an exception to the rule
  if(Tag STREQUAL "Threads")
    set(DEVICE "PTHREAD")
  else()
    string(TOUPPER ${Tag} DEVICE)
  endif()
  string(TOLOWER ${Tag} dir)

  SET(${Tag}_SOURCES
      UnitTestMainInit.cpp
      ${dir}/Test${Tag}_AtomicOperations_int.cpp
      ${dir}/Test${Tag}_AtomicOperations_unsignedint.cpp
      ${dir}/Test${Tag}_AtomicOperations_longint.cpp
      ${dir}/Test${Tag}_AtomicOperations_unsignedlongint.cpp
      ${dir}/Test${Tag}_AtomicOperations_longlongint.cpp
      ${dir}/Test${Tag}_AtomicOperations_double.cpp
      ${dir}/Test${Tag}_AtomicOperations_float.cpp
      ${dir}/Test${Tag}_AtomicOperations_complexdouble.cpp
      ${dir}/Test${Tag}_AtomicOperations_complexfloat.cpp
      ${dir}/Test${Tag}_AtomicViews.cpp
      ${dir}/Test${Tag}_Atomics.cpp
      ${dir}/Test${Tag}_Concepts.cpp
      ${dir}/Test${Tag}_Complex.cpp
      ${dir}/Test${Tag}_Crs.cpp
      ${dir}/Test${Tag}_DeepCopyAlignment.cpp
      ${dir}/Test${Tag}_FunctorAnalysis.cpp
      ${dir}/Test${Tag}_Init.cpp
      ${dir}/Test${Tag}_LocalDeepCopy.cpp
      ${dir}/Test${Tag}_MDRange_a.cpp
      ${dir}/Test${Tag}_MDRange_b.cpp
      ${dir}/Test${Tag}_MDRange_c.cpp
      ${dir}/Test${Tag}_MDRange_d.cpp
      ${dir}/Test${Tag}_MDRange_e.cpp
      ${dir}/Test${Tag}_Other.cpp
      ${dir}/Test${Tag}_RangePolicy.cpp
      ${dir}/Test${Tag}_RangePolicyRequire.cpp
      ${dir}/Test${Tag}_Reductions.cpp
      ${dir}/Test${Tag}_Reducers_a.cpp
      ${dir}/Test${Tag}_Reducers_b.cpp
      ${dir}/Test${Tag}_Reducers_c.cpp
      ${dir}/Test${Tag}_Reducers_d.cpp
      ${dir}/Test${Tag}_Reductions_DeviceView.cpp
      ${dir}/Test${Tag}_Scan.cpp
      ${dir}/Test${Tag}_SharedAlloc.cpp
      ${dir}/Test${Tag}_SubView_a.cpp
      ${dir}/Test${Tag}_SubView_b.cpp
      ${dir}/Test${Tag}_SubView_c01.cpp
      ${dir}/Test${Tag}_SubView_c02.cpp
      ${dir}/Test${Tag}_SubView_c03.cpp
      ${dir}/Test${Tag}_SubView_c04.cpp
      ${dir}/Test${Tag}_SubView_c05.cpp
      ${dir}/Test${Tag}_SubView_c06.cpp
      ${dir}/Test${Tag}_SubView_c07.cpp
      ${dir}/Test${Tag}_SubView_c08.cpp
      ${dir}/Test${Tag}_SubView_c09.cpp
      ${dir}/Test${Tag}_SubView_c10.cpp
      ${dir}/Test${Tag}_SubView_c11.cpp
      ${dir}/Test${Tag}_SubView_c12.cpp
      ${dir}/Test${Tag}_SubView_c13.cpp
      ${dir}/Test${Tag}_Team.cpp
      ${dir}/Test${Tag}_TeamReductionScan.cpp
      ${dir}/Test${Tag}_TeamScratch.cpp
      ${dir}/Test${Tag}_TeamTeamSize.cpp
      ${dir}/Test${Tag}_TeamVectorRange.cpp
      ${dir}/Test${Tag}_UniqueToken.cpp
      ${dir}/Test${Tag}_ViewAPI_a.cpp
      ${dir}/Test${Tag}_ViewAPI_b.cpp
      ${dir}/Test${Tag}_ViewAPI_c.cpp
      ${dir}/Test${Tag}_ViewAPI_d.cpp
      ${dir}/Test${Tag}_ViewAPI_e.cpp
      ${dir}/Test${Tag}_ViewLayoutStrideAssignment.cpp
      ${dir}/Test${Tag}_ViewMapping_a.cpp
      ${dir}/Test${Tag}_ViewMapping_b.cpp
      ${dir}/Test${Tag}_ViewMapping_subview.cpp
      ${dir}/Test${Tag}_ViewOfClass.cpp
      ${dir}/Test${Tag}_WorkGraph.cpp
      ${dir}/Test${Tag}_View_64bit.cpp
      ${dir}/Test${Tag}_ViewResize.cpp
  )
endforeach()

add_subdirectory(headers_self_contained)
