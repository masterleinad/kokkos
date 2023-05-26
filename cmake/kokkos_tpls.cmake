KOKKOS_CFG_DEPENDS(TPLS OPTIONS)
KOKKOS_CFG_DEPENDS(TPLS DEVICES)
KOKKOS_CFG_DEPENDS(TPLS COMPILER_ID)

FUNCTION(KOKKOS_TPL_OPTION PKG DEFAULT)
  CMAKE_PARSE_ARGUMENTS(PARSED
    ""
    ""
    ""
    ${ARGN})

  KOKKOS_ENABLE_OPTION(${PKG} ${DEFAULT} "Whether to enable the ${PKG} library")
  KOKKOS_OPTION(${PKG}_DIR "" PATH "Location of ${PKG} library")
  SET(KOKKOS_ENABLE_${PKG} ${KOKKOS_ENABLE_${PKG}} PARENT_SCOPE)
  SET(KOKKOS_${PKG}_DIR  ${KOKKOS_${PKG}_DIR} PARENT_SCOPE)
ENDFUNCTION()

KOKKOS_TPL_OPTION(HWLOC   Off)
KOKKOS_TPL_OPTION(MEMKIND Off)
IF(KOKKOS_ENABLE_MEMKIND)
  SET(KOKKOS_ENABLE_HBWSPACE ON)
ENDIF()
KOKKOS_TPL_OPTION(CUDA    ${Kokkos_ENABLE_CUDA})
KOKKOS_TPL_OPTION(LIBRT   Off)
IF(KOKKOS_ENABLE_HIP AND NOT KOKKOS_CXX_COMPILER_ID STREQUAL HIPCC)
  SET(ROCM_DEFAULT ON)
ELSE()
  SET(ROCM_DEFAULT OFF)
ENDIF()
KOKKOS_TPL_OPTION(ROCM    ${ROCM_DEFAULT})
KOKKOS_TPL_OPTION(ONEDPL  ${Kokkos_ENABLE_SYCL})

IF (WIN32)
  SET(LIBDL_DEFAULT Off)
ELSE()
  SET(LIBDL_DEFAULT On)
ENDIF()
KOKKOS_TPL_OPTION(LIBDL ${LIBDL_DEFAULT})

KOKKOS_TPL_OPTION(HPX OFF)

KOKKOS_TPL_OPTION(THREADS ${Kokkos_ENABLE_THREADS})

KOKKOS_TPL_OPTION(LIBQUADMATH OFF)

#Make sure we use our local FindKokkosCuda.cmake
KOKKOS_IMPORT_TPL(HPX INTERFACE)
IF (NOT KOKKOS_ENABLE_COMPILE_AS_CMAKE_LANGUAGE)
  KOKKOS_IMPORT_TPL(CUDA INTERFACE)
ENDIF()
KOKKOS_IMPORT_TPL(HWLOC)
KOKKOS_IMPORT_TPL(LIBRT)
KOKKOS_IMPORT_TPL(LIBDL)
KOKKOS_IMPORT_TPL(MEMKIND)
IF (NOT WIN32)
  KOKKOS_IMPORT_TPL(THREADS INTERFACE)
ENDIF()
IF (NOT KOKKOS_ENABLE_COMPILE_AS_CMAKE_LANGUAGE)
  KOKKOS_IMPORT_TPL(ROCM INTERFACE)
  KOKKOS_IMPORT_TPL(ONEDPL INTERFACE)
ENDIF()
KOKKOS_IMPORT_TPL(LIBQUADMATH)

IF (Kokkos_ENABLE_DESUL_ATOMICS_EXTERNAL)
  find_package(desul REQUIRED COMPONENTS atomics)
  KOKKOS_EXPORT_CMAKE_TPL(desul REQUIRED COMPONENTS atomics)
ENDIF()

if (Kokkos_ENABLE_IMPL_MDSPAN AND Kokkos_ENABLE_MDSPAN_EXTERNAL)
  find_package(mdspan REQUIRED)
  KOKKOS_EXPORT_CMAKE_TPL(mdspan REQUIRED)
endif()

IF (Kokkos_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  KOKKOS_EXPORT_CMAKE_TPL(OpenMP REQUIRED)
ENDIF()

#Convert list to newlines (which CMake doesn't always like in cache variables)
STRING(REPLACE ";" "\n" KOKKOS_TPL_EXPORT_TEMP "${KOKKOS_TPL_EXPORTS}")
#Convert to a regular variable
UNSET(KOKKOS_TPL_EXPORTS CACHE)
SET(KOKKOS_TPL_EXPORTS ${KOKKOS_TPL_EXPORT_TEMP})
IF (KOKKOS_ENABLE_MEMKIND)
   SET(KOKKOS_ENABLE_HBWSPACE)
   LIST(APPEND KOKKOS_MEMSPACE_LIST HBWSpace)
ENDIF()
