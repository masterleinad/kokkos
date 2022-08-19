INCLUDE(CheckIncludeFileCXX)
CHECK_INCLUDE_FILE_CXX(oneapi/dpl/execution KOKKOS_COMPILER_HAS_ONEDPL_EXECUTION_HEADER)
CHECK_INCLUDE_FILE_CXX(oneapi/dpl/algorithm KOKKOS_COMPILER_HAS_ONEDPL_ALGORITHM_HEADER)

IF (KOKKOS_COMPILER_HAS_ONEDPL_EXECUTION_HEADER AND KOKKOS_COMPILER_HAS_ONEDPL_ALGORITHM_HEADER)
  KOKKOS_CREATE_IMPORTED_TPL(ONEDPL INTERFACE)
ELSE()
  FIND_PACKAGE(oneDPL REQUIRED)
  KOKKOS_CREATE_IMPORTED_TPL(
    ONEDPL INTERFACE
    LINK_LIBRARIES oneDPL
    COMPILE_DEFINITIONS PSTL_USE_PARALLEL_POLICIES=0 _GLIBCXX_USE_TBB_PAR_BACKEND=0
  )
ENDIF()
