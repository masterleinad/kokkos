# Create tests that contain each header separately. We do not  run these tests
# but we just try to compile them.

 # Globbing all the header filenames to test for self-containment and presence of header guards
file(GLOB KOKKOS_CORE_HEADERS RELATIVE ${CMAKE_SOURCE_DIR}/core/src
     ${CMAKE_SOURCE_DIR}/core/src/*.hpp ${CMAKE_SOURCE_DIR}/core/src/*.h)
file(GLOB KOKKOS_CONTAINERS_HEADERS RELATIVE ${CMAKE_SOURCE_DIR}/containers/src
     ${CMAKE_SOURCE_DIR}/containers/src/*.hpp)
file(GLOB KOKKOS_ALGORITHMS_HEADERS RELATIVE  ${CMAKE_SOURCE_DIR}/algorithms/src
     ${CMAKE_SOURCE_DIR}/algorithms/src/*.hpp)

MESSAGE(STATUS "KOKKOS_CORE_HEADERS: ${CMAKE_SOURCE_DIR} ${KOKKOS_CORE_HEADERS}")
MESSAGE(STATUS "KOKKOS_CONTAINERS_HEADERS: ${CMAKE_SOURCE_DIR} ${KOKKOS_CONTAINERS_HEADERS}")
MESSAGE(STATUS "KOKKOS_ALGORITHMS_HEADERS: ${CMAKE_SOURCE_DIR} ${KOKKOS_ALGORITHMS_HEADERS}")

foreach (_header ${KOKKOS_CORE_HEADERS} ${KOKKOS_CONTAINERS_HEADERS} ${KOKKOS_ALGORITHMS_HEADERS})
  string(REGEX REPLACE "[\./]" "_" header_test_name ${_header})
  set(header_test_name Kokkos_HeaderSelfContained_${header_test_name})
  add_executable(${header_test_name} tstHeader.cpp)
  target_link_libraries(${header_test_name} PRIVATE kokkos)
  target_compile_definitions(${header_test_name} PRIVATE KOKKOS_HEADER_TEST_NAME=${_header})
endforeach()
