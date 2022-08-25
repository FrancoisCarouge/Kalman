cmake_minimum_required(VERSION "3.24")

project(
  kalman
  VERSION "0.1.0"
  DESCRIPTION "Kalman Filter for C++"
  LANGUAGES "CXX")

set(CMAKE_CXX_STANDARD "23" CMAKE_CXX_STANDARD_REQUIRED ON CMAKE_CXX_EXTENSIONS
                       OFF)

include(CTest)
include(FetchContent)

install(DIRECTORY "include/fcarouge" DESTINATION "include")

FetchContent_Declare(
  eigen # 3.4.0
  URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2"
  URL_HASH MD5=132dde48fe2b563211675626d29f1707)

FetchContent_Declare(
  fmt # 9.0.0
  URL "https://github.com/fmtlib/fmt/archive/refs/tags/9.0.0.tar.gz"
  URL_HASH MD5=d56c8b0612b049bb1854f07c8b133f3c)

FetchContent_MakeAvailable(eigen fmt)

add_library(kalman INTERFACE)

add_library(kalman_main "source/main.cpp")

target_include_directories(kalman INTERFACE "include"
                                            "support/include/fcarouge")

set(KALMAN_TEST_LIST
    "test/eigen_f.cpp"
    "test/eigen_h.cpp"
    "test/eigen_initialization.cpp"
    "test/f.cpp"
    "test/format.cpp"
    "test/h.cpp"
    "test/initialization.cpp")

foreach(test ${KALMAN_TEST_LIST})
  get_filename_component(test_name ${test} NAME_WE)
  add_executable(kalman_${test_name}_driver ${test})
  target_link_libraries(kalman_${test_name}_driver PRIVATE eigen fmt kalman
                                                           kalman_main)
  add_test(kalman_test_${test_name} kalman_${test_name}_driver)
endforeach()

set(KALMAN_SAMPLE_LIST
    "sample/ardupilot_soaring.cpp" "sample/building_height.cpp"
    "sample/dog_position.cpp" "sample/liquid_temperature.cpp"
    "sample/rocket_altitude.cpp" "sample/vehicle_location.cpp")

foreach(sample ${KALMAN_SAMPLE_LIST})
  get_filename_component(sample_name ${sample} NAME_WE)
  add_executable(kalman_${sample_name}_driver ${sample})
  target_link_libraries(kalman_${sample_name}_driver PRIVATE eigen fmt kalman
                                                             kalman_main)
  add_test(kalman_sample_${sample_name} kalman_${sample_name}_driver)
endforeach()
