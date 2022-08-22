cmake_minimum_required(VERSION "3.24")

project(
  kalman
  VERSION "0.1.0"
  DESCRIPTION "Kalman Filter for C++"
  LANGUAGES "CXX")

include(CTest)
include(FetchContent)

install(DIRECTORY "include/fcarouge" DESTINATION "include")

FetchContent_Declare(
  eigen
  GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git"
  GIT_TAG "3147391d946bb4b6c68edd901f2add6ac1f31f8c" # 3.4.0
)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY "https://github.com/fmtlib/fmt.git"
  GIT_TAG "c4ee726532178e556d923372f29163bd206d7732" # 9.0.0
)

FetchContent_MakeAvailable(eigen fmt)

add_library(kalman INTERFACE)

add_library(main "source/main.cpp")

target_include_directories(kalman INTERFACE "include"
                                            "support/include/fcarouge")

add_executable(
  kalman_test_driver
  "test/eigen/f.cpp"
  "test/eigen/h.cpp"
  "test/eigen/initialization.cpp"
  "test/f.cpp"
  "test/format.cpp"
  "test/h.cpp"
  "test/initialization.cpp")

target_link_libraries(kalman_test_driver PRIVATE eigen fmt kalman main)

set_target_properties(
  kalman_test_driver
  PROPERTIES CXX_STANDARD "23"
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF)

add_test(NAME kalman_test COMMAND kalman_test_driver)
