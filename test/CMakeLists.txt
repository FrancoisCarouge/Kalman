set(SOURCES
    "eigen_f.cpp"
    "eigen_h.cpp"
    "eigen_initialization.cpp"
    "f.cpp"
    "format.cpp"
    "h.cpp"
    "initialization.cpp")

foreach(test ${SOURCES})
  get_filename_component(test_name ${test} NAME_WE)
  add_executable(kalman_${test_name}_driver ${test})
  target_link_libraries(kalman_${test_name}_driver PRIVATE eigen kalman main)
  add_test(kalman_test_${test_name} kalman_${test_name}_driver)
endforeach()
