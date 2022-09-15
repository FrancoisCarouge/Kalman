set(SOURCES
    "ekf_4x1x0_ardupilot_soaring.cpp"
    "kf_1x1x0_building_height.cpp"
    "kf_1x1x0_liquid_temperature.cpp"
    "kf_1x1x1_dog_position.cpp"
    "kf_2x1x1_rocket_altitude.cpp"
    "kf_6x2x0_vehicle_location.cpp"
    "kf_8x4x0_deep_sort_bounding_box.cpp")

foreach(sample ${SOURCES})
  get_filename_component(sample_name ${sample} NAME_WE)
  add_executable(kalman_${sample_name}_driver ${sample})
  target_link_libraries(kalman_${sample_name}_driver PRIVATE eigen kalman main)
  add_test(kalman_sample_${sample_name} kalman_${sample_name}_driver)
endforeach()
