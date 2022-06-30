cmake_minimum_required(
   VERSION 3.20
)

project(kalman 
   VERSION 0.1.0
   LANGUAGES CXX
)

include(CTest)

add_library(${PROJECT_NAME}
   INTERFACE
)

target_include_directories(${PROJECT_NAME}
   INTERFACE
      ${PROJECT_SOURCE_DIR}/include/fcarouge
)

set_target_properties(${PROJECT_NAME}
   PROPERTIES
      CXX_STANDARD 23
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
)

install(
   DIRECTORY include/fcarouge
   DESTINATION include
)
