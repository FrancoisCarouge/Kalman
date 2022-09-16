add_library(format INTERFACE)
target_include_directories(format INTERFACE ".")
target_link_libraries(format INTERFACE fmt)
