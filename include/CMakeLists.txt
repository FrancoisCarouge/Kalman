add_library(kalman INTERFACE)
target_include_directories(kalman INTERFACE ".")
target_link_libraries(kalman INTERFACE format)

target_compile_options(
  kalman
  INTERFACE "-ansi"
            "-fno-check-new"
            "-fno-common"
            "-fstrict-aliasing"
            "-pedantic"
            "-Wall"
            "-Wcast-align"
            "-Wchar-subscripts"
            "-Wdouble-promotion"
            "-Wenum-conversion"
            "-Werror"
            "-Wextra"
            "-Wformat-security"
            "-Wno-long-long"
            "-Wno-psabi"
            "-Wno-variadic-macros"
            "-Wnon-virtual-dtor"
            "-Wpointer-arith"
            "-Wshadow"
            "-Wundef"
            "-Wunused-local-typedefs"
            "-Wwrite-strings")
