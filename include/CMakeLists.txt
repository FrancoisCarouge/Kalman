#[[ __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.2
https://github.com/FrancoisCarouge/Kalman

SPDX-License-Identifier: Unlicense

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org> ]]

add_library(kalman INTERFACE)
target_sources(
  kalman
  INTERFACE FILE_SET
            "kalman_headers"
            TYPE
            "HEADERS"
            FILES
            "fcarouge/kalman_forward.hpp"
            "fcarouge/kalman_internal/factory.hpp"
            "fcarouge/kalman_internal/format.hpp"
            "fcarouge/kalman_internal/function.hpp"
            "fcarouge/kalman_internal/kalman.tpp"
            "fcarouge/kalman_internal/printer.hpp"
            "fcarouge/kalman_internal/type.hpp"
            "fcarouge/kalman_internal/utility.hpp"
            "fcarouge/kalman_internal/x_z_p_q_r_h_f.hpp"
            "fcarouge/kalman_internal/x_z_p_q_r_hh_us_ps.hpp"
            "fcarouge/kalman_internal/x_z_p_q_r.hpp"
            "fcarouge/kalman_internal/x_z_p_qq_rr_f.hpp"
            "fcarouge/kalman_internal/x_z_p_r_f.hpp"
            "fcarouge/kalman_internal/x_z_p_r.hpp"
            "fcarouge/kalman_internal/x_z_u_p_q_r_f_g_ps.hpp"
            "fcarouge/kalman_internal/x_z_u_p_q_r_h_f_g_us_ps.hpp"
            "fcarouge/kalman_internal/x_z_u_p_q_r.hpp"
            "fcarouge/kalman.hpp")
install(
  TARGETS kalman
  EXPORT "fcarouge-kalman-target"
  FILE_SET "kalman_headers")

# Conditionally provide the namespace alias target which may be an imported
# target from a package, or an aliased target if built as part of the same
# buildsystem.
if(NOT TARGET fcarouge-kalman::kalman)
  add_library(fcarouge-kalman::kalman ALIAS kalman)
endif()
