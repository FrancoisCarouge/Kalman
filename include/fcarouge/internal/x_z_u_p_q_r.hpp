/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.4.0
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

For more information, please refer to <https://unlicense.org> */

#ifndef FCAROUGE_INTERNAL_X_Z_U_P_Q_R_HPP
#define FCAROUGE_INTERNAL_X_Z_U_P_Q_R_HPP

#include "fcarouge/utility.hpp"
#include "function.hpp"

#include <tuple>

namespace fcarouge::internal {
template <typename Type> struct x_z_u_p_q_r {
  using state = Type;
  using output = Type;
  using input = Type;
  using estimate_uncertainty = ᴀʙᵀ<state, state>;
  using process_uncertainty = ᴀʙᵀ<state, state>;
  using output_uncertainty = ᴀʙᵀ<output, output>;
  using gain = ᴀʙᵀ<state, output>;
  using innovation = evaluate<difference<output, output>>;
  using innovation_uncertainty = output_uncertainty;

  static inline const auto i{one<ᴀʙᵀ<state, state>>};

  state x{zero<state>};
  estimate_uncertainty p{one<estimate_uncertainty>};
  process_uncertainty q{zero<process_uncertainty>};
  output_uncertainty r{zero<output_uncertainty>};
  input u{zero<input>};
  gain k{one<gain>};
  innovation y{zero<innovation>};
  innovation_uncertainty s{one<innovation_uncertainty>};
  output z{zero<output>};

  inline constexpr void update(const auto &output_z, const auto &...outputs_z) {
    z = output{output_z, outputs_z...};
    s = p + r;
    k = p / s;
    y = z - x;
    x = x + k * y;
    p = (i - k) * p * t(i - k) + k * r * t(k);
  }

  inline constexpr void predict(const auto &input_u, const auto &...inputs_u) {
    u = input{input_u, inputs_u...};
    x = x + u;
    p = p + q;
  }
};
} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_X_Z_U_P_Q_R_HPP
