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

#ifndef FCAROUGE_INTERNAL_X_Z_HPP
#define FCAROUGE_INTERNAL_X_Z_HPP

#include "fcarouge/utility.hpp"

namespace fcarouge::internal {
template <typename State, typename Output> struct x_z_p_r_f {
  using state = State;
  using output = Output;
  using estimate_uncertainty = deduce_matrix<state, state>;
  using output_uncertainty = deduce_matrix<output, output>;
  using state_transition = deduce_matrix<state, state>;
  using gain = deduce_matrix<state, output>;
  using innovation = output;
  using innovation_uncertainty = output_uncertainty;

  static inline const auto i{identity_v<deduce_matrix<state, state>>};

  state x{zero_v<state>};
  estimate_uncertainty p{identity_v<estimate_uncertainty>};
  output_uncertainty r{zero_v<output_uncertainty>};
  state_transition f{identity_v<state_transition>};
  gain k{identity_v<gain>};
  innovation y{zero_v<innovation>};
  innovation_uncertainty s{identity_v<innovation_uncertainty>};
  output z{zero_v<output>};
  transposer t{};

  inline constexpr void update(const output &output_z) {
    z = output_z;
    s = p + r;
    k = p / s;
    y = z - x;
    x = x + k * y;
    p = (i - k) * p * t(i - k) + k * r * t(k);
  }

  inline constexpr void predict() {
    x = f * x;
    p = f * p * t(f);
  }
};
} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_X_Z_HPP