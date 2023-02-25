/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.2.0
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

#ifndef FCAROUGE_INTERNAL_KALMAN_HPP
#define FCAROUGE_INTERNAL_KALMAN_HPP

#include "fcarouge/utility.hpp"

#include <functional>
#include <tuple>

namespace fcarouge::internal {

template <typename Update, typename Predict> struct kalman {
  //! @todo Verify the update's and predict's states are the same?
  using state = typename Update::state;
  using output = typename Update::output;
  using input = typename Predict::input;
  using estimate_uncertainty = typename Update::estimate_uncertainty;
  using process_uncertainty = typename Predict::process_uncertainty;
  using output_uncertainty = typename Update::output_uncertainty;
  using state_transition = typename Predict::state_transition;
  using output_model = typename Update::output_model;
  using input_control = typename Predict::input_control;
  using gain = typename Update::gain;
  using innovation = typename Update::innovation;
  using innovation_uncertainty = typename Update::innovation_uncertainty;

  state x{zero_v<state>};
  estimate_uncertainty p{identity_v<estimate_uncertainty>};
  Update updator{x, p};
  Predict predictor{x, p};

  inline constexpr void update(const auto &...arguments) {
    updator(arguments...);
  }

  inline constexpr void predict(const auto &...arguments) {
    predictor(arguments...);
  }
};

} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_KALMAN_HPP
