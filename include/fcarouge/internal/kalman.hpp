/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter for C++
Version 0.1.0
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

#include "utility.hpp"

#include <functional>
#include <tuple>
#include <type_traits>

namespace fcarouge::internal {

template <typename PredictionModel, typename UpdateModel> struct kalman final {
  using state = typename PredictionModel::state;
  using output = typename UpdateModel::output;
  using input = typename PredictionModel::input;
  using estimate_uncertainty = typename PredictionModel::estimate_uncertainty;
  using process_uncertainty = typename PredictionModel::process_uncertainty;
  using output_uncertainty = typename UpdateModel::output_uncertainty;
  using state_transition = typename PredictionModel::state_transition;
  using output_model = typename UpdateModel::output_model;
  using input_control = typename PredictionModel::input_control;
  using gain = typename UpdateModel::gain;
  using innovation = typename UpdateModel::innovation;
  using innovation_uncertainty = typename UpdateModel::innovation_uncertainty;
  using observation_state_function =
      typename UpdateModel::observation_state_function;
  using noise_observation_function =
      typename UpdateModel::noise_observation_function;
  using transition_state_function =
      typename PredictionModel::transition_state_function;
  using noise_process_function =
      typename PredictionModel::noise_process_function;
  using transition_control_function =
      typename PredictionModel::transition_control_function;
  using transition_function = typename PredictionModel::transition_function;
  using observation_function = typename UpdateModel::observation_function;

  PredictionModel predictor;
  UpdateModel updator;
  state x{internal::zero_v<state>};
  estimate_uncertainty p{internal::identity_v<estimate_uncertainty>};

  inline constexpr void predict(const auto &...arguments) {
    std::tie(x, p) = predictor(x, p, arguments...);
  }

  inline constexpr void update(const auto &...arguments) {
    std::tie(x, p) = updator(x, p, arguments...);
  }
};

} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_KALMAN_HPP
