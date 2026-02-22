/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.3
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

#include "fcarouge/kalman.hpp"
#include "fcarouge/linalg.hpp"

#include <cassert>
#include <cmath>

namespace fcarouge::sample {
namespace {
template <auto Size> using vector = column_vector<float, Size>;
using state = fcarouge::state<vector<8>>;

//! @brief Estimating the position of bounding boxes in image space.
//!
//! @copyright This example is transcribed from Nwojke's Deep SORT filter.
//!
//! @see https://github.com/nwojke/deep_sort
//!
//! @details In this example, we would like to estimate the bounding box center
//! position `x`, `y`, aspect ratio `a`, height `h`, and their respective
//! velocities in image space. The filter models constant velocity dynamics.
//! The prediction and observation models are linear.
//!
//! @note For information, the original sample appears to saturate the velocity
//! precision early on.
//!
//! @example kf_8x4x0_deep_sort_bounding_box.cpp
[[maybe_unused]] auto sample{[] {
  const vector<4> initial_box{605.0F, 248.0F, 0.20481927710843373F, 332.0F};
  // Experimental, tunable position and velocity uncertainty standard deviation
  // weights.
  const float position_weight{1.F / 20.F};
  const float velocity_weight{1.F / 160.F};

  // Constant velocity, linear state transition model. From one image frame to
  // the other.
  const float delta_time{1.F};

  // A 8x4x0 filter, constant velocity, linear.
  kalman filter{
      // The filter is initialized with the first observed output.
      // Bounding box position and velocity estimated state X: [px,
      // py, pa, ph, vx, vy, va, vh].
      state{initial_box(0), initial_box(1), initial_box(2), initial_box(3), 0.F,
            0.F, 0.F, 0.F},
      // The output Z:
      output<vector<4>>,
      // The estimate uncertainty P:
      estimate_uncertainty{[&position_weight, &velocity_weight,
                            &initial_box]() {
        matrix<float, 8, 8> value{kalman_internal::zero<matrix<float, 8, 8>>};
        value(0, 0) = std::powf(2.F * position_weight * initial_box(3), 2);
        value(1, 1) = std::powf(2.F * position_weight * initial_box(3), 2);
        value(2, 2) = std::powf(1e-2F, 2);
        value(3, 3) = std::powf(2.F * position_weight * initial_box(3), 2);
        value(4, 4) = std::powf(10.F * velocity_weight * initial_box(3), 2);
        value(5, 5) = std::powf(10.F * velocity_weight * initial_box(3), 2);
        value(6, 6) = std::powf(1e-5F, 2);
        value(7, 7) = std::powf(10.F * velocity_weight * initial_box(3), 2);
        return value;
      }()},
      // Q
      process_uncertainty{[](const state::type &x) {
        const float weight_position{1.F / 20.F};
        const float weight_velocity{1.F / 160.F};
        matrix<float, 8, 8> value{kalman_internal::zero<matrix<float, 8, 8>>};
        value(0, 0) = std::powf(weight_position * x(3), 2);
        value(1, 1) = std::powf(weight_position * x(3), 2);
        value(2, 2) = std::powf(1e-2F, 2);
        value(3, 3) = std::powf(weight_position * x(3), 2);
        value(4, 4) = std::powf(weight_velocity * x(3), 2);
        value(5, 5) = std::powf(weight_velocity * x(3), 2);
        value(6, 6) = std::powf(1e-5F, 2);
        value(7, 7) = std::powf(weight_velocity * x(3), 2);
        return value;
      }},
      // R
      output_uncertainty{
          // Observation, measurement noise covariance.
          [](const state::type &x, [[maybe_unused]] const vector<4> &z) {
            const float weight_position{1.F / 20.F};
            matrix<float, 4, 4> value{
                kalman_internal::zero<matrix<float, 4, 4>>};
            value(0, 0) = std::powf(weight_position * x(3), 2);
            value(1, 1) = std::powf(weight_position * x(3), 2);
            value(2, 2) = std::powf(1e-1F, 2);
            value(3, 3) = std::powf(weight_position * x(3), 2);
            return value;
          }},
      // The state transition F:
      state_transition{{1.F, 0.F, 0.F, 0.F, delta_time, 0.F, 0.F, 0.F},
                       {0.F, 1.F, 0.F, 0.F, 0.F, delta_time, 0.F, 0.F},
                       {0.F, 0.F, 1.F, 0.F, 0.F, 0.F, delta_time, 0.F},
                       {0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, delta_time},
                       {0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F},
                       {0.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F},
                       {0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F},
                       {0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 1.F}}};

  // A hundred bounding box output measurements `(x, y, a, h)` from Deep SORT's
  // MOT16 sample, tracker #201.
  const vector<4> measured[]{
#include "data/kf_8x4x0_deep_sort_bounding_box.csv"
  };

  // Now we can predict the next state based on the initialization values.
  filter.predict();

  // And so on, run a step of the filter, updating and predicting, every frame.
  for (const auto &measure : measured) {
    filter.update(measure);
    filter.predict();
  }

  assert(std::abs(1 - filter.x()[0] / 370.932041394761F) < 0.001F &&
         std::abs(1 - filter.x()[1] / 251.173174229878F) < 0.001F &&
         std::abs(1 - filter.x()[2] / 0.314757138075364F) < 0.001F &&
         std::abs(1 - filter.x()[3] / 287.859996019444F) < 0.001F &&
         std::abs(1 - filter.x()[4] / 1.95865368159518F) < 0.001F &&
         std::abs(1 - filter.x()[5] / 0.229282868701086F) < 0.001F &&
         // The precision of the velocity appears to saturate early on in the
         // original example. The parameter could be scaled or larger types used
         // to improve comparison accuracy.
         std::abs(1 - filter.x()[6] / 2.46138628550094E-06F) < 0.5F &&
         std::abs(1 - filter.x()[7] / 0.81402529074969F) < 0.001F &&
         "The estimated states expected to meet Nwojke's Deep SORT filter's "
         "MOT16 sample tracker #201 dataset at 0.1% accuracy.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
