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
  // Experimental position and velocity uncertainty standard deviation
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

  // TODO: Move sim out.
  // A hundred bounding box output measurements `(x, y, a, h)` from Deep SORT's
  // MOT16 sample, tracker #201.
  const vector<4> measured[]{{603.5F, 251.5F, 0.187335092348285F, 379.F},
                             {599.F, 241.F, 0.24390243902439F, 328.F},
                             {599.F, 239.5F, 0.257234726688103F, 311.F},
                             {602.5F, 244.F, 0.240131578947368F, 304.F},
                             {598.F, 248.5F, 0.272425249169435F, 301.F},
                             {596.5F, 240.5F, 0.283276450511945F, 293.F},
                             {601.F, 227.F, 0.301587301587302F, 252.F},
                             {603.5F, 235.5F, 0.282868525896414F, 251.F},
                             {602.F, 242.5F, 0.292490118577075F, 253.F},
                             {602.5F, 253.F, 0.218562874251497F, 334.F},
                             {593.F, 254.F, 0.273291925465838F, 322.F},
                             {603.F, 264.F, 0.22360248447205F, 322.F},
                             {600.5F, 278.5F, 0.198966408268734F, 387.F},
                             {593.F, 280.F, 0.237113402061856F, 388.F},
                             {588.5F, 269.F, 0.267195767195767F, 378.F},
                             {579.F, 260.F, 0.311111111111111F, 360.F},
                             {565.5F, 268.5F, 0.339130434782609F, 345.F},
                             {558.5F, 255.5F, 0.366568914956012F, 341.F},
                             {544.F, 268.F, 0.364705882352941F, 340.F},
                             {533.F, 258.5F, 0.356083086053412F, 337.F},
                             {519.F, 258.F, 0.353293413173653F, 334.F},
                             {511.5F, 252.5F, 0.333333333333333F, 333.F},
                             {515.5F, 252.5F, 0.31306990881459F, 329.F},
                             {523.5F, 251.F, 0.298192771084337F, 332.F},
                             {540.F, 252.5F, 0.318318318318318F, 333.F},
                             {574.F, 262.F, 0.344827586206897F, 348.F},
                             {590.5F, 265.F, 0.278735632183908F, 348.F},
                             {613.F, 268.F, 0.164556962025316F, 316.F},
                             {617.F, 260.5F, 0.161172161172161F, 273.F},
                             {615.5F, 261.5F, 0.15210355987055F, 309.F},
                             {605.5F, 259.F, 0.226351351351351F, 296.F},
                             {595.5F, 258.5F, 0.289036544850498F, 301.F},
                             {588.F, 257.5F, 0.350515463917526F, 291.F},
                             {579.5F, 254.F, 0.343537414965986F, 294.F},
                             {569.5F, 258.5F, 0.353535353535354F, 297.F},
                             {565.5F, 257.F, 0.37248322147651F, 298.F},
                             {555.F, 250.F, 0.388157894736842F, 304.F},
                             {546.5F, 249.F, 0.336666666666667F, 300.F},
                             {535.F, 251.F, 0.30718954248366F, 306.F},
                             {530.F, 246.F, 0.308724832214765F, 298.F},
                             {521.F, 252.F, 0.278145695364238F, 302.F},
                             {521.5F, 254.5F, 0.331010452961672F, 287.F},
                             {521.F, 258.5F, 0.32280701754386F, 285.F},
                             {519.5F, 255.F, 0.316326530612245F, 294.F},
                             {518.5F, 255.F, 0.304794520547945F, 292.F},
                             {511.F, 253.F, 0.310810810810811F, 296.F},
                             {506.F, 255.F, 0.319727891156463F, 294.F},
                             {499.F, 256.F, 0.352112676056338F, 284.F},
                             {492.5F, 256.5F, 0.349152542372881F, 295.F},
                             {489.5F, 257.F, 0.362068965517241F, 290.F},
                             {481.F, 251.5F, 0.357894736842105F, 285.F},
                             {474.F, 249.F, 0.324137931034483F, 290.F},
                             {466.F, 250.F, 0.306122448979592F, 294.F},
                             {461.5F, 248.F, 0.304794520547945F, 292.F},
                             {450.5F, 248.5F, 0.323843416370107F, 281.F},
                             {442.F, 260.5F, 0.32280701754386F, 285.F},
                             {437.F, 255.5F, 0.329824561403509F, 285.F},
                             {427.F, 251.5F, 0.329896907216495F, 291.F},
                             {419.F, 251.F, 0.330985915492958F, 284.F},
                             {411.F, 251.F, 0.328671328671329F, 286.F},
                             {411.F, 251.5F, 0.325259515570934F, 289.F},
                             {410.F, 249.F, 0.324137931034483F, 290.F},
                             {407.F, 247.5F, 0.346020761245675F, 289.F},
                             {398.5F, 248.5F, 0.356890459363958F, 283.F},
                             {393.F, 249.F, 0.347222222222222F, 288.F},
                             {390.5F, 246.5F, 0.331058020477816F, 293.F},
                             {387.F, 246.F, 0.308724832214765F, 298.F},
                             {379.5F, 244.5F, 0.303754266211604F, 293.F},
                             {370.F, 255.5F, 0.258899676375404F, 309.F},
                             {372.F, 252.5F, 0.307167235494881F, 293.F},
                             {368.F, 254.5F, 0.311418685121107F, 289.F},
                             {365.5F, 251.F, 0.322916666666667F, 288.F},
                             {360.5F, 250.5F, 0.301694915254237F, 295.F},
                             {353.F, 251.5F, 0.316151202749141F, 291.F},
                             {349.5F, 248.5F, 0.32404181184669F, 287.F},
                             {343.5F, 246.F, 0.327464788732394F, 284.F},
                             {334.5F, 251.5F, 0.335689045936396F, 283.F},
                             {328.5F, 249.5F, 0.342960288808664F, 277.F},
                             {321.5F, 256.5F, 0.328621908127208F, 283.F},
                             {321.5F, 259.5F, 0.317073170731707F, 287.F},
                             {319.5F, 252.F, 0.313380281690141F, 284.F},
                             {317.5F, 247.5F, 0.314487632508834F, 283.F},
                             {314.5F, 248.F, 0.313380281690141F, 284.F},
                             {318.5F, 255.F, 0.311188811188811F, 286.F},
                             {324.5F, 252.F, 0.317857142857143F, 280.F},
                             {328.5F, 249.F, 0.311188811188811F, 286.F},
                             {330.F, 248.F, 0.318840579710145F, 276.F},
                             {334.5F, 245.F, 0.320143884892086F, 278.F},
                             {342.5F, 248.F, 0.324817518248175F, 274.F},
                             {348.F, 247.5F, 0.312727272727273F, 275.F},
                             {349.5F, 245.5F, 0.326007326007326F, 273.F},
                             {350.F, 250.F, 0.321167883211679F, 274.F},
                             {350.5F, 252.5F, 0.323636363636364F, 275.F},
                             {356.5F, 249.F, 0.31294964028777F, 278.F},
                             {356.5F, 245.F, 0.320143884892086F, 278.F},
                             {357.F, 245.F, 0.314285714285714F, 280.F},
                             {361.F, 246.F, 0.318840579710145F, 276.F},
                             {364.F, 251.5F, 0.308771929824561F, 285.F},
                             {368.F, 252.5F, 0.303886925795053F, 283.F},
                             {369.F, 250.5F, 0.29757785467128F, 289.F}};

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
