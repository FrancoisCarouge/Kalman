#include "fcarouge/kalman.hpp"
#include "fcarouge/linalg.hpp"

#include <cassert>
#include <cmath>

template <typename Numerator, fcarouge::algebraic Denominator>
constexpr auto fcarouge::operator/(const Numerator &lhs, const Denominator &rhs)
    -> fcarouge::quotient<Numerator, Denominator> {
  return rhs.transpose()
      .fullPivHouseholderQr()
      .solve(lhs.transpose())
      .transpose();
}

namespace fcarouge::sample {
namespace {
template <auto Size> using vector = column_vector<float, Size>;
using state = vector<8>;
using output = vector<4>;
using no_input = void;

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
  // A 8x4x0 filter, constant velocity, linear.
  using kalman = kalman<state, output, no_input>;

  kalman filter;

  // A hundred bounding box output measurements `(x, y, a, h)` from Deep SORT's
  // MOT16 sample, tracker #201.
  const kalman::output measured[]{{603.5F, 251.5F, 0.187335092348285F, 379},
                                  {599, 241, 0.24390243902439F, 328},
                                  {599, 239.5F, 0.257234726688103F, 311},
                                  {602.5F, 244, 0.240131578947368F, 304},
                                  {598, 248.5F, 0.272425249169435F, 301},
                                  {596.5F, 240.5F, 0.283276450511945F, 293},
                                  {601, 227, 0.301587301587302F, 252},
                                  {603.5F, 235.5F, 0.282868525896414F, 251},
                                  {602, 242.5F, 0.292490118577075F, 253},
                                  {602.5F, 253, 0.218562874251497F, 334},
                                  {593, 254, 0.273291925465838F, 322},
                                  {603, 264, 0.22360248447205F, 322},
                                  {600.5F, 278.5F, 0.198966408268734F, 387},
                                  {593, 280, 0.237113402061856F, 388},
                                  {588.5F, 269, 0.267195767195767F, 378},
                                  {579, 260, 0.311111111111111F, 360},
                                  {565.5F, 268.5F, 0.339130434782609F, 345},
                                  {558.5F, 255.5F, 0.366568914956012F, 341},
                                  {544, 268, 0.364705882352941F, 340},
                                  {533, 258.5F, 0.356083086053412F, 337},
                                  {519, 258, 0.353293413173653F, 334},
                                  {511.5F, 252.5F, 0.333333333333333F, 333},
                                  {515.5F, 252.5F, 0.31306990881459F, 329},
                                  {523.5F, 251, 0.298192771084337F, 332},
                                  {540, 252.5F, 0.318318318318318F, 333},
                                  {574, 262, 0.344827586206897F, 348},
                                  {590.5F, 265, 0.278735632183908F, 348},
                                  {613, 268, 0.164556962025316F, 316},
                                  {617, 260.5F, 0.161172161172161F, 273},
                                  {615.5F, 261.5F, 0.15210355987055F, 309},
                                  {605.5F, 259, 0.226351351351351F, 296},
                                  {595.5F, 258.5F, 0.289036544850498F, 301},
                                  {588, 257.5F, 0.350515463917526F, 291},
                                  {579.5F, 254, 0.343537414965986F, 294},
                                  {569.5F, 258.5F, 0.353535353535354F, 297},
                                  {565.5F, 257, 0.37248322147651F, 298},
                                  {555, 250, 0.388157894736842F, 304},
                                  {546.5F, 249, 0.336666666666667F, 300},
                                  {535, 251, 0.30718954248366F, 306},
                                  {530, 246, 0.308724832214765F, 298},
                                  {521, 252, 0.278145695364238F, 302},
                                  {521.5F, 254.5F, 0.331010452961672F, 287},
                                  {521, 258.5F, 0.32280701754386F, 285},
                                  {519.5F, 255, 0.316326530612245F, 294},
                                  {518.5F, 255, 0.304794520547945F, 292},
                                  {511, 253, 0.310810810810811F, 296},
                                  {506, 255, 0.319727891156463F, 294},
                                  {499, 256, 0.352112676056338F, 284},
                                  {492.5F, 256.5F, 0.349152542372881F, 295},
                                  {489.5F, 257, 0.362068965517241F, 290},
                                  {481, 251.5F, 0.357894736842105F, 285},
                                  {474, 249, 0.324137931034483F, 290},
                                  {466, 250, 0.306122448979592F, 294},
                                  {461.5F, 248, 0.304794520547945F, 292},
                                  {450.5F, 248.5F, 0.323843416370107F, 281},
                                  {442, 260.5F, 0.32280701754386F, 285},
                                  {437, 255.5F, 0.329824561403509F, 285},
                                  {427, 251.5F, 0.329896907216495F, 291},
                                  {419, 251, 0.330985915492958F, 284},
                                  {411, 251, 0.328671328671329F, 286},
                                  {411, 251.5F, 0.325259515570934F, 289},
                                  {410, 249, 0.324137931034483F, 290},
                                  {407, 247.5F, 0.346020761245675F, 289},
                                  {398.5F, 248.5F, 0.356890459363958F, 283},
                                  {393, 249, 0.347222222222222F, 288},
                                  {390.5F, 246.5F, 0.331058020477816F, 293},
                                  {387, 246, 0.308724832214765F, 298},
                                  {379.5F, 244.5F, 0.303754266211604F, 293},
                                  {370, 255.5F, 0.258899676375404F, 309},
                                  {372, 252.5F, 0.307167235494881F, 293},
                                  {368, 254.5F, 0.311418685121107F, 289},
                                  {365.5F, 251, 0.322916666666667F, 288},
                                  {360.5F, 250.5F, 0.301694915254237F, 295},
                                  {353, 251.5F, 0.316151202749141F, 291},
                                  {349.5F, 248.5F, 0.32404181184669F, 287},
                                  {343.5F, 246, 0.327464788732394F, 284},
                                  {334.5F, 251.5F, 0.335689045936396F, 283},
                                  {328.5F, 249.5F, 0.342960288808664F, 277},
                                  {321.5F, 256.5F, 0.328621908127208F, 283},
                                  {321.5F, 259.5F, 0.317073170731707F, 287},
                                  {319.5F, 252, 0.313380281690141F, 284},
                                  {317.5F, 247.5F, 0.314487632508834F, 283},
                                  {314.5F, 248, 0.313380281690141F, 284},
                                  {318.5F, 255, 0.311188811188811F, 286},
                                  {324.5F, 252, 0.317857142857143F, 280},
                                  {328.5F, 249, 0.311188811188811F, 286},
                                  {330, 248, 0.318840579710145F, 276},
                                  {334.5F, 245, 0.320143884892086F, 278},
                                  {342.5F, 248, 0.324817518248175F, 274},
                                  {348, 247.5F, 0.312727272727273F, 275},
                                  {349.5F, 245.5F, 0.326007326007326F, 273},
                                  {350, 250, 0.321167883211679F, 274},
                                  {350.5F, 252.5F, 0.323636363636364F, 275},
                                  {356.5F, 249, 0.31294964028777F, 278},
                                  {356.5F, 245, 0.320143884892086F, 278},
                                  {357, 245, 0.314285714285714F, 280},
                                  {361, 246, 0.318840579710145F, 276},
                                  {364, 251.5F, 0.308771929824561F, 285},
                                  {368, 252.5F, 0.303886925795053F, 283},
                                  {369, 250.5F, 0.29757785467128F, 289}};

  // Initialization
  // The filter is initialized at runtime, on bounding box detection, with the
  // first observed output. Bounding box position and velocity estimated state:
  // [px, py, pa, ph, vx, vy, va, vh].
  const kalman::output initial_box{605.0F, 248.0F, 0.20481927710843373F,
                                   332.0F};
  filter.x(initial_box(0), initial_box(1), initial_box(2), initial_box(3), 0, 0,
           0, 0);

  // Experimental position and velocity uncertainty standard deviation weights.
  const float position_weight{1.F / 20.F};
  const float velocity_weight{1.F / 160.F};

  filter.p(kalman::estimate_uncertainty{
      state{2 * position_weight * initial_box(3),
            2 * position_weight * initial_box(3), 1e-2F,
            2 * position_weight * initial_box(3),
            10 * velocity_weight * initial_box(3),
            10 * velocity_weight * initial_box(3), 1e-5F,
            10 * velocity_weight * initial_box(3)}
          .array()
          .square()
          .matrix()
          .asDiagonal()});

  // Constant velocity, linear state transition model. From one image frame to
  // the other.
  const float delta_time{1};
  filter.f(kalman::state_transition{{1, 0, 0, 0, delta_time, 0, 0, 0},
                                    {0, 1, 0, 0, 0, delta_time, 0, 0},
                                    {0, 0, 1, 0, 0, 0, delta_time, 0},
                                    {0, 0, 0, 1, 0, 0, 0, delta_time},
                                    {0, 0, 0, 0, 1, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 1, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 1, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 1}});

  filter.q([position_weight, velocity_weight](
               const kalman::state &x) -> kalman::process_uncertainty {
    return state{position_weight * x(3),
                 position_weight * x(3),
                 1e-2F,
                 position_weight * x(3),
                 velocity_weight * x(3),
                 velocity_weight * x(3),
                 1e-5F,
                 velocity_weight * x(3)}
        .array()
        .square()
        .matrix()
        .asDiagonal();
  });

  // Now we can predict the next state based on the initialization values.
  filter.predict();

  // Measure and Update
  // Direct linear observation transition model.
  filter.h(kalman::output_model{{1, 0, 0, 0, 0, 0, 0, 0},
                                {0, 1, 0, 0, 0, 0, 0, 0},
                                {0, 0, 1, 0, 0, 0, 0, 0},
                                {0, 0, 0, 1, 0, 0, 0, 0}});

  // Observation, measurement noise covariance.
  filter.r([position_weight](const kalman::state &x,
                             [[maybe_unused]] const kalman::output &z)
               -> kalman::output_uncertainty {
    return output{position_weight * x(3), position_weight * x(3), 1e-1F,
                  position_weight * x(3)}
        .array()
        .square()
        .matrix()
        .asDiagonal();
  });

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
