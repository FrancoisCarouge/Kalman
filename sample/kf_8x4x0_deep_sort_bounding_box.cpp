#include "fcarouge/eigen/kalman.hpp"

#include <cassert>
#include <cmath>

namespace fcarouge::eigen::sample {
namespace {

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
[[maybe_unused]] auto kf_8x4x0_deep_sort_bounding_box{[] {
  // A 8x4x0 filter, constant velocity, linear.
  using kalman = kalman<vector<float, 8>, vector<float, 4>>;

  kalman k;

  // A hundred directly bounding box output measurements `(x, y, a, h)` from
  // Deep SORT's MOT16 sample, tracker #201.
  const kalman::output measured[]{{603.5, 251.5, 0.187335092348285, 379},
                                  {599, 241, 0.24390243902439, 328},
                                  {599, 239.5, 0.257234726688103, 311},
                                  {602.5, 244, 0.240131578947368, 304},
                                  {598, 248.5, 0.272425249169435, 301},
                                  {596.5, 240.5, 0.283276450511945, 293},
                                  {601, 227, 0.301587301587302, 252},
                                  {603.5, 235.5, 0.282868525896414, 251},
                                  {602, 242.5, 0.292490118577075, 253},
                                  {602.5, 253, 0.218562874251497, 334},
                                  {593, 254, 0.273291925465838, 322},
                                  {603, 264, 0.22360248447205, 322},
                                  {600.5, 278.5, 0.198966408268734, 387},
                                  {593, 280, 0.237113402061856, 388},
                                  {588.5, 269, 0.267195767195767, 378},
                                  {579, 260, 0.311111111111111, 360},
                                  {565.5, 268.5, 0.339130434782609, 345},
                                  {558.5, 255.5, 0.366568914956012, 341},
                                  {544, 268, 0.364705882352941, 340},
                                  {533, 258.5, 0.356083086053412, 337},
                                  {519, 258, 0.353293413173653, 334},
                                  {511.5, 252.5, 0.333333333333333, 333},
                                  {515.5, 252.5, 0.31306990881459, 329},
                                  {523.5, 251, 0.298192771084337, 332},
                                  {540, 252.5, 0.318318318318318, 333},
                                  {574, 262, 0.344827586206897, 348},
                                  {590.5, 265, 0.278735632183908, 348},
                                  {613, 268, 0.164556962025316, 316},
                                  {617, 260.5, 0.161172161172161, 273},
                                  {615.5, 261.5, 0.15210355987055, 309},
                                  {605.5, 259, 0.226351351351351, 296},
                                  {595.5, 258.5, 0.289036544850498, 301},
                                  {588, 257.5, 0.350515463917526, 291},
                                  {579.5, 254, 0.343537414965986, 294},
                                  {569.5, 258.5, 0.353535353535354, 297},
                                  {565.5, 257, 0.37248322147651, 298},
                                  {555, 250, 0.388157894736842, 304},
                                  {546.5, 249, 0.336666666666667, 300},
                                  {535, 251, 0.30718954248366, 306},
                                  {530, 246, 0.308724832214765, 298},
                                  {521, 252, 0.278145695364238, 302},
                                  {521.5, 254.5, 0.331010452961672, 287},
                                  {521, 258.5, 0.32280701754386, 285},
                                  {519.5, 255, 0.316326530612245, 294},
                                  {518.5, 255, 0.304794520547945, 292},
                                  {511, 253, 0.310810810810811, 296},
                                  {506, 255, 0.319727891156463, 294},
                                  {499, 256, 0.352112676056338, 284},
                                  {492.5, 256.5, 0.349152542372881, 295},
                                  {489.5, 257, 0.362068965517241, 290},
                                  {481, 251.5, 0.357894736842105, 285},
                                  {474, 249, 0.324137931034483, 290},
                                  {466, 250, 0.306122448979592, 294},
                                  {461.5, 248, 0.304794520547945, 292},
                                  {450.5, 248.5, 0.323843416370107, 281},
                                  {442, 260.5, 0.32280701754386, 285},
                                  {437, 255.5, 0.329824561403509, 285},
                                  {427, 251.5, 0.329896907216495, 291},
                                  {419, 251, 0.330985915492958, 284},
                                  {411, 251, 0.328671328671329, 286},
                                  {411, 251.5, 0.325259515570934, 289},
                                  {410, 249, 0.324137931034483, 290},
                                  {407, 247.5, 0.346020761245675, 289},
                                  {398.5, 248.5, 0.356890459363958, 283},
                                  {393, 249, 0.347222222222222, 288},
                                  {390.5, 246.5, 0.331058020477816, 293},
                                  {387, 246, 0.308724832214765, 298},
                                  {379.5, 244.5, 0.303754266211604, 293},
                                  {370, 255.5, 0.258899676375404, 309},
                                  {372, 252.5, 0.307167235494881, 293},
                                  {368, 254.5, 0.311418685121107, 289},
                                  {365.5, 251, 0.322916666666667, 288},
                                  {360.5, 250.5, 0.301694915254237, 295},
                                  {353, 251.5, 0.316151202749141, 291},
                                  {349.5, 248.5, 0.32404181184669, 287},
                                  {343.5, 246, 0.327464788732394, 284},
                                  {334.5, 251.5, 0.335689045936396, 283},
                                  {328.5, 249.5, 0.342960288808664, 277},
                                  {321.5, 256.5, 0.328621908127208, 283},
                                  {321.5, 259.5, 0.317073170731707, 287},
                                  {319.5, 252, 0.313380281690141, 284},
                                  {317.5, 247.5, 0.314487632508834, 283},
                                  {314.5, 248, 0.313380281690141, 284},
                                  {318.5, 255, 0.311188811188811, 286},
                                  {324.5, 252, 0.317857142857143, 280},
                                  {328.5, 249, 0.311188811188811, 286},
                                  {330, 248, 0.318840579710145, 276},
                                  {334.5, 245, 0.320143884892086, 278},
                                  {342.5, 248, 0.324817518248175, 274},
                                  {348, 247.5, 0.312727272727273, 275},
                                  {349.5, 245.5, 0.326007326007326, 273},
                                  {350, 250, 0.321167883211679, 274},
                                  {350.5, 252.5, 0.323636363636364, 275},
                                  {356.5, 249, 0.31294964028777, 278},
                                  {356.5, 245, 0.320143884892086, 278},
                                  {357, 245, 0.314285714285714, 280},
                                  {361, 246, 0.318840579710145, 276},
                                  {364, 251.5, 0.308771929824561, 285},
                                  {368, 252.5, 0.303886925795053, 283},
                                  {369, 250.5, 0.29757785467128, 289}};

  // Initialization
  // The filter is initialized at runtime, on bounding box detection, with the
  // first observed output. Bounding box position and velocity estimated state:
  // [px, py, pa, ph, vx, vy, va, vh].
  const kalman::output initial_box{605.0, 248.0, 0.20481927710843373, 332.0};
  k.x(initial_box(0), initial_box(1), initial_box(2), initial_box(3), 0, 0, 0,
      0);

  // Experimental position and velocity uncertainty standard deviation weights.
  const float position_weight{1. / 20.};
  const float velocity_weight{1. / 160.};

  k.p(kalman::estimate_uncertainty{
      vector<float, 8>{2 * position_weight * initial_box(3),
                       2 * position_weight * initial_box(3), 1e-2,
                       2 * position_weight * initial_box(3),
                       10 * velocity_weight * initial_box(3),
                       10 * velocity_weight * initial_box(3), 1e-5,
                       10 * velocity_weight * initial_box(3)}
          .array()
          .square()
          .matrix()
          .asDiagonal()});

  // Constant velocity, linear state transition model. From one image frame to
  // the other.
  const float delta_time{1};
  k.f(kalman::state_transition{{1, 0, 0, 0, delta_time, 0, 0, 0},
                               {0, 1, 0, 0, 0, delta_time, 0, 0},
                               {0, 0, 1, 0, 0, 0, delta_time, 0},
                               {0, 0, 0, 1, 0, 0, 0, delta_time},
                               {0, 0, 0, 0, 1, 0, 0, 0},
                               {0, 0, 0, 0, 0, 1, 0, 0},
                               {0, 0, 0, 0, 0, 0, 1, 0},
                               {0, 0, 0, 0, 0, 0, 0, 1}});

  k.q([position_weight,
       velocity_weight](const kalman::state &x) -> kalman::process_uncertainty {
    return vector<float, 8>{position_weight * x(3),
                            position_weight * x(3),
                            1e-2,
                            position_weight * x(3),
                            velocity_weight * x(3),
                            velocity_weight * x(3),
                            1e-5,
                            velocity_weight * x(3)}
        .array()
        .square()
        .matrix()
        .asDiagonal();
  });

  // Now we can predict the next state based on the initialization values.
  k.predict();

  // Measure and Update
  // Direct linear observation transition model.
  k.h(kalman::output_model{{1, 0, 0, 0, 0, 0, 0, 0},
                           {0, 1, 0, 0, 0, 0, 0, 0},
                           {0, 0, 1, 0, 0, 0, 0, 0},
                           {0, 0, 0, 1, 0, 0, 0, 0}});

  // Observation, measurement noise covariance.
  k.r([position_weight](const kalman::state &x,
                        const kalman::output &z) -> kalman::output_uncertainty {
    static_cast<void>(z);
    return vector<float, 4>{position_weight * x(3), position_weight * x(3),
                            1e-1, position_weight * x(3)}
        .array()
        .square()
        .matrix()
        .asDiagonal();
  });

  // And so on, run a step of the filter, updating and predicting, every frame.
  for (auto &output : measured) {
    k(output);
  }

  assert(std::abs(1 - k.x()[0] / 370.932041394761f) < 0.001f &&
         std::abs(1 - k.x()[1] / 251.173174229878f) < 0.001f &&
         std::abs(1 - k.x()[2] / 0.314757138075364f) < 0.001f &&
         std::abs(1 - k.x()[3] / 287.859996019444f) < 0.001f &&
         std::abs(1 - k.x()[4] / 1.95865368159518f) < 0.001f &&
         std::abs(1 - k.x()[5] / 0.229282868701086f) < 0.001f &&
         // The precision of the velocity appears to saturate early on in the
         // original example. The parameter could be scaled or larger types used
         // to improve comparison accuracy.
         std::abs(1 - k.x()[6] / 2.46138628550094E-06f) < 0.5f &&
         std::abs(1 - k.x()[7] / 0.81402529074969f) < 0.001f &&
         "The estimated states expected to meet Nwojke's Deep SORT filter's "
         "MOT16 sample tracker #201 dataset at 0.1% accuracy.");

  return 0;
}()};

} // namespace
} // namespace fcarouge::eigen::sample
