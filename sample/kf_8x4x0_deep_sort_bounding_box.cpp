#include "fcarouge/kalman.hpp"

#include <Eigen/Eigen>

#include <cassert>
#include <cmath>

namespace fcarouge::sample {
namespace {

template <typename Type, auto Size> using vector = Eigen::Vector<Type, Size>;

struct divide final {
  template <typename Numerator, typename Denominator>
  [[nodiscard]] inline constexpr auto
  operator()(const Numerator &numerator, const Denominator &denominator) const {
    using result =
        typename Eigen::Matrix<typename std::decay_t<Numerator>::Scalar,
                               std::decay_t<Numerator>::RowsAtCompileTime,
                               std::decay_t<Denominator>::RowsAtCompileTime>;

    return result{denominator.transpose()
                      .fullPivHouseholderQr()
                      .solve(numerator.transpose())
                      .transpose()
                      .eval()};
  }
};

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
  using kalman = kalman<vector<float, 8>, vector<float, 4>, void, divide>;

  kalman filter;

  // A hundred bounding box output measurements `(x, y, a, h)` from Deep SORT's
  // MOT16 sample, tracker #201.
  const kalman::output measured[]{{603.5f, 251.5f, 0.187335092348285f, 379},
                                  {599, 241, 0.24390243902439f, 328},
                                  {599, 239.5f, 0.257234726688103f, 311},
                                  {602.5f, 244, 0.240131578947368f, 304},
                                  {598, 248.5f, 0.272425249169435f, 301},
                                  {596.5f, 240.5f, 0.283276450511945f, 293},
                                  {601, 227, 0.301587301587302f, 252},
                                  {603.5f, 235.5f, 0.282868525896414f, 251},
                                  {602, 242.5f, 0.292490118577075f, 253},
                                  {602.5f, 253, 0.218562874251497f, 334},
                                  {593, 254, 0.273291925465838f, 322},
                                  {603, 264, 0.22360248447205f, 322},
                                  {600.5f, 278.5f, 0.198966408268734f, 387},
                                  {593, 280, 0.237113402061856f, 388},
                                  {588.5f, 269, 0.267195767195767f, 378},
                                  {579, 260, 0.311111111111111f, 360},
                                  {565.5f, 268.5f, 0.339130434782609f, 345},
                                  {558.5f, 255.5f, 0.366568914956012f, 341},
                                  {544, 268, 0.364705882352941f, 340},
                                  {533, 258.5f, 0.356083086053412f, 337},
                                  {519, 258, 0.353293413173653f, 334},
                                  {511.5f, 252.5f, 0.333333333333333f, 333},
                                  {515.5f, 252.5f, 0.31306990881459f, 329},
                                  {523.5f, 251, 0.298192771084337f, 332},
                                  {540, 252.5f, 0.318318318318318f, 333},
                                  {574, 262, 0.344827586206897f, 348},
                                  {590.5f, 265, 0.278735632183908f, 348},
                                  {613, 268, 0.164556962025316f, 316},
                                  {617, 260.5f, 0.161172161172161f, 273},
                                  {615.5f, 261.5f, 0.15210355987055f, 309},
                                  {605.5f, 259, 0.226351351351351f, 296},
                                  {595.5f, 258.5f, 0.289036544850498f, 301},
                                  {588, 257.5f, 0.350515463917526f, 291},
                                  {579.5f, 254, 0.343537414965986f, 294},
                                  {569.5f, 258.5f, 0.353535353535354f, 297},
                                  {565.5f, 257, 0.37248322147651f, 298},
                                  {555, 250, 0.388157894736842f, 304},
                                  {546.5f, 249, 0.336666666666667f, 300},
                                  {535, 251, 0.30718954248366f, 306},
                                  {530, 246, 0.308724832214765f, 298},
                                  {521, 252, 0.278145695364238f, 302},
                                  {521.5f, 254.5f, 0.331010452961672f, 287},
                                  {521, 258.5f, 0.32280701754386f, 285},
                                  {519.5f, 255, 0.316326530612245f, 294},
                                  {518.5f, 255, 0.304794520547945f, 292},
                                  {511, 253, 0.310810810810811f, 296},
                                  {506, 255, 0.319727891156463f, 294},
                                  {499, 256, 0.352112676056338f, 284},
                                  {492.5f, 256.5f, 0.349152542372881f, 295},
                                  {489.5f, 257, 0.362068965517241f, 290},
                                  {481, 251.5f, 0.357894736842105f, 285},
                                  {474, 249, 0.324137931034483f, 290},
                                  {466, 250, 0.306122448979592f, 294},
                                  {461.5f, 248, 0.304794520547945f, 292},
                                  {450.5f, 248.5f, 0.323843416370107f, 281},
                                  {442, 260.5f, 0.32280701754386f, 285},
                                  {437, 255.5f, 0.329824561403509f, 285},
                                  {427, 251.5f, 0.329896907216495f, 291},
                                  {419, 251, 0.330985915492958f, 284},
                                  {411, 251, 0.328671328671329f, 286},
                                  {411, 251.5f, 0.325259515570934f, 289},
                                  {410, 249, 0.324137931034483f, 290},
                                  {407, 247.5f, 0.346020761245675f, 289},
                                  {398.5f, 248.5f, 0.356890459363958f, 283},
                                  {393, 249, 0.347222222222222f, 288},
                                  {390.5f, 246.5f, 0.331058020477816f, 293},
                                  {387, 246, 0.308724832214765f, 298},
                                  {379.5f, 244.5f, 0.303754266211604f, 293},
                                  {370, 255.5f, 0.258899676375404f, 309},
                                  {372, 252.5f, 0.307167235494881f, 293},
                                  {368, 254.5f, 0.311418685121107f, 289},
                                  {365.5f, 251, 0.322916666666667f, 288},
                                  {360.5f, 250.5f, 0.301694915254237f, 295},
                                  {353, 251.5f, 0.316151202749141f, 291},
                                  {349.5f, 248.5f, 0.32404181184669f, 287},
                                  {343.5f, 246, 0.327464788732394f, 284},
                                  {334.5f, 251.5f, 0.335689045936396f, 283},
                                  {328.5f, 249.5f, 0.342960288808664f, 277},
                                  {321.5f, 256.5f, 0.328621908127208f, 283},
                                  {321.5f, 259.5f, 0.317073170731707f, 287},
                                  {319.5f, 252, 0.313380281690141f, 284},
                                  {317.5f, 247.5f, 0.314487632508834f, 283},
                                  {314.5f, 248, 0.313380281690141f, 284},
                                  {318.5f, 255, 0.311188811188811f, 286},
                                  {324.5f, 252, 0.317857142857143f, 280},
                                  {328.5f, 249, 0.311188811188811f, 286},
                                  {330, 248, 0.318840579710145f, 276},
                                  {334.5f, 245, 0.320143884892086f, 278},
                                  {342.5f, 248, 0.324817518248175f, 274},
                                  {348, 247.5f, 0.312727272727273f, 275},
                                  {349.5f, 245.5f, 0.326007326007326f, 273},
                                  {350, 250, 0.321167883211679f, 274},
                                  {350.5f, 252.5f, 0.323636363636364f, 275},
                                  {356.5f, 249, 0.31294964028777f, 278},
                                  {356.5f, 245, 0.320143884892086f, 278},
                                  {357, 245, 0.314285714285714f, 280},
                                  {361, 246, 0.318840579710145f, 276},
                                  {364, 251.5f, 0.308771929824561f, 285},
                                  {368, 252.5f, 0.303886925795053f, 283},
                                  {369, 250.5f, 0.29757785467128f, 289}};

  // Initialization
  // The filter is initialized at runtime, on bounding box detection, with the
  // first observed output. Bounding box position and velocity estimated state:
  // [px, py, pa, ph, vx, vy, va, vh].
  const kalman::output initial_box{605.0f, 248.0f, 0.20481927710843373f,
                                   332.0f};
  filter.x(initial_box(0), initial_box(1), initial_box(2), initial_box(3), 0, 0,
           0, 0);

  // Experimental position and velocity uncertainty standard deviation weights.
  const float position_weight{1.f / 20.f};
  const float velocity_weight{1.f / 160.f};

  filter.p(kalman::estimate_uncertainty{
      vector<float, 8>{2 * position_weight * initial_box(3),
                       2 * position_weight * initial_box(3), 1e-2f,
                       2 * position_weight * initial_box(3),
                       10 * velocity_weight * initial_box(3),
                       10 * velocity_weight * initial_box(3), 1e-5f,
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
    return vector<float, 8>{position_weight * x(3),
                            position_weight * x(3),
                            1e-2f,
                            position_weight * x(3),
                            velocity_weight * x(3),
                            velocity_weight * x(3),
                            1e-5f,
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
  filter.r(
      [position_weight](const kalman::state &x,
                        const kalman::output &z) -> kalman::output_uncertainty {
        static_cast<void>(z);
        return vector<float, 4>{position_weight * x(3), position_weight * x(3),
                                1e-1f, position_weight * x(3)}
            .array()
            .square()
            .matrix()
            .asDiagonal();
      });

  // And so on, run a step of the filter, updating and predicting, every frame.
  for (const auto &output : measured) {
    filter.update(output);
    filter.predict();
  }

  assert(std::abs(1 - filter.x()[0] / 370.932041394761f) < 0.001f &&
         std::abs(1 - filter.x()[1] / 251.173174229878f) < 0.001f &&
         std::abs(1 - filter.x()[2] / 0.314757138075364f) < 0.001f &&
         std::abs(1 - filter.x()[3] / 287.859996019444f) < 0.001f &&
         std::abs(1 - filter.x()[4] / 1.95865368159518f) < 0.001f &&
         std::abs(1 - filter.x()[5] / 0.229282868701086f) < 0.001f &&
         // The precision of the velocity appears to saturate early on in the
         // original example. The parameter could be scaled or larger types used
         // to improve comparison accuracy.
         std::abs(1 - filter.x()[6] / 2.46138628550094E-06f) < 0.5f &&
         std::abs(1 - filter.x()[7] / 0.81402529074969f) < 0.001f &&
         "The estimated states expected to meet Nwojke's Deep SORT filter's "
         "MOT16 sample tracker #201 dataset at 0.1% accuracy.");

  return 0;
}()};

} // namespace
} // namespace fcarouge::sample
