#include "fcarouge/eigen/kalman.hpp"

#include <cassert>
#include <cmath>

namespace fcarouge::eigen::sample {
namespace {
//! @brief ArduPilot Plane Soaring
//!
//! @copyright This example is transcribed from the ArduPilot Soaring Plane
//! copyright ArduPilot Dev Team.
//!
//! @see https://ardupilot.org/plane/docs/soaring.html
//! @see https://arxiv.org/abs/1802.08215
//!
//! @details The autonomous soaring functionality in ArduPilot allows the plane
//! to respond to rising air current in order to extend endurance and gain
//! altitude with minimal use of the motor. The full technical description is
//! available in S. Tabor, I. Guilliard, A. Kolobov. ArduSoar: an Open-Source
//! Thermalling Controller for Resource-Constrained Autopilots. International
//! Conference on Intelligent Robots and Systems (IROS), 2018.
//! Estimating the parameters of a Wharington et al thermal model state X: [W,
//! R, x, y] with the speed or strength W [m.s^-1] at the center of the thermal
//! of radius R [m] with the center distance x north of the sUAV and y east of
//! the sUAV.
//!
//! @example ekf_4x1x0_ardupilot_soaring.cpp
//!
//! @todo Add a data set and assert for correctness of results.
[[maybe_unused]] auto ekf_4x1x0_ardupilot_soaring{[] {
  // 4x1 extended filter with additional parameter for prediction: driftX [m],
  // driftY [m]. Constant time step.
  using kalman = kalman<vector<float, 4>, float, void, std::tuple<float, float>,
                        std::tuple<float, float>>;

  kalman k;

  // Initialization
  const float trigger_strength{1 / 4.06};
  const float thermal_radius{80};
  const float thermal_position_x{0};
  const float thermal_position_y{0};
  k.x(trigger_strength, thermal_radius, thermal_position_x, thermal_position_y);

  const float strength_covariance{0.0049};
  const float radius_covariance{400};
  const float position_covariance{400}; // For both positions x and y.
  k.p(kalman::estimate_uncertainty{{strength_covariance, 0, 0, 0},
                                   {0, radius_covariance, 0, 0},
                                   {0, 0, position_covariance, 0},
                                   {0, 0, 0, position_covariance}});

  // No process dynamics: F = ∂f/∂X = I4 Default.

  k.transition([](const kalman::state &x, const float &drift_x,
                  const float &drift_y) -> kalman::state {
    const kalman::state drifts{0, 0, -drift_x, -drift_y};
    return x + drifts;
  });

  const float strength_noise{std::pow(0.001f, 2.f)};
  const float distance_noise{std::pow(0.03f, 2.f)};
  k.q(kalman::process_uncertainty{{strength_noise, 0, 0, 0},
                                  {0, distance_noise, 0, 0},
                                  {0, 0, distance_noise, 0},
                                  {0, 0, 0, distance_noise}});

  const float measure_noise{std::pow(0.45f, 2.f)};
  k.r(kalman::output_uncertainty{measure_noise});

  // Observation Z: [w] vertical air velocity w at the aircraft’s
  // position w.r.t. the thermal center [m.s^-1].
  k.observation([](const kalman::state &x, const float &position_x,
                   const float &position_y) -> kalman::output {
    const auto exp{std::exp(
        -(std::pow(x(2) - position_x, 2.f) + std::pow(x(3) - position_y, 2.f)) /
        std::pow(x(1), 2.f))};
    return kalman::output{x(0) * exp};
  });

  // See the ArduSoar paper for the equation for H = ∂h/∂X:
  k.h([](const kalman::state &x, const float &position_x,
         const float &position_y) -> kalman::output_model {
    const auto exp{std::exp(
        -(std::pow(x(2) - position_x, 2.f) + std::pow(x(3) - position_y, 2.f)) /
        std::pow(x(1), 2.f))};
    const kalman::output_model h{
        exp,
        2 * x(0) *
            ((std::pow(x(2) - position_x, 2.f) +
              std::pow(x(3) - position_y, 2.f)) /
             std::pow(x(1), 3.f)) *
            exp,
        -2 * (x(0) * (x(2) - position_x) / std::pow(x(1), 2.f)) * exp,
        -2 * (x(0) * (x(3) - position_y) / std::pow(x(1), 2.f)) * exp};
    return h;
  });

  // Measure and Update
  float drift_x{0};
  float drift_y{0};
  k.predict(drift_x, drift_y);

  float variometer{0};
  float position_x{0};
  float position_y{0};
  k.update(variometer, position_x, position_y);

  // Or so on.
  k(drift_x, drift_y, position_x, position_y, variometer);

  return 0;
}()};

} // namespace
} // namespace fcarouge::eigen::sample
