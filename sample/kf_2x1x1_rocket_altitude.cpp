#include "fcarouge/eigen/kalman.hpp"

#include <cassert>
#include <chrono>
#include <cmath>

namespace fcarouge::eigen::sample {
namespace {
//! @brief Estimating the Rocket Altitude
//!
//! @copyright This example is transcribed from KalmanFilter.NET copyright Alex
//! Becker.
//!
//! @see https://www.kalmanfilter.net/multiExamples.html#ex10
//!
//! @details In this example, we will estimate the altitude of a rocket. The
//! rocket is equipped with an onboard altimeter that provides altitude
//! measurements. In addition to an altimeter, the rocket is equipped with an
//! accelerometer that measures the rocket acceleration. The accelerometer
//! serves as a control input to the Kalman Filter. We assume constant
//! acceleration dynamics. Accelerometers don't sense gravity. An accelerometer
//! at rest on a table would measure 1g upwards, while accelerometers in free
//! fall will measure zero. Thus, we need to subtract the gravitational
//! acceleration constant g from each accelerometer measurement. The
//! accelerometer measurement at time step n is: an = a − g + ϵ where:
//! - a is the actual acceleration of the object (the second derivative of the
//! object position).
//! - g is the gravitational acceleration constant; g = -9.8 m.s^-2.
//! - ϵ is the accelerometer measurement error.
//! In this example, we have a control variable u, which is based on the
//! accelerometer measurement. The system state xn is defined by: xn = [ pn vn ]
//! where: pn is the rocket altitude at time n. vn is the rocket velocity at
//! time n.
//! Let us assume a rocket boosting vertically with constant acceleration. The
//! rocket is equipped with an altimeter that provides altitude measurements and
//! an accelerometer that serves as a control input.
//! - The measurements period: Δt = 0.25s
//! - The rocket acceleration: a= 30 m.s^-2
//! - The altimeter measurement error standard deviation: σxm = 20m
//! - The accelerometer measurement error standard deviation: ϵ = 0.1 m.s^-2
//!
//! @example kf_2x1x1_rocket_altitude.cpp
[[maybe_unused]] auto kf_2x1x1_rocket_altitude{[] {
  // A 2x1x1 filter, constant acceleration dynamic model, no control, step time.
  using kalman = kalman<vector<double, 2>, double, double, std::tuple<>,
                        std::tuple<std::chrono::milliseconds>>;
  kalman filter;

  // Initialization
  // We don't know the rocket location; we will set initial position and
  // velocity to 0.
  filter.x(0., 0.);

  // Since our initial state vector is a guess, we will set a very high estimate
  // uncertainty. The high estimate uncertainty results in high Kalman gain,
  // giving a high weight to the measurement.
  filter.p(kalman::estimate_uncertainty{{500, 0}, {0, 500}});

  // Prediction
  // We will assume a discrete noise model - the noise is different at each time
  // period, but it is constant between time periods. In our previous example,
  // we used the system's random variance in acceleration σ^2 as a multiplier of
  // the process noise matrix. But here, we have an accelerometer that measures
  // the system random acceleration. The accelerometer error v is much lower
  // than system's random acceleration, therefore we use ϵ^2 as a multiplier of
  // the process noise matrix. This makes our estimation uncertainty much lower!
  filter.q(
      [](const kalman::state &x, const std::chrono::milliseconds &delta_time) {
        static_cast<void>(x);
        const auto dt{std::chrono::duration<double>(delta_time).count()};
        return kalman::process_uncertainty{
            {0.1 * 0.1 * dt * dt * dt * dt / 4, 0.1 * 0.1 * dt * dt * dt / 2},
            {0.1 * 0.1 * dt * dt * dt / 2, 0.1 * 0.1 * dt * dt}};
      });

  // The state transition matrix F would be:
  filter.f([](const kalman::state &x, const kalman::input &u,
              const std::chrono::milliseconds &delta_time) {
    static_cast<void>(x);
    static_cast<void>(u);
    const auto dt{std::chrono::duration<double>(delta_time).count()};
    return kalman::state_transition{{1, dt}, {0, 1}};
  });

  // The control matrix G would be:
  filter.g([](const std::chrono::milliseconds &delta_time) {
    const auto dt{std::chrono::duration<double>(delta_time).count()};
    return kalman::input_control{0.0313, dt};
  });

  // We also don't know what the rocket acceleration is, but we can assume that
  // it's greater than zero. Let's assume: u0 = g
  const double gravity{-9.8}; // [m.s^-2]
  const std::chrono::milliseconds delta_time{250};
  filter.predict(delta_time, -gravity);

  assert(std::abs(1 - filter.x()[0] / 0.3) < 0.03 &&
         std::abs(1 - filter.x()[1] / 2.45) < 0.03 &&
         "The state estimates expected at 3% accuracy.");
  assert(std::abs(1 - filter.p()(0, 0) / 531.25) < 0.001 &&
         std::abs(1 - filter.p()(0, 1) / 125) < 0.001 &&
         std::abs(1 - filter.p()(1, 0) / 125) < 0.001 &&
         std::abs(1 - filter.p()(1, 1) / 500) < 0.001 &&
         "The estimate uncertainty expected at 0.1% accuracy.");

  // Measure and Update
  // The dimension of zn is 1x1 and the dimension of xn is 2x1, so the dimension
  // of the observation matrix H will be 1x2.
  filter.h(kalman::output_model{1., 0.});

  // For the sake of the example simplicity, we will assume a constant
  // measurement uncertainty: R1 = R2...Rn-1 = Rn = R.
  filter.r(kalman::output_uncertainty{400.});

  filter.update(-32.4);

  assert(std::abs(1 - filter.x()[0] / -18.35) < 0.001 &&
         std::abs(1 - filter.x()[1] / -1.94) < 0.001 &&
         "The state estimates expected at 0.1% accuracy.");
  assert(std::abs(1 - filter.p()(0, 0) / 228.2) < 0.001 &&
         std::abs(1 - filter.p()(0, 1) / 53.7) < 0.001 &&
         std::abs(1 - filter.p()(1, 0) / 53.7) < 0.001 &&
         std::abs(1 - filter.p()(1, 1) / 483.2) < 0.001 &&
         "The estimate uncertainty expected at 0.1% accuracy.");

  filter.predict(delta_time, 39.72 + gravity);

  assert(std::abs(1 - filter.x()[0] / -17.9) < 0.001 &&
         std::abs(1 - filter.x()[1] / 5.54) < 0.001 &&
         "The state estimates expected at 0.1% accuracy.");
  assert(std::abs(1 - filter.p()(0, 0) / 285.2) < 0.001 &&
         std::abs(1 - filter.p()(0, 1) / 174.5) < 0.001 &&
         std::abs(1 - filter.p()(1, 0) / 174.5) < 0.001 &&
         std::abs(1 - filter.p()(1, 1) / 483.2) < 0.001 &&
         "The estimate uncertainty expected at 0.1% accuracy.");

  // And so on, run a step of the filter, updating and predicting, every
  // measurements period: Δt = 250ms. The period is constant but passed as
  // variable for the example. The lambda helper shows how to simplify the
  // filter step call.
  const auto step{[&filter](double altitude,
                            std::chrono::milliseconds step_time,
                            double acceleration) {
    filter.update(altitude);
    filter.predict(step_time, acceleration);
  }};

  step(-11.1, delta_time, 40.02 + gravity);

  assert(std::abs(1 - filter.x()[0] / -12.3) < 0.002 &&
         std::abs(1 - filter.x()[1] / 14.8) < 0.002 &&
         "The state estimates expected at 0.2% accuracy.");
  assert(std::abs(1 - filter.p()(0, 0) / 244.9) < 0.001 &&
         std::abs(1 - filter.p()(0, 1) / 211.6) < 0.001 &&
         std::abs(1 - filter.p()(1, 0) / 211.6) < 0.001 &&
         std::abs(1 - filter.p()(1, 1) / 438.8) < 0.001 &&
         "The estimate uncertainty expected at 0.1% accuracy.");

  step(18., delta_time, 39.97 + gravity);
  step(22.9, delta_time, 39.81 + gravity);
  step(19.5, delta_time, 39.75 + gravity);
  step(28.5, delta_time, 39.6 + gravity);
  step(46.5, delta_time, 39.77 + gravity);
  step(68.9, delta_time, 39.83 + gravity);
  step(48.2, delta_time, 39.73 + gravity);
  step(56.1, delta_time, 39.87 + gravity);
  step(90.5, delta_time, 39.81 + gravity);
  step(104.9, delta_time, 39.92 + gravity);
  step(140.9, delta_time, 39.78 + gravity);
  step(148., delta_time, 39.98 + gravity);
  step(187.6, delta_time, 39.76 + gravity);
  step(209.2, delta_time, 39.86 + gravity);
  step(244.6, delta_time, 39.61 + gravity);
  step(276.4, delta_time, 39.86 + gravity);
  step(323.5, delta_time, 39.74 + gravity);
  step(357.3, delta_time, 39.87 + gravity);
  step(357.4, delta_time, 39.63 + gravity);
  step(398.3, delta_time, 39.67 + gravity);
  step(446.7, delta_time, 39.96 + gravity);
  step(465.1, delta_time, 39.8 + gravity);
  step(529.4, delta_time, 39.89 + gravity);
  step(570.4, delta_time, 39.85 + gravity);
  step(636.8, delta_time, 39.9 + gravity);
  step(693.3, delta_time, 39.81 + gravity);
  step(707.3, delta_time, 39.81 + gravity);

  filter.update(748.5);

  // The Kalman gain for altitude converged to 0.12, which means that the
  // estimation weight is much higher than the measurement weight.
  assert(std::abs(1 - filter.p()(0, 0) / 49.3) < 0.001 &&
         "At this point, the altitude uncertainty px = 49.3, which means that "
         "the standard deviation of the prediction is square root of 49.3: "
         "7.02m (remember that the standard deviation of the measurement is "
         "20m).");

  filter.predict(delta_time, 39.68 + gravity);

  // At the beginning, the estimated altitude is influenced by measurements and
  // it is not aligned well with the true rocket altitude, since the
  // measurements are very noisy. But as the Kalman gain converges, the noisy
  // measurement has less influence and the estimated altitude is well aligned
  // with the true altitude. In this example we don't have any maneuvers that
  // cause acceleration changes, but if we had, the control input
  // (accelerometer) would update the state extrapolation equation.

  assert(std::abs(1 - filter.x()[0] / 831.5) < 0.001 &&
         std::abs(1 - filter.x()[1] / 222.94) < 0.001 &&
         "The state estimates expected at 0.1% accuracy.");
  assert(std::abs(1 - filter.p()(0, 0) / 54.3) < 0.01 &&
         std::abs(1 - filter.p()(0, 1) / 10.4) < 0.01 &&
         std::abs(1 - filter.p()(1, 0) / 10.4) < 0.01 &&
         std::abs(1 - filter.p()(1, 1) / 2.6) < 0.01 &&
         "The estimate uncertainty expected at 1% accuracy.");

  return 0;
}()};

} // namespace
} // namespace fcarouge::eigen::sample
