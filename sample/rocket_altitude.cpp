#include "fcarouge/kalman_eigen.hpp"

#include <cassert>
#include <chrono>

namespace fcarouge::sample
{
namespace
{
//! @test Estimating the Rocket Altitude
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
[[maybe_unused]] auto rocket_altitude{ [] {
  // A 2x1x1 filter, constant acceleration dynamic model, no control, step time.
  using kalman =
      fcarouge::eigen::kalman<double, 2, 1, 1, std::chrono::milliseconds>;
  kalman k;

  // Initialization
  // We don't know the rocket location; we will set initial position and
  // velocity to 0.
  k.x(0, 0);

  // Since our initial state vector is a guess, we will set a very high estimate
  // uncertainty. The high estimate uncertainty results in high Kalman gain,
  // giving a high weight to the measurement.
  k.p(kalman::estimate_uncertainty{ { 500, 0 }, { 0, 500 } });

  // Prediction
  // We will assume a discrete noise model - the noise is different at each time
  // period, but it is constant between time periods. In our previous example,
  // we used the system's random variance in acceleration σ^2 as a multiplier of
  // the process noise matrix. But here, we have an accelerometer that measures
  // the system random acceleration. The accelerometer error v is much lower
  // than system's random acceleration, therefore we use ϵ^2 as a multiplier of
  // the process noise matrix. This makes our estimation uncertainty much lower!
  k.q([](const std::chrono::milliseconds &delta_time) {
    const auto dt{ std::chrono::duration<double>(delta_time).count() };
    return kalman::process_uncertainty{
      { 0.1 * 0.1 * dt * dt * dt * dt / 4, 0.1 * 0.1 * dt * dt * dt / 2 },
      { 0.1 * 0.1 * dt * dt * dt / 2, 0.1 * 0.1 * dt * dt }
    };
  });

  // The state transition matrix F would be:
  k.f([](const std::chrono::milliseconds &delta_time) {
    const auto dt{ std::chrono::duration<double>(delta_time).count() };
    return kalman::state_transition{ { 1, dt }, { 0, 1 } };
  });

  // The control matrix G would be:
  k.g([](const std::chrono::milliseconds &delta_time) {
    const auto dt{ std::chrono::duration<double>(delta_time).count() };
    return kalman::input_control{ 0.0313, dt };
  });

  // We also don't know what the rocket acceleration is, but we can assume that
  // it's greater than zero. Let's assume: u0 = g
  const double gravitational_acceleration{ -9.8 }; // m.s^-2
  const std::chrono::milliseconds delta_time{ 250 };
  k.predict(delta_time, gravitational_acceleration);

  // Measure and Update
  // The dimension of zn is 1x1 and the dimension of xn is 2x1, so the dimension
  // of the observation matrix H will be 1x2.
  k.h(1, 0);

  // For the sake of the example simplicity, we will assume a constant
  // measurement uncertainty: R1 = R2...Rn-1 = Rn = R.
  k.r(400);

  // The measurement values: z1 = -32.40, u1 = 39.72.
  // And so on, every measurements period: Δt = 250ms (constant, as variable).
  k.observe(-32.40);
  k.predict(delta_time, 39.72);
  k.observe(-11.1);
  k.predict(delta_time, 40.02);
  k.observe(18);
  k.predict(delta_time, 39.97);
  k.observe(22.9);
  k.predict(delta_time, 39.81);
  k.observe(19.5);
  k.predict(delta_time, 39.75);
  k.observe(28.5);
  k.predict(delta_time, 39.6);
  k.observe(46.5);
  k.predict(delta_time, 39.77);
  k.observe(68.9);
  k.predict(delta_time, 39.83);
  k.observe(48.2);
  k.predict(delta_time, 39.73);
  k.observe(56.1);
  k.predict(delta_time, 39.87);
  k.observe(90.5);
  k.predict(delta_time, 39.81);
  k.observe(104.9);
  k.predict(delta_time, 39.92);
  k.observe(140.9);
  k.predict(delta_time, 39.78);
  k.observe(148);
  k.predict(delta_time, 39.98);
  k.observe(187.6);
  k.predict(delta_time, 39.76);
  k.observe(209.2);
  k.predict(delta_time, 39.86);
  k.observe(244.6);
  k.predict(delta_time, 39.61);
  k.observe(276.4);
  k.predict(delta_time, 39.86);
  k.observe(323.5);
  k.predict(delta_time, 39.74);
  k.observe(357.3);
  k.predict(delta_time, 39.87);
  k.observe(357.4);
  k.predict(delta_time, 39.63);
  k.observe(398.3);
  k.predict(delta_time, 39.67);
  k.observe(446.7);
  k.predict(delta_time, 39.96);
  k.observe(465.1);
  k.predict(delta_time, 39.8);
  k.observe(529.4);
  k.predict(delta_time, 39.89);
  k.observe(570.4);
  k.predict(delta_time, 39.85);
  k.observe(636.8);
  k.predict(delta_time, 39.9);
  k.observe(693.3);
  k.predict(delta_time, 39.81);
  k.observe(707.3);
  k.predict(delta_time, 39.81);
  k.observe(748.5);

  // The Kalman gain for altitude converged to 0.12, which means that the
  // estimation weight is much higher than the measurement weight.
  assert(49.3 - 0.01 < k.p()(0, 0) && k.p()(0, 0) < 49.3 + 0.0001 &&
         "At this point, the altitude uncertainty px = 49.3, which means that "
         "the standard deviation of the prediction is square root of 49.3: "
         "7.02m (remember that the standard deviation of the measurement is "
         "20m).");

  k.predict(delta_time, 39.68);

  // At the beginning, the estimated altitude is influenced by measurements and
  // it is not aligned well with the true rocket altitude, since the
  // measurements are very noisy. But as the Kalman gain converges, the noisy
  // measurement has less influence and the estimated altitude is well aligned
  // with the true altitude. In this example we don't have any maneuvers that
  // cause acceleration changes, but if we had, the control input
  // (accelerometer) would update the state extrapolation equation.
  assert(831.5 - 0.001 < k.x()(0) && k.x()(0) < 831.5 + 54 &&
         222.94 - 0.001 < k.x()(1) && k.x()(1) < 222.94 + 40);

  return 0;
}() };

} // namespace
} // namespace fcarouge::sample
