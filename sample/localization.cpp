#include "fcarouge/eigen/kalman.hpp"

#include <cassert>
#include <chrono>

namespace fcarouge::sample {
namespace {
//! @test
//!
//! @copyright This example is transcribed from  copyright .
//!
//! @see https://github.com/rsasaki0109/kalman_filter_localization
//!
//! @details
//!
//! error state kalman filter!
//!
//!
//! R?
//! Variance of GNSS receiver x and y position: 0.1 m^2. long/lat?? std dev??
//! Variance of GNSS receiver z position: 0.15 m^2. altitude?? std dev??
//! Variance of angular velocity sensor: 0.01 (deg.sec^-1)^2. xyz?? std dev??
//! Variance of accelerometer: 0.01 (m.sec^-2)^2. xyz?? std dev??
//!
//!
//!
[[maybe_unused]] auto rocket_altitude{[] {
  // X: [px, py, pz, vx, vy, vz, qx, qy, qz, qw]
  // Z: [px, py, z]
  // using kalman =
  //     eigen::kalman<double, 10, 3, 0, 9, std::chrono::nanoseconds>;
  // kalman k;

  // Initialization
  // X: [px, py, pz, vx, vy, vz, qx, qy, qz, qw]
  // k.x = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };

  // // Error state: [dx, dy, dz, dvx, dvy, dvz, dthx, dthy, dthz]
  // // dx: [dx dy dz dvx dvy dvz dthx dthy dthz]
  // // dx = K * (y - part of x?)
  // // Internal to update()

  // // TODO: Add L support around Q in extrapolate_covariance
  // // TODO: Add error state vector and size supports
  // // TODO: new equation in predict for dx/update state?

  // // P: ESxES
  // // H: ZxES

  // // 9x9? error statex? Not the XxX?
  // k.p = kalman::estimate_uncertainty{
  //   { 100, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 100, 0, 0, 0, 0, 0, 0, 0 },
  //   { 0, 0, 100, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 100, 0, 0, 0, 0, 0 },
  //   { 0, 0, 0, 0, 100, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 100, 0, 0, 0 },
  //   { 0, 0, 0, 0, 0, 0, 100, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 100, 0 },
  //   { 0, 0, 0, 0, 0, 0, 0, 0, 100 }
  // };

  // // 6x6
  // k.noise_process_q = [](const std::chrono::milliseconds &delta_time) {
  //   const auto dt{ std::chrono::duration<double>(delta_time).count() };
  //   const double var_imu_acc{ 0.01 };
  //   const double var_imu_w{ 0.01 };
  //   return kalman::process_uncertainty{
  //     { var_imu_acc * dt * dt, 0, 0, 0, 0, 0 },
  //     { 0, var_imu_acc * dt * dt, 0, 0, 0, 0 },
  //     { 0, 0, var_imu_acc * dt * dt, 0, 0, 0 },
  //     { 0, 0, 0, var_imu_w * dt * dt, 0, 0 },
  //     { 0, 0, 0, 0, var_imu_w * dt * dt, 0 },
  //     { 0, 0, 0, 0, 0, var_imu_w * dt * dt }
  //   };
  // };

  // // 9x9
  // k.transition_state_f = [](const kalman::state &x,
  //                           const std::chrono::milliseconds &delta_time,
  //                           const Eigen::Vector3d &linear_acceleration) {
  //   const auto dt{ std::chrono::duration<double>(delta_time).count() };
  //   kalman::state_transition f{
  //     { 1, 0, 0, dt, 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, dt, 0, 0, 0, 0 },
  //     { 0, 0, 1, 0, 0, dt, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0, 0, 0, 0 },
  //     { 0, 0, 0, 0, 1, 0, 0, 0, 0 },  { 0, 0, 0, 0, 0, 1, 0, 0, 0 },
  //     { 0, 0, 0, 0, 0, 0, 1, 0, 0 },  { 0, 0, 0, 0, 0, 0, 0, 1, 0 },
  //     { 0, 0, 0, 0, 0, 0, 0, 0, 1 }
  //   };

  //   Eigen::Matrix3d linear_acceleration_skew{
  //     { 0, -linear_acceleration(2), linear_acceleration(1) },
  //     { linear_acceleration(2), 0, -linear_acceleration(0) },
  //     { -linear_acceleration(1), linear_acceleration(0), 0 }
  //   };
  //   // Quaternion must be normalized. Need check?
  //   Eigen::Matrix3d rotation{
  //     Eigen::Quaterniond(x(9), x(6), x(7), x(8)).toRotationMatrix()
  //   };
  //   f.block<3, 3>(3, 6) = rotation * (-linear_acceleration_skew) * dt;
  //   return f;
  // };

  // // k.l = []{
  // //   return kalman::ltype{
  // //     {0, 0, 0, 0, 0, 0},
  // //     {0, 0, 0, 0, 0, 0},
  // //     {0, 0, 0, 0, 0, 0},
  // //     {1, 0, 0, 0, 0, 0},
  // //     {0, 1, 0, 0, 0, 0},
  // //     {0, 0, 1, 0, 0, 0},
  // //     {0, 0, 0, 1, 0, 0},
  // //     {0, 0, 0, 0, 1, 0},
  // //     {0, 0, 0, 0, 0, 1}
  // //   };
  // // };

  // //   const std::chrono::nanoseconds delta_time{ 1234 };
  // // Eigen::Vector3d gyro;
  // // Eigen::Vector3d linear_acceleration;
  // //   k.predict(delta_time, gyro, linear_acceleration);

  // // 3x9
  //   k.h = [] {
  //     return kalman::observation{ { 1, 0, 0, 0, 0, 0, 0, 0, 0 },
  //                                 { 0, 1, 0, 0, 0, 0, 0, 0, 0 },
  //                                 { 0, 0, 1, 0, 0, 0, 0, 0, 0 } };
  //   };

  // // 3x3
  // k.noise_observation_r = [](const Eigen::Vector3d &variance) {
  //   return kalman::output_uncertainty{ { variance(0), 0, 0 },
  //                                                 { 0, variance(1), 0 },
  //                                                 { 0, 0, variance(2) } };
  // };

  // // Eigen::Vector3d y;
  // // Eigen::Vector3d variance;
  // // k.update(y, variance);
  // // k.predict...

  return 0;
}()};

} // namespace
} // namespace fcarouge::sample
