#include "fcarouge/eigen/kalman.hpp"

#include <unsupported/Eigen/MatrixFunctions>

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <utility>

namespace fcarouge::eigen::sample {
namespace {
//! @brief Estimating the Position and Velocity of a Falling Package
//!
//! @copyright This example is transcribed from How Kalman Filters Work
//! copyright 2016 Tucker McClure of An Uncommon Lab.
//!
//! @see
//! https://github.com/tuckermcclure/how-kalman-filters-work-examples/blob/master/lkf_demo.m
//!
//! @details ...
//!
//! @example falling_package.cpp
[[maybe_unused]] auto falling_package{[] {
  using kalman = kalman<vector<double, 4>, vector<double, 2>, void>;
  //, std::chrono::milliseconds>;
  kalman k;

  k.x(-1., 8., 1., 0.); // [m; m; m.s^-1; m.s^-1]

  // 0.5m standard deviation.
  const kalman::output_uncertainty r{0.5 * 0.5 * Eigen::Matrix2d::Identity()};
  k.r(r);

  kalman::estimate_uncertainty p{kalman::estimate_uncertainty::Zero()};
  p.block<2, 2>(0, 0) = r;
  p.block<2, 2>(2, 2) = 2 * 2 * Eigen::Matrix2d::Identity();
  k.p(std::move(p));

  // 0.5m.s^-2 standard deviation.
  const kalman::process_uncertainty q{0.5 * 0.5 * Eigen::Matrix4d::Identity()};
  k.q(q);

  kalman::output_model h{kalman::output_model::Zero()};
  h.block<2, 2>(0, 0) = Eigen::Matrix2d::Identity();
  k.h(std::move(h));

  const auto drag_coeffecient{4.4}; // [-]
  const auto mass{1.};              // [kg]
  const auto gravity{-9.81};
  const Eigen::Vector2d gravity2d{0., gravity}; // [m.s^-2]
  const auto terminal_velocity{std::sqrt(-gravity / drag_coeffecient)};
  kalman::state nominal{0., 0., 0., -terminal_velocity};

  Eigen::Matrix4d a{Eigen::Matrix4d::Zero()};
  a.block<2, 2>(0, 2) = Eigen::Matrix2d::Identity();
  a.block<2, 2>(2, 2) = -drag_coeffecient / mass * terminal_velocity *
                        Eigen::Matrix2d::Identity();

  const auto delta_time{0.1}; // [s]
  const kalman::state_transition f{(a * delta_time).exp()};
  k.f(f);

  // // To utils?
  // // Post to
  // //
  // https://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
  // const auto sample{
  //   []<typename value_type, int rows, int columns>(
  //       const Eigen::Matrix<value_type, rows, columns> &covariance) {
  //     using vector = Eigen::Vector<value_type, rows>;
  //     using svd_decomposer =
  //         Eigen::JacobiSVD<Eigen::Matrix<value_type, rows, columns>,
  //                          Eigen::ComputeThinU>;

  //     const svd_decomposer decomposed(covariance);
  //     const auto generate{ [] {
  //       static std::random_device random_device;
  //       static std::default_random_engine random_engine{ random_device() };
  //       static std::normal_distribution distribute;

  //       return distribute(random_engine);
  //     } };

  //     return vector{
  //       decomposed.matrixU() *
  //       decomposed.singularValues().array().sqrt().matrix().asDiagonal() *
  //       vector::NullaryExpr(generate)
  //     };
  //   }
  // };

  // // Simulation
  // constexpr auto steps{ 50 };
  // std::vector<kalman::state> true_states{ steps, kalman::state::Zero() };
  // true_states[0] = k.x();
  // std::vector<kalman::state> process_noises{ steps };
  // for (auto &value : process_noises) {
  //   value = sample(k.q());
  // }
  // std::vector<kalman::output> measurements{ steps, kalman::output::Zero() };
  // measurements[0] = kalman::output{ k.x()(0), k.x()(1) } + sample(k.r());
  // std::vector<kalman::output> measurement_noises{ steps,
  //                                                 kalman::output::Zero() };
  // for (auto &value : measurement_noises) {
  //   value = sample(k.r());
  // }
  // std::vector<kalman::state> estimate_states{ steps, kalman::state::Zero() };
  // const auto pp{ sample(
  //     Eigen::Matrix<double, 2, 2>{ k.p().block<2, 2>(2, 2) }) };
  // estimate_states[0] =
  //     kalman::state{ measurements[0](0), measurements[0](1),
  //                    true_states[0](0) + pp(0), true_states[0](1) + pp(1) };
  // std::vector<kalman::estimate_uncertainty> estimate_uncertainties{
  //   steps, kalman::estimate_uncertainty::Zero()
  // };
  // estimate_uncertainties[0] = k.p();

  return 0;
}()};

} // namespace
} // namespace fcarouge::eigen::sample
