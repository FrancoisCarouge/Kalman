#include "fcarouge/kalman.hpp"
#include "fcarouge/linalg.hpp"

#include <cassert>
#include <chrono>
#include <cmath>

// #include <iostream> ////////////////////////////////////////////////////////

namespace fcarouge::sample {
namespace {
template <auto Size> using vector = column_vector<double, Size>;
using state = vector<2>;
using output = vector<1>;
using input = vector<1>;

//! @brief
[[maybe_unused]] /*constexpr*/ auto kf_2x1x1_rocket_altitude{[] {
  using kalman = kalman<state, output, input, std::tuple<>,
                        std::tuple<std::chrono::milliseconds>>;

  kalman filter;

  filter.x(0., 0.);
  filter.p(kalman::estimate_uncertainty{{500, 0}, {0, 500}});
  filter.q([]([[maybe_unused]] const kalman::state &x,
              const std::chrono::milliseconds &delta_time) {
    const auto dt{std::chrono::duration<double>(delta_time).count()};
    return kalman::process_uncertainty{
        {0.1 * 0.1 * dt * dt * dt * dt / 4, 0.1 * 0.1 * dt * dt * dt / 2},
        {0.1 * 0.1 * dt * dt * dt / 2, 0.1 * 0.1 * dt * dt}};
  });
  filter.f([]([[maybe_unused]] const kalman::state &x,
              [[maybe_unused]] const kalman::input &u,
              const std::chrono::milliseconds &delta_time) {
    const auto dt{std::chrono::duration<double>(delta_time).count()};
    return kalman::state_transition{{1, dt}, {0, 1}};
  });
  filter.g([](const std::chrono::milliseconds &delta_time) {
    const auto dt{std::chrono::duration<double>(delta_time).count()};
    return kalman::input_control{0.0313, dt};
  });
  const double gravity{-9.8};
  const std::chrono::milliseconds delta_time{250};
  filter.predict(delta_time, -gravity);

  //   std::cout << "Result: " << filter.p()(0, 0) << std::endl;
  //   ////////////////////

  assert(std::abs(1 - filter.x()[0] / 0.3) < 0.03 &&
         std::abs(1 - filter.x()[1] / 2.45) < 0.03 &&
         "The state estimates expected at 3% accuracy.");
  assert(std::abs(1 - filter.p()(0, 0) / 531.25) < 0.001 &&
         std::abs(1 - filter.p()(0, 1) / 125) < 0.001 &&
         std::abs(1 - filter.p()(1, 0) / 125) < 0.001 &&
         std::abs(1 - filter.p()(1, 1) / 500) < 0.001 &&
         "The estimate uncertainty expected at 0.1% accuracy.");

  filter.h(kalman::output_model{1., 0.});
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

  assert(std::abs(1 - filter.p()(0, 0) / 49.3) < 0.001 &&
         "At this point, the altitude uncertainty px = 49.3, which means that "
         "the standard deviation of the prediction is square root of 49.3: "
         "7.02m (remember that the standard deviation of the measurement is "
         "20m).");

  filter.predict(delta_time, 39.68 + gravity);

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
} // namespace fcarouge::sample
