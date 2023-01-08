#include "fcarouge/kalman.hpp"

#include <Eigen/Eigen>

#include <cassert>
#include <chrono>
#include <cmath>
#include <ranges>

namespace fcarouge::test {
namespace {

template <typename Type, auto Size> using vector = Eigen::Vector<Type, Size>;
using kalman = kalman<vector<double, 2>, double, double, std::divides<void>,
                      std::tuple<>, std::tuple<double>>;

// operator | ()

const kalman::output altitudes[]{
    -32.4, -11.1, 18.0,  22.9,  19.5,  28.5,  46.5,  68.9,  48.2,  56.1,
    90.5,  104.9, 140.9, 148.,  187.6, 209.2, 244.6, 276.4, 323.5, 357.3,
    357.4, 398.3, 446.7, 465.1, 529.4, 570.4, 636.8, 693.3, 707.3, 748.5};

// C++23: const auto delta_times{std::views::repeat(.25) |
// std::views::take(std::size(altitudes))};
const double delta_times[]{.25, .25, .25, .25, .25, .25, .25, .25, .25, .25,
                           .25, .25, .25, .25, .25, .25, .25, .25, .25, .25,
                           .25, .25, .25, .25, .25, .25, .25, .25, .25, .25};

const kalman::input accelerations[]{
    29.92, 30.22, 30.17, 30.01, 29.95, 29.8,  29.97, 30.03, 29.93, 30.07,
    30.01, 30.12, 29.98, 30.18, 29.96, 30.06, 29.81, 30.06, 29.94, 30.07,
    29.83, 29.87, 30.16, 30,    30.09, 30.05, 30.1,  30.01, 30.01, 29.88};

static_assert(std::size(altitudes) == std::size(accelerations));
static_assert(std::size(delta_times) == std::size(accelerations));

//! @test
[[maybe_unused]] auto pipe0{[] {
  kalman filter;

  filter.x(0., 0.);
  filter.p(kalman::estimate_uncertainty{{500, 0}, {0, 500}});
  filter.q([]([[maybe_unused]] const kalman::state &x, const double &dt) {
    return kalman::process_uncertainty{
        {0.1 * 0.1 * dt * dt * dt * dt / 4, 0.1 * 0.1 * dt * dt * dt / 2},
        {0.1 * 0.1 * dt * dt * dt / 2, 0.1 * 0.1 * dt * dt}};
  });
  filter.f([]([[maybe_unused]] const kalman::state &x,
              [[maybe_unused]] const kalman::input &u, const double &dt) {
    return kalman::state_transition{{1, dt}, {0, 1}};
  });
  filter.g([](const double &dt) { return kalman::input_control{0.0313, dt}; });
  filter.h(kalman::output_model{1., 0.});
  filter.r(kalman::output_uncertainty{400.});

  filter.predict(.25, 9.8);

  const auto step{
      [&filter](double altitude, double step_time, double acceleration) {
        filter.update(altitude);
        filter.predict(step_time, acceleration);
      }};

  // C++23
  // for (auto &&data : std::views::zip(altitudes, delta_times, accelerations))
  // {
  //   step(std::get<0>(data), std::get<1>(data), std::get<2>(data))
  // }

  step(-32.4, .25, 29.92);
  step(-11.1, .25, 30.22);
  step(18.0, .25, 30.17);
  step(22.9, .25, 30.01);
  step(19.5, .25, 29.95);
  step(28.5, .25, 29.8);
  step(46.5, .25, 29.97);
  step(68.9, .25, 30.03);
  step(48.2, .25, 29.93);
  step(56.1, .25, 30.07);
  step(90.5, .25, 30.01);
  step(104.9, .25, 30.12);
  step(140.9, .25, 29.98);
  step(148., .25, 30.18);
  step(187.6, .25, 29.96);
  step(209.2, .25, 30.06);
  step(244.6, .25, 29.81);
  step(276.4, .25, 30.06);
  step(323.5, .25, 29.94);
  step(357.3, .25, 30.07);
  step(357.4, .25, 29.83);
  step(398.3, .25, 29.87);
  step(446.7, .25, 30.16);
  step(465.1, .25, 30);
  step(529.4, .25, 30.09);
  step(570.4, .25, 30.05);
  step(636.8, .25, 30.1);
  step(693.3, .25, 30.01);
  step(707.3, .25, 30.01);
  step(748.5, .25, 29.88);

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

struct datum {
  double altitude;
  double delta_time;
  double acceleration;
};

const datum data[]{
    {-32.4, .25, 29.92}, {-11.1, .25, 30.22}, {18.0, .25, 30.17},
    {22.9, .25, 30.01},  {19.5, .25, 29.95},  {28.5, .25, 29.8},
    {46.5, .25, 29.97},  {68.9, .25, 30.03},  {48.2, .25, 29.93},
    {56.1, .25, 30.07},  {90.5, .25, 30.01},  {104.9, .25, 30.12},
    {140.9, .25, 29.98}, {148., .25, 30.18},  {187.6, .25, 29.96},
    {209.2, .25, 30.06}, {244.6, .25, 29.81}, {276.4, .25, 30.06},
    {323.5, .25, 29.94}, {357.3, .25, 30.07}, {357.4, .25, 29.83},
    {398.3, .25, 29.87}, {446.7, .25, 30.16}, {465.1, .25, 30},
    {529.4, .25, 30.09}, {570.4, .25, 30.05}, {636.8, .25, 30.1},
    {693.3, .25, 30.01}, {707.3, .25, 30.01}, {748.5, .25, 29.88}};

//! @test
[[maybe_unused]] auto pipe1{[] {
  kalman filter;

  filter.x(0., 0.);
  filter.p(kalman::estimate_uncertainty{{500, 0}, {0, 500}});
  filter.q([]([[maybe_unused]] const kalman::state &x, const double &dt) {
    return kalman::process_uncertainty{
        {0.1 * 0.1 * dt * dt * dt * dt / 4, 0.1 * 0.1 * dt * dt * dt / 2},
        {0.1 * 0.1 * dt * dt * dt / 2, 0.1 * 0.1 * dt * dt}};
  });
  filter.f([]([[maybe_unused]] const kalman::state &x,
              [[maybe_unused]] const kalman::input &u, const double &dt) {
    return kalman::state_transition{{1, dt}, {0, 1}};
  });
  filter.g([](const double &dt) { return kalman::input_control{0.0313, dt}; });
  filter.h(kalman::output_model{1., 0.});
  filter.r(kalman::output_uncertainty{400.});

  filter.predict(.25, 9.8);

  const auto step{
      [&filter](double altitude, double step_time, double acceleration) {
        filter.update(altitude);
        filter.predict(step_time, acceleration);
      }};

  step(-32.4, .25, 29.92);
  step(-11.1, .25, 30.22);
  step(18.0, .25, 30.17);
  step(22.9, .25, 30.01);
  step(19.5, .25, 29.95);
  step(28.5, .25, 29.8);
  step(46.5, .25, 29.97);
  step(68.9, .25, 30.03);
  step(48.2, .25, 29.93);
  step(56.1, .25, 30.07);
  step(90.5, .25, 30.01);
  step(104.9, .25, 30.12);
  step(140.9, .25, 29.98);
  step(148., .25, 30.18);
  step(187.6, .25, 29.96);
  step(209.2, .25, 30.06);
  step(244.6, .25, 29.81);
  step(276.4, .25, 30.06);
  step(323.5, .25, 29.94);
  step(357.3, .25, 30.07);
  step(357.4, .25, 29.83);
  step(398.3, .25, 29.87);
  step(446.7, .25, 30.16);
  step(465.1, .25, 30);
  step(529.4, .25, 30.09);
  step(570.4, .25, 30.05);
  step(636.8, .25, 30.1);
  step(693.3, .25, 30.01);
  step(707.3, .25, 30.01);
  step(748.5, .25, 29.88);

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
} // namespace fcarouge::test
