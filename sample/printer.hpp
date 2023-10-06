/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.3.0
https://github.com/FrancoisCarouge/Kalman

SPDX-License-Identifier: Unlicense

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org> */

#ifndef FCAROUGE_PRINTER_HPP
#define FCAROUGE_PRINTER_HPP

//! @file
//! @brief ...
//!
//! @details ...

// #include "internal/utility.hpp"
#include <iostream>
#include <ostream>
#include <print>

namespace fcarouge {

// Decorator?
template <typename KalmanFilter> class printer final {

public:
  using state = typename KalmanFilter::state;
  using output = typename KalmanFilter::output;
  using input = typename KalmanFilter::input;
  using estimate_uncertainty = typename KalmanFilter::estimate_uncertainty;
  using process_uncertainty = typename KalmanFilter::process_uncertainty;
  using output_uncertainty = typename KalmanFilter::output_uncertainty;
  using state_transition = typename KalmanFilter::state_transition;
  using output_model = typename KalmanFilter::output_model;
  using input_control = typename KalmanFilter::input_control;
  using gain = typename KalmanFilter::gain;
  using innovation = typename KalmanFilter::innovation;
  using innovation_uncertainty = typename KalmanFilter::innovation_uncertainty;

  inline constexpr explicit printer(KalmanFilter &&kalman_filter)
      : filter{std::forward<KalmanFilter>(kalman_filter)} {
    std::println("{{\"event\": \"construction\", \"filter\":{}}}", filter);
  }

  inline constexpr printer(printer &&other) noexcept = default;
  inline constexpr auto operator=(printer &&other) noexcept
      -> printer & = default;

  inline constexpr ~printer() {
    std::println("{{\"event\": \"destruction\", \"filter\":{}}}", filter);
    std::flush(std::cout);
  }

  inline constexpr auto x() const -> const state &;
  inline constexpr auto x() -> state & { return filter.x(); }
  inline constexpr void x(const auto &value, const auto &...values) {
    filter.x(value, values...);
    std::println("{{\"event\": \"x\", \"filter\":{}}}", filter);
  }
  // inline constexpr auto z() const -> const output &;
  // inline constexpr auto u() const
  //     -> const input &requires(not std::is_same_v<Input, void>);
  inline constexpr auto p() const -> const estimate_uncertainty &;
  inline constexpr auto p() -> estimate_uncertainty & { return filter.p(); }
  inline constexpr void p(const auto &value, const auto &...values) {
    filter.p(value, values...);
    std::println("{{\"event\": \"p\", \"filter\":{}}}", filter);
  }
  inline constexpr auto q() const -> const process_uncertainty &;
  inline constexpr auto q() -> process_uncertainty &;
  inline constexpr void q(const auto &value, const auto &...values);
  inline constexpr auto r() const -> const output_uncertainty &;
  inline constexpr auto r() -> output_uncertainty &;
  inline constexpr void r(const auto &value, const auto &...values) {
    filter.r(value, values...);
    std::println("{{\"event\": \"r\", \"filter\":{}}}", filter);
  }
  // inline constexpr auto f() const -> const state_transition &;
  // inline constexpr auto f() -> state_transition &;
  // inline constexpr void f(const auto &value, const auto &...values);
  // inline constexpr auto h() const -> const output_model &;
  // inline constexpr auto h() -> output_model &;
  // inline constexpr void h(const auto &value, const auto &...values);
  // inline constexpr auto g() const
  //     -> const input_control &requires(not std::is_same_v<Input, void>);
  // inline constexpr auto g()
  //     -> input_control &requires(not std::is_same_v<Input, void>);
  // inline constexpr void g(const auto &value, const auto &...values)
  //   requires(not std::is_same_v<Input, void>);
  // inline constexpr auto k() const -> const gain &;
  // inline constexpr auto y() const -> const innovation &;
  // inline constexpr auto s() const -> const innovation_uncertainty &;
  // inline constexpr void transition(const auto &callable);
  // inline constexpr void observation(const auto &callable);
  inline constexpr void predict(const auto &...arguments) {
    filter.predict(arguments...);
    std::println("{{\"event\": \"predict\", \"filter\":{}}}", filter);
  }
  // template <std::size_t Position> inline constexpr auto predict() const;
  inline constexpr void update(const auto &...arguments) {
    filter.update(arguments...);
    std::println("{{\"event\": \"update\", \"filter\":{}}}", filter);
  }
  // template <std::size_t Position> inline constexpr auto update() const;

private:
  KalmanFilter filter;
};

} // namespace fcarouge

#endif // FCAROUGE_PRINTER_HPP
