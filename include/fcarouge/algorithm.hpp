/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.2.0
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

#ifndef FCAROUGE_ALGORITHM_HPP
#define FCAROUGE_ALGORITHM_HPP

//! @file
//! @brief The collection of Kalman algorithms.
//!
//! @details Public Kalman algorithm definitions and documentation.
//!
//! @todo Implement and test.

#include "fcarouge/utility.hpp"

#include <type_traits>

namespace fcarouge {

//! @brief Produces estimates of the state variables and uncertainties.
//!
//! @details Also known as the propagation step. Implements the total
//! probability theorem. Estimate the next state by suming the known
//! probabilities.
//!
//! @note No input.
inline constexpr void predict(auto f, auto &p, auto q, auto &x) {
  x = f * x;
  p = f * p * transposed(f) + q;
}

//! @brief Produces estimates of the state variables and uncertainties.
//!
//! @details Also known as the propagation step. Implements the total
//! probability theorem. Estimate the next state by suming the known
//! probabilities.
inline constexpr void predict(auto f, auto g, auto &p, auto q, auto u,
                              auto &x) {
  x = f * x + g * u;
  p = f * p * transposed(f) + q;
}

//! @brief Updates the estimates with the outcome of a measurement.
//!
//! @details Also known as the observation or correction step. Implements the
//! Bayes' theorem. Combine one measurement and the prior estimate by applying
//! the multiplicative law.
//!
//! @note Joseph form.
inline constexpr void update(auto h, auto &p, auto r, auto &x, auto z) {
  auto i{identity_v<decltype(p)>};
  auto y{z - h * x};
  auto s{h * p * transposed(h) + r};
  auto k{p * transposed(h) / s};
  x += k * y;
  p = (i - k * h) * p * transposed(i - k * h) + k * r * transposed(k);
}

//! @brief Updates the estimates with the outcome of a measurement.
//!
//! @details Also known as the observation or correction step. Implements the
//! Bayes' theorem. Combine one measurement and the prior estimate by applying
//! the multiplicative law.
//!
//! @note Joseph form, identity observation transition.
inline constexpr void update(auto &p, auto r, auto &x, auto z) {
  auto i{identity_v<decltype(p)>};
  auto y{z - x};
  auto s{p + r};
  auto k{p / s};
  x += k * y;
  p = (i - k) * p * transposed(i - k) + k * r * transposed(k);
}

//! @brief Updates the estimates with the outcome of a measurement.
//!
//! @details Also known as the observation or correction step. Implements the
//! Bayes' theorem. Combine one measurement and the prior estimate by applying
//! the multiplicative law.
//!
//! @note Optimal gain form. Tradeoff stability for performance if gain is
//! optimal.
inline constexpr void update2(auto h, auto &p, auto r, auto &x, auto z) {
  auto i{identity_v<decltype(p)>};
  auto y{z - h * x};
  auto s{h * p * transposed(h) + r};
  auto k{p * transposed(h) / s};
  x += k * y;
  p = (i - k * h) * p;
}

//! @brief Updates the estimates with the outcome of a measurement.
//!
//! @details Also known as the observation or correction step. Implements the
//! Bayes' theorem. Combine one measurement and the prior estimate by applying
//! the multiplicative law.
//!
//! @note Optimal gain form, identity observation transition.
inline constexpr void update2(auto &p, auto r, auto &x, auto z) {
  auto y{z - x};
  auto s{p + r};
  auto k{p / s};
  x += k * y;
  p -= k * p;
}

} // namespace fcarouge

#endif // FCAROUGE_ALGORITHM_HPP
