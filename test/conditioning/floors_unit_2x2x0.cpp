/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.4
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

#include "fcarouge/kalman.hpp"
#include "fcarouge/linalg.hpp"

#include <cassert>

#include <mp-units/systems/isq/space_and_time.h>

namespace fcarouge::test {
namespace {
using mp_units::si::unit_symbols::m;
using mp_units::si::unit_symbols::m2;
using mp_units::si::unit_symbols::s;
using mp_units::si::unit_symbols::s2;
using position = mp_units::quantity<mp_units::isq::length[m]>;
using velocity = mp_units::quantity<mp_units::isq::velocity[m / s]>;

template <typename... Types> using vector = column_vector<double, Types...>;
using motion = vector<position, velocity>;
using covariance = kf::ᴀʙᵀ<motion, motion>;

//! @test Verifies the runtime, constant, and mixed floors of a filter of
//! quantities raise each diagonal variance below the square of its standard
//! deviation floor, in its units, through the element setters of a typed
//! estimate uncertainty whose elements cannot be referenced, and keep the other
//! elements.
[[maybe_unused]] const auto test{[] -> int {
  const auto declare{[](const auto &policy) -> auto {
    covariance p;
    p.at<0, 0>(0.25 * m2);
    p.at<0, 1>(0.125 * m2 / s);
    p.at<1, 0>(0.125 * m2 / s);
    p.at<1, 1>(4. * m2 / s2);

    return kalman{state{motion{position{0. * m}, velocity{0. * m / s}}},
                  output<motion>,
                  estimate_uncertainty{p},
                  process_uncertainty{covariance{}},
                  output_uncertainty{covariance{}},
                  policy};
  }};

  auto runtime{declare(floors{position{1. * m}, velocity{1. * m / s}})};
  auto constant{declare(floors{cw<1. * m>, cw<1. * m / s>})};
  auto mixed{declare(floors{position{1. * m}, cw<1. * m / s>})};

  runtime.predict();
  constant.predict();
  mixed.predict();

  assert((runtime.p().at<0, 0>() == 1. * m2));
  assert((runtime.p().at<1, 1>() == 4. * m2 / s2));
  assert((runtime.p().at<0, 1>() == 0.125 * m2 / s));
  assert((runtime.p().at<1, 0>() == 0.125 * m2 / s));
  assert((constant.p().at<0, 0>() == runtime.p().at<0, 0>()));
  assert((constant.p().at<1, 1>() == runtime.p().at<1, 1>()));
  assert((mixed.p().at<0, 0>() == runtime.p().at<0, 0>()));
  assert((mixed.p().at<1, 1>() == runtime.p().at<1, 1>()));

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
