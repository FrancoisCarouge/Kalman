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
using position = mp_units::quantity<mp_units::isq::length[m], float>;
using velocity = mp_units::quantity<mp_units::isq::velocity[m / s], float>;

template <typename... Types> using vector = column_vector<float, Types...>;
using motion = vector<position, velocity>;
using covariance = kf::ᴀʙᵀ<motion, motion>;

//! @test Verifies the symmetrization of a single precision estimate uncertainty
//! of quantities averages it with its transpose in its own representation,
//! without promoting its elements to double precision.
[[maybe_unused]] const auto test{[] -> int {
  covariance p;
  p.at<0, 0>(0.25F * m2);
  p.at<0, 1>(0.5F * m2 / s);
  p.at<1, 0>(0.25F * m2 / s);
  p.at<1, 1>(4.F * m2 / s2);

  kalman filter{state{motion{position{0.F * m}, velocity{0.F * m / s}}},
                output<motion>,
                estimate_uncertainty{p},
                process_uncertainty{covariance{}},
                output_uncertainty{covariance{}},
                symmetrize};

  filter.predict();

  assert((filter.p().at<0, 0>() == 0.25F * m2));
  assert((filter.p().at<0, 1>() == 0.375F * m2 / s));
  assert((filter.p().at<1, 0>() == 0.375F * m2 / s));
  assert((filter.p().at<1, 1>() == 4.F * m2 / s2));

  filter.update(motion{position{1.F * m}, velocity{2.F * m / s}});

  assert((filter.p().at<0, 1>() == filter.p().at<1, 0>()));

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
