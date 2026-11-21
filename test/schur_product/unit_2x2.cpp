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

#include "fcarouge/linalg.hpp"

#include <cassert>
#include <concepts>
#include <tuple>

#include <mp-units/systems/isq/space_and_time.h>

namespace fcarouge::test {
namespace {
using mp_units::si::unit_symbols::m;
using mp_units::si::unit_symbols::m2;
using mp_units::si::unit_symbols::s;
using mp_units::si::unit_symbols::s2;
using position = mp_units::quantity<mp_units::isq::length[m]>;
using velocity = mp_units::quantity<mp_units::isq::velocity[m / s]>;

using covariance = matrix<double, std::tuple<position, velocity>,
                          std::tuple<position, velocity>>;
using taper = typed_sparse_matrix<double, sparse::diagonal<2>>;
using localized = typed_sparse_matrix<double, sparse::diagonal<2>,
                                      std::tuple<position, velocity>,
                                      std::tuple<position, velocity>>;

//! @test Verifies the Schur product of a dimensionless taper by a matrix of
//! quantities keeps the units of the elements of the matrix, in the pattern of
//! the taper.
[[maybe_unused]] const auto test{[] -> int {
  covariance p;
  p.at<0, 0>(4. * m2);
  p.at<0, 1>(1. * m2 / s);
  p.at<1, 0>(1. * m2 / s);
  p.at<1, 1>(2. * m2 / s2);

  const auto product{schur_product(
      taper{sparse_matrix<double, sparse::diagonal<2>>{0.5, 0.25}}, p)};

  static_assert(std::same_as<decltype(product), const localized>);
  assert((product.at<0, 0>() == 2. * m2));
  assert((product.at<1, 1>() == 0.5 * m2 / s2));

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
