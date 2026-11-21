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

namespace fcarouge::test {
namespace {
using literals::operator""_o;

using banded = sparse_matrix<double, sparse::tridiagonal<3>>;
using lower = sparse_matrix<double, sparse::lower_triangular<3>>;
using bidiagonal = sparse_matrix<double, sparse::banded<3, 3, 1, 0>>;
using typed_banded = typed_sparse_matrix<double, sparse::tridiagonal<3>>;
using typed_dense = matrix<double, 3, 3>;

//! @test Verifies the Schur product multiplies the elements stored by both
//! operands, of the same shape, and stores the intersection of their patterns,
//! of the underlying matrices and of the typed matrices.
[[maybe_unused]] const auto test{[] -> int {
  const banded taper{{1., 0.5, 0_o}, {0.5, 1., 0.5}, {0_o, 0.5, 1.}};
  const lower value{{2., 0_o, 0_o}, {4., 6., 0_o}, {8., 10., 12.}};

  const auto product{schur_product(taper, value)};

  static_assert(std::same_as<decltype(product), const bidiagonal>);
  assert(
      (product == bidiagonal{{2., 0_o, 0_o}, {2., 6., 0_o}, {0_o, 5., 12.}}));

  const typed_banded typed_taper{taper};
  const typed_dense typed_value{{2., 4., 8.}, {4., 6., 10.}, {8., 10., 12.}};

  const auto typed_product{schur_product(typed_taper, typed_value)};

  static_assert(std::same_as<decltype(typed_product), const typed_banded>);
  assert((typed_product ==
          typed_banded{banded{{2., 2., 0_o}, {2., 6., 5.}, {0_o, 5., 12.}}}));

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
