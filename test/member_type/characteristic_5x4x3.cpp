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

#include <concepts>
#include <utility>

namespace fcarouge::test {
namespace {
template <auto Size> using vector = column_vector<double, Size>;
template <auto Row, auto Column> using matrix = matrix<double, Row, Column>;

//! @test Verifies the characteristics are read as constant references from an
//! lvalue filter, without copies, and as values from an rvalue filter, without
//! references dangling past the full-expression.
using filter = decltype(kalman{state{vector<5>{0., 0., 0., 0., 0.}},
                               output<vector<4>>, input<vector<3>>});

static_assert(
    std::same_as<decltype(std::declval<filter &>().x()), const vector<5> &>);
static_assert(
    std::same_as<decltype(std::declval<filter &>().z()), const vector<4> &>);
static_assert(
    std::same_as<decltype(std::declval<filter &>().u()), const vector<3> &>);
static_assert(
    std::same_as<decltype(std::declval<filter &>().p()), const matrix<5, 5> &>);
static_assert(
    std::same_as<decltype(std::declval<filter &>().q()), const matrix<5, 5> &>);
static_assert(
    std::same_as<decltype(std::declval<filter &>().r()), const matrix<4, 4> &>);
static_assert(
    std::same_as<decltype(std::declval<filter &>().f()), const matrix<5, 5> &>);
static_assert(
    std::same_as<decltype(std::declval<filter &>().h()), const matrix<4, 5> &>);
static_assert(
    std::same_as<decltype(std::declval<filter &>().g()), const matrix<5, 3> &>);
static_assert(
    std::same_as<decltype(std::declval<filter &>().k()), const matrix<5, 4> &>);
static_assert(
    std::same_as<decltype(std::declval<filter &>().y()), const vector<4> &>);
static_assert(
    std::same_as<decltype(std::declval<filter &>().s()), const matrix<4, 4> &>);

static_assert(std::same_as<decltype(std::declval<const filter &>().x()),
                           const vector<5> &>);
static_assert(std::same_as<decltype(std::declval<const filter &>().p()),
                           const matrix<5, 5> &>);

static_assert(std::same_as<decltype(std::declval<filter>().x()), vector<5>>);
static_assert(std::same_as<decltype(std::declval<filter>().p()), matrix<5, 5>>);
static_assert(
    std::same_as<decltype(std::declval<const filter>().x()), vector<5>>);
} // namespace
} // namespace fcarouge::test
