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

namespace fcarouge::test {
namespace {
//! @brief A single-dimension algebraic type without one and zero members.
class value {
public:
  constexpr value() = default;

  constexpr explicit value(double initial) : datum{initial} {}

  [[nodiscard]] constexpr auto operator==(const value &) const
      -> bool = default;

  [[nodiscard]] constexpr auto operator+(const value &rhs) const -> value {
    return value{datum + rhs.datum};
  }

  [[nodiscard]] constexpr auto operator-(const value &rhs) const -> value {
    return value{datum - rhs.datum};
  }

  [[nodiscard]] constexpr auto operator*(const value &rhs) const -> value {
    return value{datum * rhs.datum};
  }

  [[nodiscard]] constexpr auto operator/(const value &rhs) const -> value {
    return value{datum / rhs.datum};
  }

private:
  double datum{};
};
} // namespace
} // namespace fcarouge::test

//! @brief The constant one value, given through the factory.
template <>
struct fcarouge::kalman_filter::internal::ones<fcarouge::test::value> {
  [[nodiscard]] static constexpr auto operator()() {
    return fcarouge::test::value{1.};
  }
};

//! @brief The constant zero value, given through the factory.
template <>
struct fcarouge::kalman_filter::internal::zeros<fcarouge::test::value> {
  [[nodiscard]] static constexpr auto operator()() {
    return fcarouge::test::value{0.};
  }
};

namespace fcarouge::test {
namespace {
//! @test Verifies a filter of a type with constant one and zero values given
//! through the factories is constant-initialized, in read-only data, and
//! filters in constant evaluation.
[[maybe_unused]] constexpr kalman rom_filter{state{value{0.}}, output<value>};

static_assert(rom_filter.p() == value{1.});
static_assert(rom_filter.x() == value{0.});
static_assert([] -> bool {
  kalman filter{state{value{0.}}, output<value>};

  filter.update(value{1.});

  return filter.x() == value{1.} && filter.p() == value{0.};
}());
} // namespace
} // namespace fcarouge::test
