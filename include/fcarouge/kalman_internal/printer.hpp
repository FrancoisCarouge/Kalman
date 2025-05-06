/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.1
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

#ifndef FCAROUGE_KALMAN_INTERNAL_PRINTER_HPP
#define FCAROUGE_KALMAN_INTERNAL_PRINTER_HPP

//! @file
//! @brief Filter decoration support for printing operations.
//!
//! @todo Make this internal utility public and fix the namespace and name.

#include <print>

namespace fcarouge {
namespace decorator {
//! @brief Decorates a filter with operation printing.
//!
//! @todo Explain, review the inheritance.
//! @todo Review, change the namespace in the context of multiple projects
//! sharing the top-level namespace.
template <typename Filter> class printer : public Filter {
public:
  using Filter::p;
  using Filter::x;

  inline constexpr explicit printer([[maybe_unused]] Filter &&filter)
      : Filter{std::forward<Filter>(filter)} {
    std::println("{{\"event\": \"construction\", \"filter\":{}}}", *this);
  }

  inline constexpr ~printer() {
    std::println("{{\"event\": \"destruction\", \"filter\":{}}}", *this);
  }

  inline constexpr void x(const auto &value, const auto &...values) {
    Filter::x(value, values...);
    std::println("{{\"event\": \"x\", \"filter\":{}}}", *this);
  }

  inline constexpr void p(const auto &value, const auto &...values) {
    Filter::p(value, values...);
    std::println("{{\"event\": \"p\", \"filter\":{}}}", *this);
  }

  inline constexpr void predict(const auto &...arguments) {
    Filter::predict(arguments...);
    std::println("{{\"event\": \"predict\", \"filter\":{}}}", *this);
  }

  inline constexpr void update(const auto &...arguments) {
    Filter::update(arguments...);
    std::println("{{\"event\": \"update\", \"filter\":{}}}", *this);
  }
};
} // namespace decorator

struct printer_decorator {};

inline constexpr printer_decorator printer;

template <typename Filter>
inline constexpr auto
operator|(Filter &&filter, [[maybe_unused]] const printer_decorator decorator) {
  return decorator::printer<Filter>(std::forward<Filter>(filter));
}
} // namespace fcarouge

#endif // FCAROUGE_KALMAN_INTERNAL_PRINTER_HPP
