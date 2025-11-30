/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.3
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

#ifndef FCAROUGE_KALMAN_INTERNAL_PRINT_HPP
#define FCAROUGE_KALMAN_INTERNAL_PRINT_HPP

#include "utility.hpp"

#include <print>
#include <utility>

namespace fcarouge {
namespace kalman_internal {
template <typename Filter> class printer : public Filter {
public:
  constexpr explicit printer(Filter &&filter);
  constexpr ~printer();
  constexpr decltype(auto) x(this auto &&self, const auto &...values)
    requires(kalman_internal::has_state<Filter>);
  constexpr decltype(auto) z(this auto &&self, const auto &...values)
    requires(kalman_internal::has_output<Filter>);
  constexpr decltype(auto) u(this auto &&self, const auto &...values)
    requires(kalman_internal::has_input<Filter>);
  constexpr decltype(auto) p(this auto &&self, const auto &...values)
    requires(kalman_internal::has_estimate_uncertainty<Filter>);
  constexpr decltype(auto) q(this auto &&self, const auto &...values)
    requires(kalman_internal::has_process_uncertainty<Filter>);
  constexpr decltype(auto) r(this auto &&self, const auto &...values)
    requires(kalman_internal::has_output_uncertainty<Filter>);
  constexpr decltype(auto) f(this auto &&self, const auto &...values)
    requires(kalman_internal::has_state_transition<Filter>);
  constexpr decltype(auto) h(this auto &&self, const auto &...values)
    requires(kalman_internal::has_output_model<Filter>);
  constexpr decltype(auto) g(this auto &&self, const auto &...values)
    requires(kalman_internal::has_input_control<Filter>);
  constexpr decltype(auto) k(this auto &&self, const auto &...values)
    requires(kalman_internal::has_gain<Filter>);
  constexpr decltype(auto) y(this auto &&self, const auto &...values)
    requires(kalman_internal::has_innovation<Filter>);
  constexpr decltype(auto) s(this auto &&self, const auto &...values)
    requires(kalman_internal::has_innovation_uncertainty<Filter>);
  constexpr void predict(const auto &...arguments);
  template <auto Position> constexpr auto predict() const;
  constexpr void update(const auto &...arguments);
  template <auto Position> constexpr auto update() const;
};

template <typename Filter>
constexpr printer<Filter>::printer(Filter &&filter)
    : Filter{std::forward<Filter>(filter)} {
  const Filter &base{*this};
  std::println(R"({{"event": "construction", "filter":{}}})", base);
}

template <typename Filter> constexpr printer<Filter>::~printer() {
  const Filter &base{*this};
  std::println(R"({{"event": "destruction", "filter":{}}})", base);
}

template <typename Filter>
constexpr decltype(auto) printer<Filter>::x(this auto &&self,
                                            const auto &...values)
  requires(kalman_internal::has_state<Filter>)
{
  kalman_internal::scope_exit on_exit{[&self] {
    const Filter &base{self};
    std::println(R"({{"event": "x", "filter":{}}})", base);
  }};

  return std::forward<decltype(self)>(self).Filter::x(values...);
}

template <typename Filter>
constexpr decltype(auto) printer<Filter>::z(this auto &&self,
                                            const auto &...values)
  requires(kalman_internal::has_output<Filter>)
{
  kalman_internal::scope_exit on_exit{[&self] {
    const Filter &base{self};
    std::println(R"({{"event": "z", "filter":{}}})", base);
  }};

  return std::forward<decltype(self)>(self).Filter::z(values...);
}

template <typename Filter>
constexpr decltype(auto) printer<Filter>::u(this auto &&self,
                                            const auto &...values)
  requires(kalman_internal::has_input<Filter>)
{
  kalman_internal::scope_exit on_exit{[&self] {
    const Filter &base{self};
    std::println(R"({{"event": "u", "filter":{}}})", base);
  }};

  return std::forward<decltype(self)>(self).Filter::u(values...);
}

template <typename Filter>
constexpr decltype(auto) printer<Filter>::p(this auto &&self,
                                            const auto &...values)
  requires(kalman_internal::has_estimate_uncertainty<Filter>)
{
  kalman_internal::scope_exit on_exit{[&self] {
    const Filter &base{self};
    std::println(R"({{"event": "p", "filter":{}}})", base);
  }};

  return std::forward<decltype(self)>(self).Filter::p(values...);
}

template <typename Filter>
constexpr decltype(auto) printer<Filter>::q(this auto &&self,
                                            const auto &...values)
  requires(kalman_internal::has_process_uncertainty<Filter>)
{
  kalman_internal::scope_exit on_exit{[&self] {
    const Filter &base{self};
    std::println(R"({{"event": "q", "filter":{}}})", base);
  }};

  return std::forward<decltype(self)>(self).Filter::q(values...);
}

template <typename Filter>
constexpr decltype(auto) printer<Filter>::r(this auto &&self,
                                            const auto &...values)
  requires(kalman_internal::has_output_uncertainty<Filter>)
{
  kalman_internal::scope_exit on_exit{[&self] {
    const Filter &base{self};
    std::println(R"({{"event": "r", "filter":{}}})", base);
  }};

  return std::forward<decltype(self)>(self).Filter::r(values...);
}

template <typename Filter>
constexpr decltype(auto) printer<Filter>::f(this auto &&self,
                                            const auto &...values)
  requires(kalman_internal::has_state_transition<Filter>)
{
  kalman_internal::scope_exit on_exit{[&self] {
    const Filter &base{self};
    std::println(R"({{"event": "f", "filter":{}}})", base);
  }};

  return std::forward<decltype(self)>(self).Filter::f(values...);
}

template <typename Filter>
constexpr decltype(auto) printer<Filter>::h(this auto &&self,
                                            const auto &...values)
  requires(kalman_internal::has_output_model<Filter>)
{
  kalman_internal::scope_exit on_exit{[&self] {
    const Filter &base{self};
    std::println(R"({{"event": "h", "filter":{}}})", base);
  }};

  return std::forward<decltype(self)>(self).Filter::h(values...);
}

template <typename Filter>
constexpr decltype(auto) printer<Filter>::g(this auto &&self,
                                            const auto &...values)
  requires(kalman_internal::has_input_control<Filter>)
{
  kalman_internal::scope_exit on_exit{[&self] {
    const Filter &base{self};
    std::println(R"({{"event": "g", "filter":{}}})", base);
  }};

  return std::forward<decltype(self)>(self).Filter::g(values...);
}

template <typename Filter>
constexpr decltype(auto) printer<Filter>::k(this auto &&self,
                                            const auto &...values)
  requires(kalman_internal::has_gain<Filter>)
{
  kalman_internal::scope_exit on_exit{[&self] {
    const Filter &base{self};
    std::println(R"({{"event": "k", "filter":{}}})", base);
  }};

  return std::forward<decltype(self)>(self).Filter::k(values...);
}

template <typename Filter>
constexpr decltype(auto) printer<Filter>::y(this auto &&self,
                                            const auto &...values)
  requires(kalman_internal::has_innovation<Filter>)
{
  kalman_internal::scope_exit on_exit{[&self] {
    const Filter &base{self};
    std::println(R"({{"event": "y", "filter":{}}})", base);
  }};

  return std::forward<decltype(self)>(self).Filter::y(values...);
}

template <typename Filter>
constexpr decltype(auto) printer<Filter>::s(this auto &&self,
                                            const auto &...values)
  requires(kalman_internal::has_innovation_uncertainty<Filter>)
{
  kalman_internal::scope_exit on_exit{[&self] {
    const Filter &base{self};
    std::println(R"({{"event": "s", "filter":{}}})", base);
  }};

  return std::forward<decltype(self)>(self).Filter::s(values...);
}

template <typename Filter>
constexpr void printer<Filter>::predict(const auto &...arguments) {
  Filter::predict(arguments...);

  const Filter &base{*this};
  std::println(R"({{"event": "predict", "filter":{}}})", base);
}

template <typename Filter>
template <auto Position>
[[nodiscard("The returned prediction argument is unexpectedly "
            "discarded.")]] constexpr auto
printer<Filter>::predict() const {
  const Filter &base{*this};
  std::println(R"({{"event": "predict_{}", "filter":{}}})", Position, base);

  return Filter::template predict<Position>();
}

template <typename Filter>
constexpr void printer<Filter>::update(const auto &...arguments) {
  Filter::update(arguments...);

  const Filter &base{*this};
  std::println(R"({{"event": "update", "filter":{}}})", base);
}
} // namespace kalman_internal

struct printer {};

template <typename Filter>
constexpr auto operator|(Filter &&filter,
                         [[maybe_unused]] const printer &decorator) {
  return kalman_internal::printer<Filter>(std::forward<Filter>(filter));
}
} // namespace fcarouge

#endif // FCAROUGE_KALMAN_INTERNAL_PRINT_HPP
