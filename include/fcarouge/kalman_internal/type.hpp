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

#ifndef FCAROUGE_KALMAN_INTERNAL_TYPE_HPP
#define FCAROUGE_KALMAN_INTERNAL_TYPE_HPP

#include "utility.hpp"

#include <initializer_list>
#include <type_traits>

namespace fcarouge::kalman_internal {
//! @todo Provide the ::type member access and _t shorthand for simplifying
//! syntax?
template <typename Type> struct state {
  using type = Type;

  Type value;

  constexpr explicit state(Type v) : value{v} {}

  template <typename... Types>
    requires(sizeof...(Types) > 1)
  constexpr state(Types... elements) : value{elements...} {}
};

template <typename Type> state(Type) -> state<Type>;

template <typename... Types>
state(Types... elements)
    -> state<std::remove_cvref_t<first<Types...>>[sizeof...(Types)]>;

template <typename Type> struct estimate_uncertainty {
  using type = Type;

  Type value;

  template <typename Element>
  constexpr explicit estimate_uncertainty(
      std::initializer_list<std::initializer_list<Element>> p)
      : value(p) {}

  template <typename Element>
  constexpr explicit estimate_uncertainty(Element p) : value{p} {}
};

template <typename Element>
estimate_uncertainty(Element) -> estimate_uncertainty<Element>;

template <typename Element>
estimate_uncertainty(std::initializer_list<std::initializer_list<Element>>)
    -> estimate_uncertainty<
        std::initializer_list<std::initializer_list<Element>>>;

template <typename Type> struct output_uncertainty {
  using type = Type;

  Type value;

  template <typename Element>
  constexpr explicit output_uncertainty(
      std::initializer_list<std::initializer_list<Element>> r)
      : value(r) {}

  template <typename Element>
  constexpr explicit output_uncertainty(Element r) : value{r} {}
};

template <typename Element>
output_uncertainty(Element) -> output_uncertainty<Element>;

template <typename Element>
output_uncertainty(std::initializer_list<std::initializer_list<Element>>)
    -> output_uncertainty<
        std::initializer_list<std::initializer_list<Element>>>;

template <typename Type> struct process_uncertainty {
  using type = Type;

  Type value;

  template <typename Element>
  constexpr explicit process_uncertainty(
      std::initializer_list<std::initializer_list<Element>> q)
      : value(q) {}

  template <typename Element>
  constexpr explicit process_uncertainty(Element q) : value{q} {}
};

template <typename Element>
process_uncertainty(Element) -> process_uncertainty<Element>;

template <typename Element>
process_uncertainty(std::initializer_list<std::initializer_list<Element>>)
    -> process_uncertainty<
        std::initializer_list<std::initializer_list<Element>>>;

template <typename Type> struct input_t {
  using type = Type;
};

template <typename Type> inline input_t<Type> input{};

template <typename Type> struct output_t {
  using type = Type;
};

template <typename Type> inline output_t<Type> output{};

template <typename Type> struct output_model {
  using type = Type;

  Type value;

  template <typename Element>
  constexpr explicit output_model(
      std::initializer_list<std::initializer_list<Element>> h)
      : value(h) {}

  template <typename Element>
  constexpr explicit output_model(Element h) : value{h} {}
};

template <typename Element> output_model(Element) -> output_model<Element>;

template <typename Element>
output_model(std::initializer_list<std::initializer_list<Element>>)
    -> output_model<std::initializer_list<std::initializer_list<Element>>>;

template <typename Type> struct state_transition {
  using type = Type;

  Type value;

  template <typename Element>
  constexpr explicit state_transition(
      std::initializer_list<std::initializer_list<Element>> f)
      : value(f) {}

  template <typename Element>
  constexpr explicit state_transition(Element f) : value{f} {}
};

template <typename Element>
state_transition(Element) -> state_transition<Element>;

template <typename Element>
state_transition(std::initializer_list<std::initializer_list<Element>>)
    -> state_transition<std::initializer_list<std::initializer_list<Element>>>;

template <typename Type> struct input_control {
  using type = Type;

  Type value;

  template <typename Element>
  constexpr explicit input_control(
      std::initializer_list<std::initializer_list<Element>> g)
      : value(g) {}

  template <typename Element>
  constexpr explicit input_control(Element g) : value{g} {}
};

template <typename Element> input_control(Element) -> input_control<Element>;

template <typename Element>
input_control(std::initializer_list<std::initializer_list<Element>>)
    -> input_control<std::initializer_list<std::initializer_list<Element>>>;

//! @todo Simplify?
template <typename Type> struct transition {
  using type = Type;

  Type value;

  template <typename Element>
  constexpr explicit transition(Element ff) : value{ff} {}
};

template <typename Element> transition(Element) -> transition<Element>;

template <typename Type> struct observation {
  using type = Type;

  Type value;

  template <typename Element>
  constexpr explicit observation(Element hh) : value{hh} {}
};

template <typename Element> observation(Element) -> observation<Element>;

//! @todo Better name not ending by *_types?
template <typename... Types> struct update_types_t {};

template <typename... Types> inline update_types_t<Types...> update_types{};

template <typename... Types> struct prediction_types_t {};

template <typename... Types>
inline prediction_types_t<Types...> prediction_types{};
} // namespace fcarouge::kalman_internal

#endif // FCAROUGE_KALMAN_INTERNAL_TYPE_HPP
