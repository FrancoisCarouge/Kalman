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

#ifndef FCAROUGE_KALMAN_FILTER_INTERNAL_TYPE_HPP
#define FCAROUGE_KALMAN_FILTER_INTERNAL_TYPE_HPP

#include "utility.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <type_traits>

namespace fcarouge::kalman_filter::internal {
// Owns the elements of a characteristic declared from a braced list of rows,
// each row a braced list of elements. The declaration is not held as a list of
// lists: a `std::initializer_list` refers to a backing array which does not
// outlive the full-expression of the declaration, so a declared characteristic
// stored in a variable, a data member, or returned from a function would
// dangle. The numbers of rows and columns are part of the type: rows of
// different sizes are rejected at compile-time.
template <typename Element, std::size_t Rows, std::size_t Columns>
struct row_list {
  std::array<std::array<Element, Columns>, Rows> elements;

  template <std::size_t... Sizes>
  constexpr explicit row_list(const Element (&...rows)[Sizes])
      : elements{{std::to_array(rows)...}} {}

  // Converts to a type constructible from a list of lists, the support
  // expected from a linear algebra backend matrix. One recursion per row, each
  // holding the list of that row, so that every list outlives the construction.
  template <typename Type, typename... Lists>
  [[nodiscard]] constexpr auto to(const Lists &...lists) const -> Type {
    if constexpr (sizeof...(Lists) < Rows) {
      return std::apply(
          [&](const auto &...row) -> Type {
            const std::initializer_list<Element> list{row...};

            return to<Type>(lists..., list);
          },
          elements[sizeof...(Lists)]);
    } else {
      const std::initializer_list<std::initializer_list<Element>> list{
          lists...};

      return Type(list);
    }
  }
};

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

  template <typename Element, std::size_t Columns, std::size_t... Sizes>
  constexpr explicit estimate_uncertainty(const Element (&row)[Columns],
                                          const Element (&...rows)[Sizes])
      : value(row, rows...) {}

  template <typename Element>
  constexpr explicit estimate_uncertainty(Element p) : value{p} {}
};

template <typename Element>
estimate_uncertainty(Element) -> estimate_uncertainty<Element>;

template <typename Element, std::size_t Columns, std::size_t... Sizes>
  requires((Sizes == Columns) && ...)
estimate_uncertainty(const Element (&row)[Columns],
                     const Element (&...rows)[Sizes])
    -> estimate_uncertainty<row_list<Element, 1 + sizeof...(Sizes), Columns>>;

template <typename Type> struct output_uncertainty {
  using type = Type;

  Type value;

  template <typename Element, std::size_t Columns, std::size_t... Sizes>
  constexpr explicit output_uncertainty(const Element (&row)[Columns],
                                        const Element (&...rows)[Sizes])
      : value(row, rows...) {}

  template <typename Element>
  constexpr explicit output_uncertainty(Element r) : value{r} {}
};

template <typename Element>
output_uncertainty(Element) -> output_uncertainty<Element>;

template <typename Element, std::size_t Columns, std::size_t... Sizes>
  requires((Sizes == Columns) && ...)
output_uncertainty(const Element (&row)[Columns],
                   const Element (&...rows)[Sizes])
    -> output_uncertainty<row_list<Element, 1 + sizeof...(Sizes), Columns>>;

template <typename Type> struct process_uncertainty {
  using type = Type;

  Type value;

  template <typename Element, std::size_t Columns, std::size_t... Sizes>
  constexpr explicit process_uncertainty(const Element (&row)[Columns],
                                         const Element (&...rows)[Sizes])
      : value(row, rows...) {}

  template <typename Element>
  constexpr explicit process_uncertainty(Element q) : value{q} {}
};

template <typename Element>
process_uncertainty(Element) -> process_uncertainty<Element>;

template <typename Element, std::size_t Columns, std::size_t... Sizes>
  requires((Sizes == Columns) && ...)
process_uncertainty(const Element (&row)[Columns],
                    const Element (&...rows)[Sizes])
    -> process_uncertainty<row_list<Element, 1 + sizeof...(Sizes), Columns>>;

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

  template <typename Element, std::size_t Columns, std::size_t... Sizes>
  constexpr explicit output_model(const Element (&row)[Columns],
                                  const Element (&...rows)[Sizes])
      : value(row, rows...) {}

  template <typename Element>
  constexpr explicit output_model(Element h) : value{h} {}
};

template <typename Element> output_model(Element) -> output_model<Element>;

template <typename Element, std::size_t Columns, std::size_t... Sizes>
  requires((Sizes == Columns) && ...)
output_model(const Element (&row)[Columns], const Element (&...rows)[Sizes])
    -> output_model<row_list<Element, 1 + sizeof...(Sizes), Columns>>;

template <typename Type> struct state_transition {
  using type = Type;

  Type value;

  template <typename Element, std::size_t Columns, std::size_t... Sizes>
  constexpr explicit state_transition(const Element (&row)[Columns],
                                      const Element (&...rows)[Sizes])
      : value(row, rows...) {}

  template <typename Element>
  constexpr explicit state_transition(Element f) : value{f} {}
};

template <typename Element>
state_transition(Element) -> state_transition<Element>;

template <typename Element, std::size_t Columns, std::size_t... Sizes>
  requires((Sizes == Columns) && ...)
state_transition(const Element (&row)[Columns], const Element (&...rows)[Sizes])
    -> state_transition<row_list<Element, 1 + sizeof...(Sizes), Columns>>;

template <typename Type> struct input_control {
  using type = Type;

  Type value;

  template <typename Element, std::size_t Columns, std::size_t... Sizes>
  constexpr explicit input_control(const Element (&row)[Columns],
                                   const Element (&...rows)[Sizes])
      : value(row, rows...) {}

  template <typename Element>
  constexpr explicit input_control(Element g) : value{g} {}
};

template <typename Element> input_control(Element) -> input_control<Element>;

template <typename Element, std::size_t Columns, std::size_t... Sizes>
  requires((Sizes == Columns) && ...)
input_control(const Element (&row)[Columns], const Element (&...rows)[Sizes])
    -> input_control<row_list<Element, 1 + sizeof...(Sizes), Columns>>;

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

template <typename... Types> struct update_types_t {};

template <typename... Types> inline update_types_t<Types...> update_types{};

template <typename... Types> struct prediction_types_t {};

template <typename... Types>
inline prediction_types_t<Types...> prediction_types{};
} // namespace fcarouge::kalman_filter::internal

#endif // FCAROUGE_KALMAN_FILTER_INTERNAL_TYPE_HPP
