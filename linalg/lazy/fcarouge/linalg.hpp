/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.3.0
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

#ifndef FCAROUGE_LINALG_HPP
#define FCAROUGE_LINALG_HPP

//! @file
//! @brief Linear algebra coroutine-based lazy implementation.
//!
//! @details Matrix, vectors, and named algebraic values.
//!
//! @note Idea from:
//! https://gieseanw.wordpress.com/2019/10/20/we-dont-need-no-stinking-expression-templates/

#include "fcarouge/utility.hpp"

// #include <range/v3/range.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <coroutine>
#include <format>
#include <generator>
#include <ranges>
#include <type_traits>
#include <utility>

namespace fcarouge {
// Semantic? Guarantees? to_generator? std::ranges::to overload? make_generator?
// Overloads for C-array?
// Need one copy, avoid any extra?
template <std::ranges::range Range>
inline constexpr std::generator<typename Range::value_type>
make_generator(Range elements) {
  return [](auto elements_copy)
             -> std::generator<typename decltype(elements)::value_type> {
    for (auto &&element : elements_copy) { // std::ranges::elements_of
      co_yield element;
    }
  }(elements);
}

// Need one copy, avoid any extra?
template <typename Type>
inline constexpr auto make_generator(Type element) -> std::generator<Type> {
  return [](Type element_copy) -> std::generator<Type> {
    co_yield element_copy;
  }(element);
}

//! @name Algebraic Types
//! @{
//! @brief Coroutine lazy matrix.
//!
//! @details The matrix is a generator. A coroutine range. Lazily generated
//! elements and computed operations. Commonalities with ranges.
//!
//! @tparam Type The matrix element type.
//! @tparam Row The number of rows of the matrix.
//! @tparam Column The number of columns of the matrix.
//! @tparam Copyable Whether the instance is fully lazy or deeply clones itself
//! on copies. Useful for named values. May not be commonly used.
//!
//! @note Lifetime management with coroutine is tricky. The generators cannot
//! use the lambda captures. The parameters of the lambda can however be copied
//! to guarentee their lifetime for the usage of the generator. This technique
//! is useful for initialization of the generator.
//! @note A design decision for the composed generator to be mutable, traded
//! off for const member function API. Similar to the mutable mutex member
//! practice.
//! @note Why genie? Because genies generate on demand...
//!
//! @todo Explore and compare performance.
//! @todo Explore optimization of heap allocations?
//! @todo Explore constexpr support?
//! @todo Explore an implementation where each element is a generator?
//! @todo Explore cyclic generator to keep moving forward and not track results?
//! @todo Explore verification of lazy evaluation?
//! @todo Remove unecessary empty paramaters when MSVC supports lambda without
//! them.
template <typename Type = double, auto Row = 1, auto Column = 1,
          bool Copyable = false>
struct matrix {
  //! @todo Report the ICE for generator construction in member declaration. Use
  //! constructor's initializer list for now.
  inline constexpr matrix()
      : genie{[]() -> std::generator<Type> {
          for (auto k{Row * Column}; k > 0; --k) { // repeat(Type{}) | take(R*C)
            co_yield {};
          }
        }()} {}

  inline constexpr matrix(const matrix<Type, Row, Column, false> &other)
      : genie{std::move(other.genie)} {}

  inline constexpr matrix(const matrix<Type, Row, Column, true> &other)
      : genie{other.clone()} {}

  inline constexpr matrix &
  operator=(const matrix<Type, Row, Column, false> &other) {
    genie = std::move(other.genie);
    return *this;
  }

  inline constexpr matrix &
  operator=(const matrix<Type, Row, Column, true> &other) {
    genie = other.clone();
    return *this;
  }

  inline constexpr matrix(matrix<Type, Row, Column, false> &&other)
      : genie{std::move(other.genie)} {}

  inline constexpr matrix(matrix<Type, Row, Column, true> &&other)
      : genie{other.clone()} {}

  inline constexpr matrix &operator=(matrix<Type, Row, Column, false> &&other) {
    genie = std::move(other.genie);
    return *this;
  }

  inline constexpr matrix &operator=(matrix<Type, Row, Column, true> &&other) {
    genie = other.clone();
    return *this;
  }

  inline constexpr explicit(false)
      matrix(const std::same_as<Type> auto &...elements)
    requires(sizeof...(elements) == Row * Column)
      : genie{[](auto... elements_copy) -> std::generator<Type> {
          (co_yield elements_copy, ...);
        }(elements...)} {}

  inline constexpr explicit matrix(Type (&elements)[Row * Column])
    requires(Row == 1 || Column == 1)
      : genie{make_generator(std::to_array(elements))} {}

  template <typename... Types, auto... Columns>
  matrix(const Types (&...rows)[Columns])
    requires(std::conjunction_v<std::is_same<Type, Types>...> &&
             ((Columns == Column) && ... && true))
      : genie{[](auto rows_copy) -> std::generator<Type> {
          for (auto &&row : rows_copy) {
            for (auto &&element : row) { // std::ranges::elements_of
              co_yield element;
            }
          }
        }(std::to_array({std::to_array(rows)...}))} {}

  inline constexpr matrix(std::generator<Type> other)
      : genie{std::move(other)} {}

  inline constexpr matrix(std::invocable auto other) : genie{other()} {}

  [[nodiscard]] inline constexpr std::generator<Type> clone() const {
    std::array<Type, Row * Column> elements; // std::ranges::to
    std::ranges::copy(genie, elements.begin());

    genie = make_generator(elements);

    return make_generator(elements);
  }

  [[nodiscard]] inline constexpr explicit(false)
  operator matrix<Type, Row, Column, !Copyable>() const {
    return clone();
  }

  [[nodiscard]] inline constexpr explicit(false) operator Type() const
    requires(Row == 1 && Column == 1)
  {
    Type element{*genie.begin()};

    genie = make_generator(element);

    return element;
  }

  [[nodiscard]] inline explicit(false) operator std::generator<Type>() const {
    co_yield std::ranges::elements_of(genie);
  }

  [[nodiscard]] inline constexpr Type operator[](auto index) const
    requires(Row != 1 && Column == 1)
  {
    std::array<Type, Row * Column> elements; // std::ranges::to

    std::ranges::copy(genie, elements.begin());
    genie = make_generator(elements);

    return elements[index];
  }

  [[nodiscard]] inline constexpr Type operator[](auto index) const
    requires(Row == 1)
  {
    std::array<Type, Row * Column> elements; // std::ranges::to

    std::ranges::copy(genie, elements.begin());
    genie = make_generator(elements);

    return elements[index];
  }

  [[nodiscard]] inline constexpr Type operator()(auto index) const
    requires(Row != 1 && Column == 1)
  {
    std::array<Type, Row * Column> elements; // std::ranges::to

    std::ranges::copy(genie, elements.begin());
    genie = make_generator(elements);

    return elements[index];
  }

  [[nodiscard]] inline constexpr Type operator()(auto index) const
    requires(Row == 1)
  {
    std::array<Type, Row * Column> elements; // std::ranges::to

    std::ranges::copy(genie, elements.begin());
    genie = make_generator(elements);

    return elements[index];
  }

  //! @todo Don't evaluate unless needed. How to do? Some kind of recursive
  //! genie recomposer?
  [[nodiscard]] inline constexpr Type operator()(auto row, auto column) const {
    std::array<Type, Row * Column> elements; // std::ranges::to

    std::ranges::copy(genie, elements.begin());
    genie = make_generator(elements);

    return elements[row * Column + column];
  }

  using generator = std::generator<Type>;
  using promise_type = std::coroutine_traits<generator>::promise_type;
  // Add other aliases such as iterator or value type?

  auto begin() const { return genie.begin(); }
  auto end() const { return genie.end(); }

  mutable generator genie;
};

//! @brief Coroutine lazy row vector.
template <typename Type = double, auto Column = 1, bool Copyable = false>
using row_vector = matrix<Type, decltype(Column){1}, Column, Copyable>;

//! @brief Coroutine lazy column vector.
template <typename Type = double, auto Row = 1, bool Copyable = false>
using column_vector = matrix<Type, Row, decltype(Row){1}, Copyable>;

//! @}

//! @name Deduction Guides
//! @{
template <typename Type> matrix(Type) -> matrix<Type, 1, 1>;

template <typename Type, auto Row, auto Column>
matrix(const Type (&)[Row][Column]) -> matrix<Type, Row, Column>;

template <typename Type, auto Row>
matrix(const Type (&)[Row]) -> matrix<Type, Row, 1>;

template <typename... Types, auto... Columns>
  requires(std::conjunction_v<std::is_same<first_t<Types...>, Types>...> &&
           ((Columns == first_v<Columns>) && ... && true))
matrix(const Types (&...rows)[Columns])
    -> matrix<std::remove_cvref_t<first_t<Types...>>, sizeof...(Columns),
              (Columns, ...)>;
//! @}

template <typename Type, auto Row, auto Column>
auto make_identity_generator{[]() -> std::generator<Type> {
  for (decltype(Row) i{0}; i < Row; ++i) {
    for (decltype(Column) j{0}; j < Column; ++j) {
      co_yield i == j;
    }
  }
}};

template <typename Type, auto Row, auto Column>
auto make_zero_generator{[]() -> std::generator<Type> {
  for (auto k{Row * Column}; k > 0; --k) {
    co_yield 0.0;
  }
}};

//! @name Algebraic Named Values
//! @{
//! @brief The identity matrix lazy specialization.
template <typename Type, auto Row, auto Column, bool CopyableOrNot>
auto identity_v<matrix<Type, Row, Column, CopyableOrNot>>{[](auto... args) {
  matrix<Type, Row, Column, true> m{
      make_identity_generator<Type, Row, Column>()};
  if constexpr (sizeof...(args)) {
    return m(args...);
  } else {
    return m;
  }
}()};

//! @brief The zero matrix lazy specialization.
template <typename Type, auto Row, auto Column, bool CopyableOrNot>
auto zero_v<matrix<Type, Row, Column, CopyableOrNot>>{[](auto... args) {
  matrix<Type, Row, Column, true> m{make_zero_generator<Type, Row, Column>()};
  if constexpr (sizeof...(args)) {
    return m(args...);
  } else {
    return m;
  }
}()};

//! @}

template <std::invocable L> bool operator==(L lhs, L rhs) {
  return lhs() == rhs();
}

template <std::invocable L, typename T> bool operator==(L lhs, T rhs) {
  return lhs() == rhs;
}

template <typename Type, auto Row, auto Column, bool CopyableOrNot1,
          bool CopyableOrNot2>
[[nodiscard]] inline constexpr bool
operator==(matrix<Type, Row, Column, CopyableOrNot1> lhs,
           matrix<Type, Row, Column, CopyableOrNot2> rhs) {
  std::array<Type, Row * Column> lhs_elements; // std::ranges::to
  std::ranges::copy(lhs.genie, lhs_elements.begin());
  lhs.genie = make_generator(lhs_elements);

  std::array<Type, Row * Column> rhs_elements; // std::ranges::to
  std::ranges::copy(rhs.genie, rhs_elements.begin());
  rhs.genie = make_generator(rhs_elements);

  return lhs_elements == rhs_elements;
}

template <typename Type, auto Row, auto Column>
[[nodiscard]] inline matrix<Type, Row, Column>
operator*(matrix<Type, Row, Column> lhs, arithmetic auto rhs) {
  auto next{lhs.begin()};
  for (auto k{Row * Column}; k > 0; --k, ++next) {
    co_yield *next *rhs;
  }
}

template <typename Type, auto Row, auto Size, bool CopyableOrNot1,
          bool CopyableOrNot2>
[[nodiscard]] inline matrix<Type, Row, 1>
operator*(matrix<Type, Row, Size, CopyableOrNot1> lhs,
          matrix<Type, Size, 1, CopyableOrNot2> rhs) {
  // fix me?
  auto next1{lhs.begin()};
  for (decltype(Row) i{0}; i < Row; ++i) {       // chunk_by_rows
    matrix<Type, Size, 1> rhs_copy{rhs.clone()}; // repeat_n
    auto next2{rhs_copy.begin()};
    Type element{}; // inner_product?
    for (decltype(Size) k{0}; k < Size; ++k, ++next1, ++next2) {
      element += *next1 * *next2;
    }
    co_yield element;
  }
}

// //! @todo Implement me.
// template <typename Type, auto Row, auto Size, auto Column>
// [[nodiscard]] inline matrix<Type, Row, Column>
// operator*(matrix<Type, Row, Size> lhs, matrix<Type, Size, Column> rhs) {
//   static_cast<void>(lhs);
//   static_cast<void>(rhs);
//   for (auto k{Row * Column}; k > 0; --k) {
//     co_yield Type{};
//   }
// }

template <typename Type>
[[nodiscard]] inline matrix<Type, 1, 1> operator+(Type lhs,
                                                  matrix<Type, 1, 1> rhs) {
  co_yield lhs + *rhs.begin();
}

template <typename Type, auto Row, auto Column>
[[nodiscard]] inline matrix<Type, Row, Column>
operator+(matrix<Type, Row, Column> lhs, matrix<Type, Row, Column> rhs) {
  auto next1{lhs.begin()};
  auto next2{rhs.begin()};
  for (auto k{Row * Column}; k > 0; --k, ++next1, ++next2) {
    co_yield *next1 + *next2;
  }
}

// //! @todo Implement me.
// template <typename Type, auto Row, auto Column>
// [[nodiscard]] inline matrix<Type, Column, Row>
// transpose(matrix<Type, Row, Column> other) {
//   static_cast<void>(other);
//   for (auto k{Row * Column}; k > 0; --k) {
//     co_yield Type{};
//   }
// }
} // namespace fcarouge

// template <typename Type, auto Row, auto Column, typename Char>
// struct std::formatter<fcarouge::matrix<Type, Row, Column>, Char>
//     : public std::formatter<Type, Char> {
//   constexpr auto parse(std::basic_format_parse_context<Char> &parse_context)
//   {
//     return parse_context.begin();
//   }

//   //! @todo P2585 may be useful in simplifying and standardizing the support.
//   template <typename OutputIt>
//   auto format(const fcarouge::matrix<Type, Row, Column> &other,
//               std::basic_format_context<OutputIt, Char> &format_context)
//       -> OutputIt {

//     return format_context.out();
//   }
// };

#endif // FCAROUGE_LINALG_HPP
