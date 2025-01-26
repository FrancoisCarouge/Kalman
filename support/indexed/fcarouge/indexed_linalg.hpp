/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.4.0
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

#ifndef FCAROUGE_INDEXED_LINALG_HPP
#define FCAROUGE_INDEXED_LINALG_HPP

//! @file
//! @brief Index typed linear algebra implementation.
//!
//! @details Matrix, vectors, and named algebraic values.

#include "fcarouge/utility.hpp"

#include <concepts>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <tuple>
#include <utility>

#include <print>

namespace fcarouge::indexed {

//! @name Types
//! @{

//! @brief Indexed matrix.
//!
//! @details Compose a linear algebra backend matrix into an indexed matrix. Row
//! and column indexes provide each element's index type.
//!
//! @tparam Matrix The underlying linear algebra matrix.
//! @tparam RowIndexes The packed types of the row indexes.
//! @tparam ColumnIndexes The packed types of the column indexes.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
class matrix {
public:
  inline constexpr matrix() = default;

  inline constexpr matrix(const matrix &other) = default;

  inline constexpr matrix &operator=(const matrix &other) = default;

  inline constexpr matrix(matrix &&other) = default;

  inline constexpr matrix &operator=(matrix &&other) = default;

  template <typename OtherMatrix>
  inline constexpr matrix(
      const matrix<OtherMatrix, RowIndexes, ColumnIndexes> &other)
      : data{other.data} {}

  //! @todo Constrain me more?
  inline constexpr matrix(const auto &other) : data{other} {}

  template <typename Type>
  inline constexpr explicit matrix(
      std::initializer_list<std::initializer_list<Type>> rows) {
    for (std::size_t i{0}; const auto &row : rows) {
      for (std::size_t j{0}; const auto &element : row) {
        data(i, j) = element;
        ++j;
      }
      ++i;
    }
  }

  template <typename... Types>
    requires(size<ColumnIndexes> == 1 && size<RowIndexes> != 1 &&
             sizeof...(Types) == size<RowIndexes>)
  explicit inline constexpr matrix(const Types &...elements) {
    std::tuple element_pack{elements...};
    for_constexpr<0, size<RowIndexes>, 1>([this, &element_pack](auto position) {
      data[position] = std::get<position>(element_pack);
    });
  }

  template <typename... Types>
    requires(size<RowIndexes> == 1 && size<ColumnIndexes> != 1 &&
             sizeof...(Types) == size<ColumnIndexes>)
  explicit inline constexpr matrix(const Types &...elements) {
    std::tuple element_pack{elements...};
    for_constexpr<0, size<ColumnIndexes>, 1>(
        [this, &element_pack](auto position) {
          data[position] = std::get<position>(element_pack);
        });
  }

  //! @todo Shorten with self deduced this?
  [[nodiscard]] inline constexpr auto &operator[](std::size_t index)
    requires(size<RowIndexes> != 1 && size<ColumnIndexes> == 1)
  {
    return data(index, 0);
  }

  [[nodiscard]] inline constexpr const auto &operator[](std::size_t index) const
    requires(size<RowIndexes> != 1 && size<ColumnIndexes> == 1)
  {
    return data(index, 0);
  }

  [[nodiscard]] inline constexpr auto &operator[](std::size_t index)
    requires(size<RowIndexes> == 1)
  {
    return data(0, index);
  }

  [[nodiscard]] inline constexpr const auto &operator[](std::size_t index) const
    requires(size<RowIndexes> == 1)
  {
    return data(0, index);
  }

  [[nodiscard]] inline constexpr auto &operator[](std::size_t row,
                                                  std::size_t column) {
    return data(row, column);
  }

  [[nodiscard]] inline constexpr const auto &
  operator[](std::size_t row, std::size_t column) const {
    return data(row, column);
  }

  [[nodiscard]] inline constexpr auto &operator()(std::size_t index)
    requires(size<RowIndexes> != 1 && size<ColumnIndexes> == 1)
  {
    return data(index, 0);
  }

  [[nodiscard]] inline constexpr const auto &operator()(std::size_t index) const
    requires(size<RowIndexes> != 1 && size<ColumnIndexes> == 1)
  {
    return data(index, 0);
  }

  [[nodiscard]] inline constexpr auto &operator()(std::size_t index)
    requires(size<RowIndexes> == 1)
  {
    return data(0, index);
  }

  [[nodiscard]] inline constexpr const auto &operator()(std::size_t index) const
    requires(size<RowIndexes> == 1)
  {
    return data(0, index);
  }

  [[nodiscard]] inline constexpr auto &operator()(std::size_t row,
                                                  std::size_t column) {
    return data(row, column);
  }

  [[nodiscard]] inline constexpr const auto &
  operator()(std::size_t row, std::size_t column) const {
    return data(row, column);
  }

  //! @todo Privatize me.
  Matrix data;
};

using one_row = std::tuple<int>;
using one_column = std::tuple<int>;

//! @brief Row vector.
template <typename Matrix, typename... ColumnIndexes>
using indexed_row_vector =
    matrix<Matrix, one_row, std::tuple<ColumnIndexes...>>;

//! @brief Column vector.
template <typename Matrix, typename... RowIndexes>
using indexed_column_vector =
    matrix<Matrix, std::tuple<RowIndexes...>, one_column>;

//! @}

//! @name Deduction Guides
//! @{

//! @}

template <typename Matrix1, typename Matrix2, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr bool
operator==(const matrix<Matrix1, RowIndexes, ColumnIndexes> &lhs,
           const matrix<Matrix2, RowIndexes, ColumnIndexes> &rhs) {
  return lhs.data == rhs.data;
}

template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator+(const matrix<Matrix, RowIndexes, ColumnIndexes> &lhs,
          const matrix<Matrix, RowIndexes, ColumnIndexes> &rhs) {
  return matrix<evaluate<Matrix>, RowIndexes, ColumnIndexes>{lhs.data +
                                                             rhs.data};
}

template <typename Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator+(const matrix<Matrix, RowIndexes, ColumnIndexes> &lhs, Scalar rhs) {
  return matrix<evaluate<Matrix>, RowIndexes, ColumnIndexes>{lhs.data + rhs};
}

template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator-(const matrix<Matrix, RowIndexes, ColumnIndexes> &lhs,
          const matrix<Matrix, RowIndexes, ColumnIndexes> &rhs) {
  return matrix<evaluate<Matrix>, RowIndexes, ColumnIndexes>{lhs.data -
                                                             rhs.data};
}

template <typename Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator-(Scalar lhs, const matrix<Matrix, RowIndexes, ColumnIndexes> &rhs) {
  //! @todo Return scalar?
  return matrix<evaluate<Matrix>, RowIndexes, ColumnIndexes>{lhs - rhs.data};
}

template <typename Matrix1, typename Matrix2, typename RowIndexes,
          typename Indexes, typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator*(const matrix<Matrix1, RowIndexes, Indexes> &lhs,
          const matrix<Matrix2, Indexes, ColumnIndexes> &rhs) {
  return matrix<evaluate<product<Matrix1, Matrix2>>, RowIndexes, ColumnIndexes>{
      lhs.data * rhs.data};
}

template <typename Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator*(Scalar lhs, const matrix<Matrix, RowIndexes, ColumnIndexes> &rhs) {
  return matrix<evaluate<Matrix>, RowIndexes, ColumnIndexes>{lhs * rhs.data};
}

template <typename Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator*(const matrix<Matrix, RowIndexes, ColumnIndexes> &lhs, Scalar rhs) {
  return matrix<evaluate<Matrix>, RowIndexes, ColumnIndexes>{lhs.data * rhs};
}

template <typename Matrix1, typename Matrix2, typename RowIndexes1,
          typename RowIndexes2, typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator/(const matrix<Matrix1, RowIndexes1, ColumnIndexes> &lhs,
          const matrix<Matrix2, RowIndexes2, ColumnIndexes> &rhs) {
  return matrix<evaluate<divide<Matrix1, Matrix2>>, RowIndexes1, RowIndexes2>{
      lhs.data / rhs.data};
}

template <fcarouge::arithmetic Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator/(const matrix<Matrix, RowIndexes, ColumnIndexes> &lhs, Scalar rhs) {
  return matrix<evaluate<Matrix>, RowIndexes, ColumnIndexes>{lhs.data / rhs};
}
} // namespace fcarouge::indexed

//! @brief Specialization of the standard formatter for the indexed linear
//! algebra matrix.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes,
          typename Char>
struct std::formatter<
    fcarouge::indexed::matrix<Matrix, RowIndexes, ColumnIndexes>, Char> {
  constexpr auto parse(std::basic_format_parse_context<Char> &parse_context) {
    return parse_context.begin();
  }

  template <typename OutputIterator>
  constexpr auto format(
      const fcarouge::indexed::matrix<Matrix, RowIndexes, ColumnIndexes> &value,
      std::basic_format_context<OutputIterator, Char> &format_context) const
      -> OutputIterator {
    format_context.advance_to(std::format_to(format_context.out(), "["));

    fcarouge::for_constexpr<0, fcarouge::size<RowIndexes>, 1>([&format_context,
                                                               &value](auto i) {
      if (i > 0) {
        format_context.advance_to(std::format_to(format_context.out(), ", "));
      }

      format_context.advance_to(std::format_to(format_context.out(), "["));

      fcarouge::for_constexpr<0, fcarouge::size<ColumnIndexes>, 1>(
          [&format_context, &value, i](auto j) {
            if (j > 0) {
              format_context.advance_to(
                  std::format_to(format_context.out(), ", "));
            }

            format_context.advance_to(
                std::format_to(format_context.out(), "{}",
                               value.data(std::size_t{i}, std::size_t{j})));
          });

      format_context.advance_to(std::format_to(format_context.out(), "]"));
    });

    format_context.advance_to(std::format_to(format_context.out(), "]"));

    return format_context.out();
  }

  template <typename OutputIterator>
  constexpr auto format(
      const fcarouge::indexed::matrix<Matrix, RowIndexes, ColumnIndexes> &value,
      std::basic_format_context<OutputIterator, Char> &format_context) const
      -> OutputIterator
    requires(fcarouge::size<RowIndexes> == 1 &&
             fcarouge::size<ColumnIndexes> != 1)
  {
    format_context.advance_to(std::format_to(format_context.out(), "["));

    fcarouge::for_constexpr<0, fcarouge::size<ColumnIndexes>, 1>(
        [&format_context, &value](auto j) {
          if (j > 0) {
            format_context.advance_to(
                std::format_to(format_context.out(), ", "));
          }

          format_context.advance_to(std::format_to(format_context.out(), "{}",
                                                   value.data(std::size_t{j})));
        });

    format_context.advance_to(std::format_to(format_context.out(), "]"));

    return format_context.out();
  }

  template <typename OutputIterator>
  constexpr auto format(
      const fcarouge::indexed::matrix<Matrix, RowIndexes, ColumnIndexes> &value,
      std::basic_format_context<OutputIterator, Char> &format_context) const
      -> OutputIterator
    requires(fcarouge::size<RowIndexes> == 1 &&
             fcarouge::size<ColumnIndexes> == 1)
  {
    format_context.advance_to(
        std::format_to(format_context.out(), "{}", value.data(0)));

    return format_context.out();
  }
};

namespace fcarouge {
//! @brief Specialization of the evaluation type.
//!
//! @note Implementation not needed.
template <template <typename, typename, typename> typename IndexedMatrix,
          typename Matrix, typename RowIndexes, typename ColumnIndexes>
struct evaluater<IndexedMatrix<Matrix, RowIndexes, ColumnIndexes>> {
  [[nodiscard]] inline constexpr auto operator()() const
      -> IndexedMatrix<evaluate<Matrix>, RowIndexes, ColumnIndexes>;
};

//! @brief Specialization of the transposer.
template <template <typename, typename, typename> typename IndexedMatrix,
          typename Matrix, typename RowIndexes, typename ColumnIndexes>
  requires requires(IndexedMatrix<Matrix, RowIndexes, ColumnIndexes> m) {
    m.data;
  }
struct transposer<IndexedMatrix<Matrix, RowIndexes, ColumnIndexes>> {
  [[nodiscard]] inline constexpr auto operator()(
      const IndexedMatrix<Matrix, RowIndexes, ColumnIndexes> &value) const {
    return IndexedMatrix<evaluate<transpose<Matrix>>, ColumnIndexes,
                         RowIndexes>{t(value.data)};
  }
};

//! @}

//! @name Algebraic Named Values
//! @{

//! @brief The identity matrix indexed specialization.
//!
//! @todo The identity doesn't really make sense for matrices without units?
//! Unless it's the matrix without units? Even then, the identity matrix is
//! supposed to be square? What's the name of this thing then?
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
inline indexed::matrix<decltype(identity<Matrix>), RowIndexes, ColumnIndexes>
    identity<indexed::matrix<Matrix, RowIndexes, ColumnIndexes>>{
        identity<Matrix>};

//! @brief The zero matrix indexed specialization.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
inline indexed::matrix<decltype(zero<Matrix>), RowIndexes, ColumnIndexes>
    zero<indexed::matrix<Matrix, RowIndexes, ColumnIndexes>>{zero<Matrix>};

//! @}

} // namespace fcarouge

#endif // FCAROUGE_INDEXED_LINALG_HPP
