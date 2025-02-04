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

#ifndef FCAROUGE_INDEXED_HPP
#define FCAROUGE_INDEXED_HPP

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

namespace fcarouge::indexed {

//! @todo Move to utility.
template <std::size_t RowIndex, typename RowIndexes, std::size_t ColumnIndex,
          typename ColumnIndexes>
using element = product<std::tuple_element_t<RowIndex, RowIndexes>,
                        std::tuple_element_t<ColumnIndex, ColumnIndexes>>;

//! @brief The given row and column indexes form a colum-vector/matrix.
template <typename RowIndexes, typename ColumnIndexes>
concept column = size<ColumnIndexes> == 1;

//! @brief The given row and column indexes form a row-vector/matrix.
template <typename RowIndexes, typename ColumnIndexes>
concept row = size<RowIndexes> == 1;

//! @brief The given row and column indexes form a singleton matrix.
template <typename RowIndexes, typename ColumnIndexes>
concept singleton =
    column<RowIndexes, ColumnIndexes> && row<RowIndexes, ColumnIndexes>;

//! @brief The packs have the same count of types.
template <typename Pack1, typename Pack2>
concept equal_size = size<Pack1> == size<Pack2>;

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
struct matrix {
  inline constexpr matrix() = default;

  inline constexpr matrix(const matrix &other) = default;

  inline constexpr matrix &operator=(const matrix &other) = default;

  inline constexpr matrix(matrix &&other) = default;

  inline constexpr matrix &operator=(matrix &&other) = default;

  template <typename OtherMatrix>
  inline constexpr matrix(
      const matrix<OtherMatrix, RowIndexes, ColumnIndexes> &other)
      : data{other.data} {}

  inline constexpr matrix(const auto &other) : data{other} {}

  template <arithmetic Type>
  inline constexpr explicit matrix(
      std::initializer_list<std::initializer_list<Type>> rows) {
    for (std::size_t i{0}; const auto &row : rows) {
      for (std::size_t j{0}; const auto &value : row) {
        data(i, j) = value;
        ++j;
      }
      ++i;
    }
  }

  template <typename... Types>
    requires column<RowIndexes, ColumnIndexes> &&
             equal_size<RowIndexes, std::tuple<Types...>>
  inline constexpr matrix(const Types &...values) {
    std::tuple value_pack{values...};
    for_constexpr<0, size<RowIndexes>, 1>([this, &value_pack](auto position) {
      data[position] = std::get<position>(value_pack);
    });
  }

  template <typename... Types>
    requires(size<RowIndexes> == 1 && size<ColumnIndexes> != 1 &&
             sizeof...(Types) == size<ColumnIndexes>)
  explicit inline constexpr matrix(const Types &...values) {
    std::tuple value_pack{values...};
    for_constexpr<0, size<ColumnIndexes>, 1>(
        [this, &value_pack](auto position) {
          data[position] = std::get<position>(value_pack);
        });
  }

  [[nodiscard]] inline constexpr explicit(false)
  operator element<0, RowIndexes, 0, ColumnIndexes>() const
    requires singleton<RowIndexes, ColumnIndexes>
  {
    return element<0, RowIndexes, 0, ColumnIndexes>{data(0, 0)};
  }

  [[nodiscard]] inline constexpr auto &&operator[](this auto &&self,
                                                   std::size_t index)
    requires column<RowIndexes, ColumnIndexes> &&
             (not row<RowIndexes, ColumnIndexes>)
  {
    return std::forward<decltype(self)>(self).data(index, 0);
  }

  [[nodiscard]] inline constexpr auto &&operator[](this auto &&self,
                                                   std::size_t index)
    requires row<RowIndexes, ColumnIndexes>
  {
    return std::forward<decltype(self)>(self).data(0, index);
  }

  [[nodiscard]] inline constexpr auto &&
  operator[](this auto &&self, std::size_t row, std::size_t column) {
    return std::forward<decltype(self)>(self).data(row, column);
  }

  [[nodiscard]] inline constexpr auto &&operator()(this auto &&self,
                                                   std::size_t index)
    requires column<RowIndexes, ColumnIndexes> &&
             (not row<RowIndexes, ColumnIndexes>)
  {
    return std::forward<decltype(self)>(self).data(index, 0);
  }

  [[nodiscard]] inline constexpr auto &&operator()(this auto &&self,
                                                   std::size_t index)
    requires row<RowIndexes, ColumnIndexes>
  {
    return std::forward<decltype(self)>(self).data(0, index);
  }

  [[nodiscard]] inline constexpr auto &&
  operator()(this auto &&self, std::size_t row, std::size_t column) {
    return std::forward<decltype(self)>(self).data(row, column);
  }

  Matrix data;
};

//! @brief Row vector.
template <typename Matrix, typename... ColumnIndexes>
using row_vector =
    matrix<Matrix, std::tuple<int>, std::tuple<ColumnIndexes...>>;

//! @brief Column vector.
template <typename Matrix, typename... RowIndexes>
using column_vector =
    matrix<Matrix, std::tuple<RowIndexes...>, std::tuple<int>>;

//! @}

//! @todo Add deduction guides.

template <typename Matrix1, typename Matrix2, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr bool
operator==(const matrix<Matrix1, RowIndexes, ColumnIndexes> &lhs,
           const matrix<Matrix2, RowIndexes, ColumnIndexes> &rhs) {
  return lhs.data == rhs.data;
}

template <typename Matrix1, typename Matrix2, typename RowIndexes,
          typename ColumnIndexes, typename Indexes>
[[nodiscard]] inline constexpr auto
operator*(const matrix<Matrix1, RowIndexes, Indexes> &lhs,
          const matrix<Matrix2, Indexes, ColumnIndexes> &rhs) {
  //! @todo Don't evaluate as much as possible?
  return matrix<evaluate<product<Matrix1, Matrix2>>, RowIndexes, ColumnIndexes>{
      lhs.data * rhs.data};
}

template <typename Matrix1, typename Matrix2, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator+(const matrix<Matrix1, RowIndexes, ColumnIndexes> &lhs,
          const matrix<Matrix2, RowIndexes, ColumnIndexes> &rhs) {
  return matrix<evaluate<Matrix1>, RowIndexes, ColumnIndexes>{lhs.data +
                                                              rhs.data};
}

template <typename Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
  requires singleton<RowIndexes, ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator+(const matrix<Matrix, RowIndexes, ColumnIndexes> &lhs, Scalar rhs) {
  //! @todo Scalar will become Index with constraints.
  return element<0, RowIndexes, 0, ColumnIndexes>{lhs.data(0) + rhs};
}

template <typename Matrix1, typename Matrix2, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator-(const matrix<Matrix1, RowIndexes, ColumnIndexes> &lhs,
          const matrix<Matrix2, RowIndexes, ColumnIndexes> &rhs) {
  return matrix<evaluate<Matrix1>, RowIndexes, ColumnIndexes>{lhs.data -
                                                              rhs.data};
}

template <typename Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
  requires singleton<RowIndexes, ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator-(Scalar lhs, const matrix<Matrix, RowIndexes, ColumnIndexes> &rhs) {
  //! @todo Don't evaluate? Return the expression?
  return element<0, RowIndexes, 0, ColumnIndexes>{lhs - rhs.data(0)};
}

template <typename Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
  requires singleton<RowIndexes, ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator*(Scalar lhs, const matrix<Matrix, RowIndexes, ColumnIndexes> &rhs) {
  return element<0, RowIndexes, 0, ColumnIndexes>{lhs * rhs.data(0)};
}

template <typename Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator*(Scalar lhs, const matrix<Matrix, RowIndexes, ColumnIndexes> &rhs) {
  return matrix<evaluate<Matrix>, RowIndexes, ColumnIndexes>{lhs * rhs.data};
}

template <typename Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
  requires singleton<RowIndexes, ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator*(const matrix<Matrix, RowIndexes, ColumnIndexes> &lhs, Scalar rhs) {
  return element<0, RowIndexes, 0, ColumnIndexes>{lhs.data(0) * rhs};
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

template <fcarouge::arithmetic Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
  requires singleton<RowIndexes, ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator/(const matrix<Matrix, RowIndexes, ColumnIndexes> &lhs, Scalar rhs) {
  return element<0, RowIndexes, 0, ColumnIndexes>{lhs.data(0) / rhs};
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

    for (std::size_t i{0}; i < fcarouge::size<RowIndexes>; ++i) {
      if (i > 0) {
        format_context.advance_to(std::format_to(format_context.out(), ", "));
      }

      format_context.advance_to(std::format_to(format_context.out(), "["));

      for (std::size_t j{0}; j < fcarouge::size<ColumnIndexes>; ++j) {
        if (j > 0) {
          format_context.advance_to(std::format_to(format_context.out(), ", "));
        }

        format_context.advance_to(
            std::format_to(format_context.out(), "{}", value.data(i, j)));
      }

      format_context.advance_to(std::format_to(format_context.out(), "]"));
    }

    format_context.advance_to(std::format_to(format_context.out(), "]"));

    return format_context.out();
  }

  template <typename OutputIterator>
  constexpr auto format(
      const fcarouge::indexed::matrix<Matrix, RowIndexes, ColumnIndexes> &value,
      std::basic_format_context<OutputIterator, Char> &format_context) const
      -> OutputIterator
    requires fcarouge::indexed::row<RowIndexes, ColumnIndexes>
  {
    format_context.advance_to(std::format_to(format_context.out(), "["));

    for (std::size_t j{0}; j < fcarouge::size<ColumnIndexes>; ++j) {
      if (j > 0) {
        format_context.advance_to(std::format_to(format_context.out(), ", "));
      }

      format_context.advance_to(
          std::format_to(format_context.out(), "{}", value.data(0, j)));
    }

    format_context.advance_to(std::format_to(format_context.out(), "]"));

    return format_context.out();
  }

  template <typename OutputIterator>
  constexpr auto format(
      const fcarouge::indexed::matrix<Matrix, RowIndexes, ColumnIndexes> &value,
      std::basic_format_context<OutputIterator, Char> &format_context) const
      -> OutputIterator
    requires fcarouge::indexed::singleton<RowIndexes, ColumnIndexes>
  {
    format_context.advance_to(
        std::format_to(format_context.out(), "{}", value.data(0, 0)));

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

//! @name Algebraic Named Values
//! @{

//! @brief The identity matrix indexed specialization.
//!
//! @todo The identity doesn't really make sense for matrices with units?
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

#endif // FCAROUGE_INDEXED_HPP
