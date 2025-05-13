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

#ifndef FCAROUGE_TYPED_LINEAR_ALGEBRA_HPP
#define FCAROUGE_TYPED_LINEAR_ALGEBRA_HPP

//! @file
//! @brief Index typed linear algebra implementation.
//!
//! @details Matrix, vectors, and named algebraic values.

#include "typed_linear_algebra_internal/utility.hpp"

#include <concepts>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <tuple>

namespace fcarouge {

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
//!
//! @note Type safety cannot be guaranteed at compilation time without index
//! safety. The index can either be non-type template parameters or strong types
//! overloadings. Converting a runtime index to a dependent template type is not
//! possible. A proxy reference could be used to allow traditional assignment
//! syntax but the runtime check and extra indirection are not interesting
//! tradeoffs. A template call operator can be used for getting a type safe
//! value but impractical syntax for setting. Without index safety, the accepted
//! tradeoff is a templated index `at<i, j>()` method.
//!
//! @note Deduction guides are tricky because a given element type comes from
//! a row and column index to be deduced.
template <typed_linear_algebra_internal::algebraic Matrix, typename RowIndexes,
          typename ColumnIndexes>
struct typed_matrix {
  //! @todo Privatize this section.
public:
  //! @name Private Member Types
  //! @{

  //! @brief The type of the element's underlying storage.
  using underlying = typed_linear_algebra_internal::underlying_t<Matrix>;

  //! @}

  //! @name Private Member Functions
  //! @{

  //! @todo Can this be removed altogether?
  explicit inline constexpr typed_matrix(const Matrix &other) : data{other} {}

  //! @}

  //! @name Private Member Variables
  //! @{

  Matrix data;

  //! @}

public:
  //! @name Public Member Types
  //! @{

  //! @brief The tuple with the row components of the indexes.
  using row_indexes = RowIndexes;

  //! @brief The tuple with the column components of the indexes.
  using column_indexes = ColumnIndexes;

  //! @brief The type of the element at the given matrix indexes position.
  template <std::size_t RowIndex, std::size_t ColumnIndex>
  using element = typed_linear_algebra_internal::element<typed_matrix, RowIndex,
                                                         ColumnIndex>;

  //! @}

  //! @name Public Member Variables
  //! @{

  //! @brief The count of rows.
  inline constexpr static std::size_t rows{
      typed_linear_algebra_internal::size<row_indexes>};

  //! @brief The count of rows.
  inline constexpr static std::size_t columns{
      typed_linear_algebra_internal::size<column_indexes>};

  //! @}

  //! @name Public Member Functions
  //! @{

  inline constexpr typed_matrix() = default;

  inline constexpr typed_matrix(const typed_matrix &other) = default;

  inline constexpr typed_matrix &operator=(const typed_matrix &other) = default;

  inline constexpr typed_matrix(typed_matrix &&other) = default;

  inline constexpr typed_matrix &operator=(typed_matrix &&other) = default;

  //! @todo Requires evaluated types of Matrix and OtherMatrix are identical?
  template <typed_linear_algebra_internal::algebraic OtherMatrix>
  inline constexpr typed_matrix(
      const typed_matrix<OtherMatrix, RowIndexes, ColumnIndexes> &other)
      : data{other.data} {}

  inline constexpr explicit typed_matrix(const element<0, 0> (
      &elements)[typed_linear_algebra_internal::size<RowIndexes> *
                 typed_linear_algebra_internal::size<ColumnIndexes>])
    requires typed_linear_algebra_internal::uniform<typed_matrix> &&
             typed_linear_algebra_internal::one_dimension<typed_matrix>
      : data{elements} {}

  template <typed_linear_algebra_internal::arithmetic Type>
    requires typed_linear_algebra_internal::singleton<typed_matrix>
  explicit inline constexpr typed_matrix(const Type &value) {
    data(0, 0) = typed_linear_algebra_internal::element_traits<
        underlying, Type>::to_underlying(value);
  }

  //! @todo Verify the list sizes at runtime?
  template <typename Type>
  inline constexpr explicit typed_matrix(
      std::initializer_list<std::initializer_list<Type>> row_list)
    requires typed_linear_algebra_internal::uniform<typed_matrix>
  {
    for (std::size_t i{0}; const auto &row : row_list) {
      for (std::size_t j{0}; const auto &value : row) {
        data(i, j) = typed_linear_algebra_internal::element_traits<
            underlying, Type>::to_underlying(value);
        ++j;
      }
      ++i;
    }
  }

  //! @todo Combine the two constructors in ome?
  //! @todo Verify if the types are the same, or assignable, for nicer error?
  //! @todo Rewrite with a fold expression over the pack?
  template <typename... Types>
    requires typed_linear_algebra_internal::row<typed_matrix> &&
             (not typed_linear_algebra_internal::column<typed_matrix>) &&
             typed_linear_algebra_internal::same_size<ColumnIndexes,
                                                      std::tuple<Types...>>
  explicit inline constexpr typed_matrix(const Types &...values) {
    std::tuple value_pack{values...};
    typed_linear_algebra_internal::for_constexpr<
        0, typed_linear_algebra_internal::size<ColumnIndexes>, 1>(
        [this, &value_pack](auto position) {
          auto value{std::get<position>(value_pack)};
          using type = std::remove_cvref_t<decltype(value)>;
          data[position] = typed_linear_algebra_internal::element_traits<
              underlying, type>::to_underlying(value);
        });
  }

  template <typename... Types>
    requires typed_linear_algebra_internal::column<typed_matrix> &&
             (not typed_linear_algebra_internal::row<typed_matrix>) &&
             typed_linear_algebra_internal::same_size<RowIndexes,
                                                      std::tuple<Types...>>
  inline constexpr typed_matrix(const Types &...values) {
    std::tuple value_pack{values...};
    typed_linear_algebra_internal::for_constexpr<
        0, typed_linear_algebra_internal::size<RowIndexes>, 1>(
        [this, &value_pack](auto position) {
          auto value{std::get<position>(value_pack)};
          using type = std::remove_cvref_t<decltype(value)>;
          data[position] = typed_linear_algebra_internal::element_traits<
              underlying, type>::to_underlying(value);
        });
  }

  [[nodiscard]] inline constexpr explicit(false) operator element<0, 0> &()
    requires typed_linear_algebra_internal::singleton<typed_matrix>
  {
    return typed_linear_algebra_internal::element_traits<
        underlying, element<0, 0>>::from_underlying(data(0, 0));
  }

  [[nodiscard]] inline constexpr auto &&operator[](this auto &&self,
                                                   std::size_t index)
    requires typed_linear_algebra_internal::uniform<typed_matrix> &&
             typed_linear_algebra_internal::one_dimension<typed_matrix>
  {
    return std::forward<decltype(self)>(self).data(index);
  }

  [[nodiscard]] inline constexpr auto &&
  operator[](this auto &&self, std::size_t row, std::size_t column)
    requires typed_linear_algebra_internal::uniform<typed_matrix>
  {
    return std::forward<decltype(self)>(self).data(row, column);
  }

  [[nodiscard]] inline constexpr auto &&operator()(this auto &&self,
                                                   std::size_t index)
    requires typed_linear_algebra_internal::uniform<typed_matrix> &&
             typed_linear_algebra_internal::one_dimension<typed_matrix>
  {
    return std::forward<decltype(self)>(self).data(index);
  }

  [[nodiscard]] inline constexpr auto &&
  operator()(this auto &&self, std::size_t row, std::size_t column)
    requires typed_linear_algebra_internal::uniform<typed_matrix>
  {
    return std::forward<decltype(self)>(self).data(row, column);
  }

  template <std::size_t Row, std::size_t Column>
    requires typed_linear_algebra_internal::in_range<
                 Row, 0, typed_linear_algebra_internal::size<RowIndexes>> &&
             typed_linear_algebra_internal::in_range<
                 Column, 0, typed_linear_algebra_internal::size<ColumnIndexes>>
  [[nodiscard]] inline constexpr element<Row, Column> &at() {
    return typed_linear_algebra_internal::
        element_traits<underlying, element<Row, Column>>::from_underlying(
            data(std::size_t{Row}, std::size_t{Column}));
  }

  template <std::size_t Index>
    requires typed_linear_algebra_internal::column<typed_matrix> &&
             typed_linear_algebra_internal::in_range<
                 Index, 0, typed_linear_algebra_internal::size<RowIndexes>>
  [[nodiscard]] inline constexpr element<Index, 0> &at() {
    return typed_linear_algebra_internal::element_traits<
        underlying, element<Index, 0>>::from_underlying(data(std::size_t{
        Index}));
  }

  //! @}
};

//! @brief Row vector.
template <typename Matrix, typename... ColumnIndexes>
using typed_row_vector =
    typed_matrix<Matrix, std::tuple<int>, std::tuple<ColumnIndexes...>>;

//! @brief Column vector.
template <typename Matrix, typename... RowIndexes>
using typed_column_vector =
    typed_matrix<Matrix, std::tuple<RowIndexes...>, std::tuple<int>>;

//! @}

template <typename Matrix1, typename Matrix2, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr bool
operator==(const typed_matrix<Matrix1, RowIndexes, ColumnIndexes> &lhs,
           const typed_matrix<Matrix2, RowIndexes, ColumnIndexes> &rhs) {
  return lhs.data == rhs.data;
}

template <typename Matrix1, typename Matrix2, typename RowIndexes,
          typename ColumnIndexes, typename Indexes>
[[nodiscard]] inline constexpr auto
operator*(const typed_matrix<Matrix1, RowIndexes, Indexes> &lhs,
          const typed_matrix<Matrix2, Indexes, ColumnIndexes> &rhs) {
  return typed_matrix<
      typed_linear_algebra_internal::evaluate<
          typed_linear_algebra_internal::product<Matrix1, Matrix2>>,
      RowIndexes, ColumnIndexes>{lhs.data * rhs.data};
}

template <typed_linear_algebra_internal::arithmetic Scalar, typename Matrix,
          typename RowIndexes, typename ColumnIndexes>
  requires typed_linear_algebra_internal::singleton<Matrix>
[[nodiscard]] inline constexpr auto operator*(Scalar lhs, const Matrix &rhs) {
  return typed_linear_algebra_internal::element<Matrix, 0, 0>{lhs *
                                                              rhs.data(0)};
}

template <typed_linear_algebra_internal::arithmetic Scalar, typename Matrix,
          typename RowIndexes, typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator*(Scalar lhs,
          const typed_matrix<Matrix, RowIndexes, ColumnIndexes> &rhs) {
  return typed_matrix<typed_linear_algebra_internal::evaluate<Matrix>,
                      RowIndexes, ColumnIndexes>{lhs * rhs.data};
}

template <typed_linear_algebra_internal::arithmetic Scalar, typename Matrix,
          typename RowIndexes, typename ColumnIndexes>
  requires typed_linear_algebra_internal::singleton<Matrix>
[[nodiscard]] inline constexpr auto
operator*(const typed_matrix<Matrix, RowIndexes, ColumnIndexes> &lhs,
          Scalar rhs) {
  return typed_linear_algebra_internal::element<Matrix, 0, 0>{lhs.data(0) *
                                                              rhs};
}

template <typed_linear_algebra_internal::arithmetic Scalar, typename Matrix,
          typename RowIndexes, typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator*(const typed_matrix<Matrix, RowIndexes, ColumnIndexes> &lhs,
          Scalar rhs) {
  return typed_matrix<typed_linear_algebra_internal::evaluate<Matrix>,
                      RowIndexes, ColumnIndexes>{lhs.data * rhs};
}

template <typename Matrix1, typename Matrix2, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator+(const typed_matrix<Matrix1, RowIndexes, ColumnIndexes> &lhs,
          const typed_matrix<Matrix2, RowIndexes, ColumnIndexes> &rhs) {
  return typed_matrix<typed_linear_algebra_internal::evaluate<Matrix1>,
                      RowIndexes, ColumnIndexes>{lhs.data + rhs.data};
}

template <typed_linear_algebra_internal::arithmetic Scalar, typename Matrix,
          typename RowIndexes, typename ColumnIndexes>
  requires typed_linear_algebra_internal::singleton<Matrix>
[[nodiscard]] inline constexpr auto operator+(const Matrix &lhs, Scalar rhs) {
  //! @todo Scalar will become Index with constraints.
  return typed_linear_algebra_internal::element<Matrix, 0, 0>{lhs.data(0) +
                                                              rhs};
}

template <typename Matrix1, typename Matrix2, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator-(const typed_matrix<Matrix1, RowIndexes, ColumnIndexes> &lhs,
          const typed_matrix<Matrix2, RowIndexes, ColumnIndexes> &rhs) {
  return typed_matrix<typed_linear_algebra_internal::evaluate<Matrix1>,
                      RowIndexes, ColumnIndexes>{lhs.data - rhs.data};
}

template <typed_linear_algebra_internal::arithmetic Scalar, typename Matrix,
          typename RowIndexes, typename ColumnIndexes>
  requires typed_linear_algebra_internal::singleton<Matrix>
[[nodiscard]] inline constexpr auto operator-(Scalar lhs, const Matrix &rhs) {
  return typed_linear_algebra_internal::element<Matrix, 0, 0>{lhs -
                                                              rhs.data(0)};
}

template <typename Matrix1, typename Matrix2, typename RowIndexes1,
          typename RowIndexes2, typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator/(const typed_matrix<Matrix1, RowIndexes1, ColumnIndexes> &lhs,
          const typed_matrix<Matrix2, RowIndexes2, ColumnIndexes> &rhs) {
  return typed_matrix<
      typed_linear_algebra_internal::evaluate<
          typed_linear_algebra_internal::quotient<Matrix1, Matrix2>>,
      RowIndexes1, RowIndexes2>{lhs.data / rhs.data};
}

template <typed_linear_algebra_internal::arithmetic Scalar, typename Matrix,
          typename RowIndexes, typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator/(const typed_matrix<Matrix, RowIndexes, ColumnIndexes> &lhs,
          Scalar rhs) {
  return typed_matrix<typed_linear_algebra_internal::evaluate<Matrix>,
                      RowIndexes, ColumnIndexes>{lhs.data / rhs};
}

template <typed_linear_algebra_internal::arithmetic Scalar, typename Matrix,
          typename RowIndexes, typename ColumnIndexes>
  requires typed_linear_algebra_internal::singleton<Matrix>
[[nodiscard]] inline constexpr auto operator/(const Matrix &lhs, Scalar rhs) {
  return typed_linear_algebra_internal::element<Matrix, 0, 0>{lhs.data(0) /
                                                              rhs};
}
} // namespace fcarouge

//! @brief Specialization of the standard formatter for the indexed linear
//! algebra matrix.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes,
          typename Char>
struct std::formatter<fcarouge::typed_matrix<Matrix, RowIndexes, ColumnIndexes>,
                      Char> {
  constexpr auto parse(std::basic_format_parse_context<Char> &parse_context) {
    return parse_context.begin();
  }

  template <typename OutputIterator>
  constexpr auto
  format(const fcarouge::typed_matrix<Matrix, RowIndexes, ColumnIndexes> &value,
         std::basic_format_context<OutputIterator, Char> &format_context) const
      -> OutputIterator {
    format_context.advance_to(std::format_to(format_context.out(), "["));

    for (std::size_t i{0};
         i < fcarouge::typed_linear_algebra_internal::size<RowIndexes>; ++i) {
      if (i > 0) {
        format_context.advance_to(std::format_to(format_context.out(), ", "));
      }

      format_context.advance_to(std::format_to(format_context.out(), "["));

      for (std::size_t j{0};
           j < fcarouge::typed_linear_algebra_internal::size<ColumnIndexes>;
           ++j) {
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
  constexpr auto
  format(const fcarouge::typed_matrix<Matrix, RowIndexes, ColumnIndexes> &value,
         std::basic_format_context<OutputIterator, Char> &format_context) const
      -> OutputIterator
    requires fcarouge::typed_linear_algebra_internal::row<
        fcarouge::typed_matrix<Matrix, RowIndexes, ColumnIndexes>>
  {
    format_context.advance_to(std::format_to(format_context.out(), "["));

    for (std::size_t j{0};
         j < fcarouge::typed_linear_algebra_internal::size<ColumnIndexes>;
         ++j) {
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
  constexpr auto
  format(const fcarouge::typed_matrix<Matrix, RowIndexes, ColumnIndexes> &value,
         std::basic_format_context<OutputIterator, Char> &format_context) const
      -> OutputIterator
    requires fcarouge::typed_linear_algebra_internal::singleton<
        fcarouge::typed_matrix<Matrix, RowIndexes, ColumnIndexes>>
  {
    format_context.advance_to(
        std::format_to(format_context.out(), "{}", value.data(0, 0)));

    return format_context.out();
  }
};

#endif // FCAROUGE_TYPED_LINEAR_ALGEBRA_HPP
