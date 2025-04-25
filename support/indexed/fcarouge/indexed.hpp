/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.0
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

//! @brief The underlying storage type of the matrix's elements.
template <typename Matrix>
using underlying_t =
    std::remove_cvref_t<decltype(std::declval<Matrix>()(0, 0))>;

//! @brief The type of the element at the given matrix indexes position.
template <typename Matrix, std::size_t RowIndex, std::size_t ColumnIndex>
using element =
    product<std::tuple_element_t<RowIndex, typename Matrix::row_indexes>,
            std::tuple_element_t<ColumnIndex, typename Matrix::column_indexes>>;

//! @brief Every element types of the matrix are the same.
//!
//! @details Matrices with uniform types are type safe even with the traditional
//! operators.
//!
//! @note A matrix may be uniform with different row and column indexes.
//!
//! @todo There may be a way to write this concepts via two fold expressions.
template <typename Matrix>
concept uniform = []() {
  bool result{true};

  for_constexpr<0, Matrix::rows, 1>([&result](auto i) {
    for_constexpr<0, Matrix::columns, 1>([&result, &i](auto j) {
      result &= std::is_same_v<element<Matrix, i, j>, element<Matrix, 0, 0>>;
    });
  });

  return result;
}();

//! @brief The index is within the range, inclusive.
template <std::size_t Index, std::size_t Begin, std::size_t End>
concept in_range = Begin <= Index && Index <= End;

//! @brief The given matrix is a single column.
template <typename Matrix>
concept column = Matrix::columns == 1;

//! @brief The matrix is a single row.
template <typename Matrix>
concept row = Matrix::rows == 1;

//! @brief The given matrix is a single dimension, that is a row or a column.
template <typename Matrix>
concept one_dimension = column<Matrix> || row<Matrix>;

//! @brief The given row and column indexes form a singleton matrix.
template <typename Matrix>
concept singleton = column<Matrix> && row<Matrix>;

//! @brief The packs have the same count of types.
template <typename Pack1, typename Pack2>
concept same_size = size<Pack1> == size<Pack2>;

//! @brief Element traits for conversions.
template <typename Underlying, typename Type> struct element_traits {
  [[nodiscard]] static inline constexpr Underlying to_underlying(Type value) {
    return value;
  }

  [[nodiscard]] static inline constexpr Type &
  from_underlying(Underlying &value) {
    return value;
  }
};

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
template <algebraic Matrix, typename RowIndexes, typename ColumnIndexes>
struct matrix {
  //! @todo Privatize this section.
public:
  //! @name Private Member Types
  //! @{

  //! @brief The type of the element's underlying storage.
  using underlying = underlying_t<Matrix>;

  //! @}

  //! @name Private Member Functions
  //! @{

  //! @todo Can this be removed altogether?
  explicit inline constexpr matrix(const Matrix &other) : data{other} {}

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
  using element = element<matrix, RowIndex, ColumnIndex>;

  //! @}

  //! @name Public Member Variables
  //! @{

  //! @brief The count of rows.
  inline constexpr static std::size_t rows{size<row_indexes>};

  //! @brief The count of rows.
  inline constexpr static std::size_t columns{size<column_indexes>};

  //! @}

  //! @name Public Member Functions
  //! @{

  inline constexpr matrix() = default;

  inline constexpr matrix(const matrix &other) = default;

  inline constexpr matrix &operator=(const matrix &other) = default;

  inline constexpr matrix(matrix &&other) = default;

  inline constexpr matrix &operator=(matrix &&other) = default;

  //! @todo Requires evaluated types of Matrix and OtherMatrix are identical?
  template <algebraic OtherMatrix>
  inline constexpr matrix(
      const matrix<OtherMatrix, RowIndexes, ColumnIndexes> &other)
      : data{other.data} {}

  inline constexpr explicit matrix(
      const element<0, 0> (&elements)[size<RowIndexes> * size<ColumnIndexes>])
    requires uniform<matrix> && one_dimension<matrix>
      : data{elements} {}

  template <arithmetic Type>
    requires singleton<matrix>
  explicit inline constexpr matrix(const Type &value) {
    data(0, 0) = element_traits<underlying, Type>::to_underlying(value);
  }

  //! @todo Verify the list sizes at runtime?
  template <typename Type>
  inline constexpr explicit matrix(
      std::initializer_list<std::initializer_list<Type>> row_list)
    requires uniform<matrix>
  {
    for (std::size_t i{0}; const auto &row : row_list) {
      for (std::size_t j{0}; const auto &value : row) {
        data(i, j) = element_traits<underlying, Type>::to_underlying(value);
        ++j;
      }
      ++i;
    }
  }

  //! @todo Combine the two constructors in ome?
  //! @todo Verify if the types are the same, or assignable, for nicer error?
  //! @todo Rewrite with a fold expression over the pack?
  template <typename... Types>
    requires row<matrix> && (not column<matrix>) &&
             same_size<ColumnIndexes, std::tuple<Types...>>
  explicit inline constexpr matrix(const Types &...values) {
    std::tuple value_pack{values...};
    for_constexpr<0, size<ColumnIndexes>, 1>([this,
                                              &value_pack](auto position) {
      auto value{std::get<position>(value_pack)};
      using type = std::remove_cvref_t<decltype(value)>;
      data[position] = element_traits<underlying, type>::to_underlying(value);
    });
  }

  template <typename... Types>
    requires column<matrix> &&
             (not row<matrix>) && same_size<RowIndexes, std::tuple<Types...>>
  inline constexpr matrix(const Types &...values) {
    std::tuple value_pack{values...};
    for_constexpr<0, size<RowIndexes>, 1>([this, &value_pack](auto position) {
      auto value{std::get<position>(value_pack)};
      using type = std::remove_cvref_t<decltype(value)>;
      data[position] = element_traits<underlying, type>::to_underlying(value);
    });
  }

  [[nodiscard]] inline constexpr explicit(false) operator element<0, 0> &()
    requires singleton<matrix>
  {
    return element_traits<underlying, element<0, 0>>::from_underlying(
        data(0, 0));
  }

  [[nodiscard]] inline constexpr auto &&operator[](this auto &&self,
                                                   std::size_t index)
    requires uniform<matrix> && one_dimension<matrix>
  {
    return std::forward<decltype(self)>(self).data(index);
  }

  [[nodiscard]] inline constexpr auto &&
  operator[](this auto &&self, std::size_t row, std::size_t column)
    requires uniform<matrix>
  {
    return std::forward<decltype(self)>(self).data(row, column);
  }

  [[nodiscard]] inline constexpr auto &&operator()(this auto &&self,
                                                   std::size_t index)
    requires uniform<matrix> && one_dimension<matrix>
  {
    return std::forward<decltype(self)>(self).data(index);
  }

  [[nodiscard]] inline constexpr auto &&
  operator()(this auto &&self, std::size_t row, std::size_t column)
    requires uniform<matrix>
  {
    return std::forward<decltype(self)>(self).data(row, column);
  }

  template <std::size_t Row, std::size_t Column>
    requires in_range<Row, 0, size<RowIndexes>> &&
             in_range<Column, 0, size<ColumnIndexes>>
  [[nodiscard]] inline constexpr element<Row, Column> &at() {
    return element_traits<underlying, element<Row, Column>>::from_underlying(
        data(std::size_t{Row}, std::size_t{Column}));
  }

  template <std::size_t Index>
    requires column<matrix> && in_range<Index, 0, size<RowIndexes>>
  [[nodiscard]] inline constexpr element<Index, 0> &at() {
    return element_traits<underlying, element<Index, 0>>::from_underlying(
        data(std::size_t{Index}));
  }

  //! @}
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
  return matrix<evaluate<product<Matrix1, Matrix2>>, RowIndexes, ColumnIndexes>{
      lhs.data * rhs.data};
}

template <arithmetic Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
  requires singleton<Matrix>
[[nodiscard]] inline constexpr auto operator*(Scalar lhs, const Matrix &rhs) {
  return element<Matrix, 0, 0>{lhs * rhs.data(0)};
}

template <arithmetic Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator*(Scalar lhs, const matrix<Matrix, RowIndexes, ColumnIndexes> &rhs) {
  return matrix<evaluate<Matrix>, RowIndexes, ColumnIndexes>{lhs * rhs.data};
}

template <arithmetic Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
  requires singleton<Matrix>
[[nodiscard]] inline constexpr auto
operator*(const matrix<Matrix, RowIndexes, ColumnIndexes> &lhs, Scalar rhs) {
  return element<Matrix, 0, 0>{lhs.data(0) * rhs};
}

template <arithmetic Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator*(const matrix<Matrix, RowIndexes, ColumnIndexes> &lhs, Scalar rhs) {
  return matrix<evaluate<Matrix>, RowIndexes, ColumnIndexes>{lhs.data * rhs};
}

template <typename Matrix1, typename Matrix2, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator+(const matrix<Matrix1, RowIndexes, ColumnIndexes> &lhs,
          const matrix<Matrix2, RowIndexes, ColumnIndexes> &rhs) {
  return matrix<evaluate<Matrix1>, RowIndexes, ColumnIndexes>{lhs.data +
                                                              rhs.data};
}

template <arithmetic Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
  requires singleton<Matrix>
[[nodiscard]] inline constexpr auto operator+(const Matrix &lhs, Scalar rhs) {
  //! @todo Scalar will become Index with constraints.
  return element<Matrix, 0, 0>{lhs.data(0) + rhs};
}

template <typename Matrix1, typename Matrix2, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator-(const matrix<Matrix1, RowIndexes, ColumnIndexes> &lhs,
          const matrix<Matrix2, RowIndexes, ColumnIndexes> &rhs) {
  return matrix<evaluate<Matrix1>, RowIndexes, ColumnIndexes>{lhs.data -
                                                              rhs.data};
}

template <arithmetic Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
  requires singleton<Matrix>
[[nodiscard]] inline constexpr auto operator-(Scalar lhs, const Matrix &rhs) {
  return element<Matrix, 0, 0>{lhs - rhs.data(0)};
}

template <typename Matrix1, typename Matrix2, typename RowIndexes1,
          typename RowIndexes2, typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator/(const matrix<Matrix1, RowIndexes1, ColumnIndexes> &lhs,
          const matrix<Matrix2, RowIndexes2, ColumnIndexes> &rhs) {
  return matrix<evaluate<quotient<Matrix1, Matrix2>>, RowIndexes1, RowIndexes2>{
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
  requires singleton<Matrix>
[[nodiscard]] inline constexpr auto operator/(const Matrix &lhs, Scalar rhs) {
  return element<Matrix, 0, 0>{lhs.data(0) / rhs};
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
    requires fcarouge::indexed::row<
        fcarouge::indexed::matrix<Matrix, RowIndexes, ColumnIndexes>>
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
    requires fcarouge::indexed::singleton<
        fcarouge::indexed::matrix<Matrix, RowIndexes, ColumnIndexes>>
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
struct evaluates<IndexedMatrix<Matrix, RowIndexes, ColumnIndexes>> {
  [[nodiscard]] inline constexpr auto operator()() const
      -> IndexedMatrix<evaluate<Matrix>, RowIndexes, ColumnIndexes>;
};

//! @brief Specialization of the transposes.
template <template <typename, typename, typename> typename IndexedMatrix,
          typename Matrix, typename RowIndexes, typename ColumnIndexes>
  requires requires(IndexedMatrix<Matrix, RowIndexes, ColumnIndexes> m) {
    m.data;
  }
struct transposes<IndexedMatrix<Matrix, RowIndexes, ColumnIndexes>> {
  [[nodiscard]] inline constexpr auto operator()(
      const IndexedMatrix<Matrix, RowIndexes, ColumnIndexes> &value) const {
    return IndexedMatrix<evaluate<transpose<Matrix>>, ColumnIndexes,
                         RowIndexes>{t(value.data)};
  }
};

//! @name Algebraic Named Values
//! @{

//! @brief The one matrix indexed specialization.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
inline indexed::matrix<decltype(one<Matrix>), RowIndexes, ColumnIndexes>
    one<indexed::matrix<Matrix, RowIndexes, ColumnIndexes>>{one<Matrix>};

//! @brief The zero matrix indexed specialization.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
inline indexed::matrix<decltype(zero<Matrix>), RowIndexes, ColumnIndexes>
    zero<indexed::matrix<Matrix, RowIndexes, ColumnIndexes>>{zero<Matrix>};

//! @}

} // namespace fcarouge

#endif // FCAROUGE_INDEXED_HPP
