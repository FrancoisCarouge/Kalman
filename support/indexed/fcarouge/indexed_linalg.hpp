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

namespace fcarouge {

//! @todo Provide a conformance concept for the index types?
template <std::size_t Index, typename Indexes>
using index_t = typename std::tuple_element_t<Index, Indexes>::type;

template <std::size_t RowIndex, typename RowIndexes, std::size_t ColumnIndex,
          typename ColumnIndexes>
using element_t =
    product<index_t<RowIndex, RowIndexes>, index_t<ColumnIndex, ColumnIndexes>>;

//! @name Types
//! @{

//! @brief Indexed matrix.
//!
//! @details Compose a matrix into a indexed matrix. Supports type safety. Unit
//! safety. Row and column indexes provide each index type.
//!
//! @tparam Matrix The underlying linear algebra matrix.
//! @tparam RowIndexes The packed types of the row indexes.
//! @tparam ColumnIndexes The packed types of the column indexes.
//!
//! @todo Explore various types of indexes: arithmetic, units, frames, etc...
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
class indexed_matrix {
public:
  inline constexpr indexed_matrix() = default;

  inline constexpr indexed_matrix(const indexed_matrix &other) = default;

  inline constexpr indexed_matrix &
  operator=(const indexed_matrix &other) = default;

  inline constexpr indexed_matrix(indexed_matrix &&other) = default;

  inline constexpr indexed_matrix &operator=(indexed_matrix &&other) = default;

  //! @todo Is this function safe? Correct?
  //! @todo Add other move assignement function?
  template <typename M, typename R, typename C>
  inline constexpr indexed_matrix &
  operator=(const indexed_matrix<M, R, C> &other) {
    data = other.data;
    return *this;
  }

  //!@todo Is this function safe? Correct?
  explicit inline constexpr indexed_matrix(const Matrix &other) : data{other} {}

  template <typename Type>
  inline constexpr explicit indexed_matrix(
      std::initializer_list<std::initializer_list<Type>> rows) {
    static_cast<void>(rows);
  }

  //! @todo Can the tuple packing be avoided altogether?
  explicit inline constexpr indexed_matrix(const auto &...elements)
    requires(size<ColumnIndexes> == 1 && size<RowIndexes> != 1 &&
             sizeof...(elements) == size<RowIndexes>)
  {
    std::tuple element_pack{elements...};
    for_constexpr<0, size<RowIndexes>, 1>([this, &element_pack](auto position) {
      data[position] = std::tuple_element_t<position, RowIndexes>::convert(
          std::get<position>(element_pack));
    });
  }

  //! @todo Can the tuple packing be avoided altogether?
  explicit inline constexpr indexed_matrix(const auto &...elements)
    requires(size<RowIndexes> == 1 && size<ColumnIndexes> != 1 &&
             sizeof...(elements) == size<ColumnIndexes>)
  {
    std::tuple element_pack{elements...};
    for_constexpr<0, size<ColumnIndexes>, 1>([this,
                                              &element_pack](auto position) {
      data[position] = std::tuple_element_t<position, ColumnIndexes>::convert(
          std::get<position>(element_pack));
    });
  }

  //! @todo Is this function safe? Correct?
  template <typename Type>
    requires requires(Type value) { value.data; }
  explicit inline constexpr indexed_matrix(const Type &other)
      : data{other.data} {}

  [[nodiscard]] inline constexpr explicit(false) operator auto() const
    requires(size<RowIndexes> == 1 && size<ColumnIndexes> == 1)
  {
    return data(0, 0); // return not underlying.
  }

  template <auto Index>
  inline constexpr auto operator()()
    requires(size<RowIndexes> != 1 && size<ColumnIndexes> == 1)
  {
    using type = element_t<Index, RowIndexes, 0, ColumnIndexes>;

    return data(Index, 0) * identity<type>;
  }

  // Proxy reference in support of indexed type conversion to and from
  // underlying scalar.
  template <typename Type> struct reference {
    Type &value;

    //! @todo Fix the conversion indexes?
    inline constexpr double operator=(const auto &element) {
      value = std::tuple_element_t<0, RowIndexes>::convert(element);
      return 0.;
    }

    inline constexpr auto operator==(const Type &v) const {
      static_cast<void>(v);
      return value == v;
    }

    inline constexpr auto operator/(const Type &v) const { return value / v; }
  };

  inline constexpr auto operator[](std::size_t i)
    requires(size<RowIndexes> != 1 && size<ColumnIndexes> == 1)
  {
    return reference{data(i, 0)};
  }

  inline constexpr auto operator[](std::size_t i, std::size_t j) {
    return reference{data(i, j)};
  }

  template <std::size_t Index>
  [[nodiscard]] inline constexpr auto operator[]() const {
    using type = element_t<Index, RowIndexes, 0, ColumnIndexes>;

    return data(Index, 0) * identity<type>;
  }

  inline constexpr auto operator()(std::size_t i, std::size_t j) {
    return reference{data(i, j)};
  }

  Matrix data;
};

struct placeholder_index {
  using type = int;
};

using one_row = std::tuple<placeholder_index>;
using one_column = std::tuple<placeholder_index>;

//! @brief Column vector.
template <typename Matrix, typename... RowIndexes>
using indexed_column_vector =
    indexed_matrix<Matrix, std::tuple<RowIndexes...>, one_column>;

//! @brief Row vector.
template <typename Matrix, typename... ColumnIndexes>
using indexed_row_vector =
    indexed_matrix<Matrix, one_row, std::tuple<ColumnIndexes...>>;

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

    evaluate<Matrix> result{value.data};

    return IndexedMatrix<evaluate<transpose<Matrix>>, ColumnIndexes,
                         RowIndexes>{t(result)};
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
inline indexed_matrix<Matrix, RowIndexes, ColumnIndexes>
    identity<indexed_matrix<Matrix, RowIndexes, ColumnIndexes>>{
        identity<Matrix>};

//! @brief The zero matrix indexed specialization.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
inline indexed_matrix<Matrix, RowIndexes, ColumnIndexes>
    zero<indexed_matrix<Matrix, RowIndexes, ColumnIndexes>>{zero<Matrix>};

//! @}

template <typename Matrix1, typename Matrix2, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator+(const indexed_matrix<Matrix1, RowIndexes, ColumnIndexes> &lhs,
          const indexed_matrix<Matrix2, RowIndexes, ColumnIndexes> &rhs) {
  auto result{lhs.data + rhs.data};

  return indexed_matrix<decltype(result), RowIndexes, ColumnIndexes>{result};
}

template <typename Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator+(const indexed_matrix<Matrix, RowIndexes, ColumnIndexes> &lhs,
          Scalar rhs) {
  auto result{lhs.data + rhs};

  return indexed_matrix<decltype(result), RowIndexes, ColumnIndexes>{result};
}

template <typename Matrix1, typename Matrix2, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator-(const indexed_matrix<Matrix1, RowIndexes, ColumnIndexes> &lhs,
          const indexed_matrix<Matrix2, RowIndexes, ColumnIndexes> &rhs) {
  auto result{lhs.data - rhs.data};

  return indexed_matrix<decltype(result), RowIndexes, ColumnIndexes>{result};
}

template <typename Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator-(Scalar lhs,
          const indexed_matrix<Matrix, RowIndexes, ColumnIndexes> &rhs) {
  return indexed_matrix<Matrix, RowIndexes, ColumnIndexes>{lhs - rhs.data};
}

template <typename Matrix1, typename Matrix2, typename RowIndexes,
          typename Indexes, typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator*(const indexed_matrix<Matrix1, RowIndexes, Indexes> &lhs,
          const indexed_matrix<Matrix2, Indexes, ColumnIndexes> &rhs) {
  auto result{lhs.data * rhs.data};

  return indexed_matrix<decltype(result), RowIndexes, ColumnIndexes>{result};
}

template <typename Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator*(Scalar lhs,
          const indexed_matrix<Matrix, RowIndexes, ColumnIndexes> &rhs) {
  return indexed_matrix<Matrix, RowIndexes, ColumnIndexes>{lhs * rhs.data};
}

template <typename Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator*(const indexed_matrix<Matrix, RowIndexes, ColumnIndexes> &lhs,
          Scalar rhs) {
  return indexed_matrix<Matrix, RowIndexes, ColumnIndexes>{lhs.data * rhs};
}

template <typename Matrix1, typename Matrix2, typename RowIndexes1,
          typename RowIndexes2, typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator/(const indexed_matrix<Matrix1, RowIndexes1, ColumnIndexes> &lhs,
          const indexed_matrix<Matrix2, RowIndexes2, ColumnIndexes> &rhs) {
  auto result{lhs.data / rhs.data};

  return indexed_matrix<decltype(result), RowIndexes1, RowIndexes2>{result};
}

template <typename Scalar, typename Matrix, typename RowIndexes,
          typename ColumnIndexes>
[[nodiscard]] inline constexpr auto
operator/(const indexed_matrix<Matrix, RowIndexes, ColumnIndexes> &lhs,
          Scalar rhs) {
  return indexed_matrix<Matrix, RowIndexes, ColumnIndexes>{lhs.data / rhs};
}
} // namespace fcarouge

//! @brief Specialization of the standard formatter for the indexed linear
//! algebra matrix.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes,
          typename Char>
struct std::formatter<
    fcarouge::indexed_matrix<Matrix, RowIndexes, ColumnIndexes>, Char> {
  constexpr auto parse(std::basic_format_parse_context<Char> &parse_context) {
    return parse_context.begin();
  }

  template <typename OutputIterator>
  constexpr auto format(
      const fcarouge::indexed_matrix<Matrix, RowIndexes, ColumnIndexes> &value,
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

            using type = fcarouge::element_t<i, RowIndexes, j, ColumnIndexes>;

            auto element{value.data(i.value, j.value) *
                         fcarouge::identity<type>};

            format_context.advance_to(
                std::format_to(format_context.out(), "{}", element));
          });

      format_context.advance_to(std::format_to(format_context.out(), "]"));
    });

    format_context.advance_to(std::format_to(format_context.out(), "]"));

    return format_context.out();
  }

  template <typename OutputIterator>
  constexpr auto format(
      const fcarouge::indexed_matrix<Matrix, RowIndexes, ColumnIndexes> &value,
      std::basic_format_context<OutputIterator, Char> &format_context) const
      -> OutputIterator
    requires(size<RowIndexes> == 1 && size<ColumnIndexes> != 1)
  {
    format_context.advance_to(std::format_to(format_context.out(), "["));

    fcarouge::for_constexpr<0, size<ColumnIndexes>, 1>([&format_context,
                                                        &value](auto j) {
      if (j > 0) {
        format_context.advance_to(std::format_to(format_context.out(), ", "));
      }

      using type = fcarouge::element_t<0, RowIndexes, j, ColumnIndexes>;

      auto element{value.data(j) * fcarouge::identity<type>};

      format_context.advance_to(
          std::format_to(format_context.out(), "{}", element));
    });

    format_context.advance_to(std::format_to(format_context.out(), "]"));

    return format_context.out();
  }

  template <typename OutputIterator>
  constexpr auto format(
      const fcarouge::indexed_matrix<Matrix, RowIndexes, ColumnIndexes> &value,
      std::basic_format_context<OutputIterator, Char> &format_context) const
      -> OutputIterator
    requires(size<RowIndexes> == 1 && size<ColumnIndexes> == 1)
  {
    using type = fcarouge::element_t<0, RowIndexes, 0, ColumnIndexes>;

    auto element{value.data() * fcarouge::identity<type>};

    format_context.advance_to(
        std::format_to(format_context.out(), "{}", element));

    return format_context.out();
  }
};

#endif // FCAROUGE_INDEXED_LINALG_HPP