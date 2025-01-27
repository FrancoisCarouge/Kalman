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

#ifndef FCAROUGE_NAIVE_HPP
#define FCAROUGE_NAIVE_HPP

//! @file
//! @brief Linear algebra array-based naive implementation.
//!
//! @details Matrix, vectors, and named algebraic values.

#include "fcarouge/utility.hpp"

#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

namespace fcarouge::naive {

//! @name Types
//! @{

//! @brief Naive matrix.
//!
//! @details An array-of-arrays naive implementation matrix.
//!
//! @tparam Type The matrix element type.
//! @tparam Row The number of rows of the matrix.
//! @tparam Column The number of columns of the matrix.
template <typename Type = double, std::size_t Row = 1, std::size_t Column = 1>
struct matrix {
  inline constexpr matrix() = default;

  inline constexpr matrix(const matrix &other) = default;

  inline constexpr matrix &operator=(const matrix &other) = default;

  inline constexpr matrix(matrix &&other) = default;

  inline constexpr matrix &operator=(matrix &&other) = default;

  inline constexpr matrix(const std::same_as<Type> auto &...elements)
    requires(Row == 1 && sizeof...(elements) == Column)
      : data{{elements...}} {}

  inline constexpr matrix(const std::same_as<Type> auto &...elements)
    requires(Row != 1 && Column == 1 && sizeof...(elements) == Row)
  {
    decltype(Row) i{0};
    ([&] { data[i++][0] = elements; }(), ...);
  }

  inline constexpr explicit matrix(const Type (&elements)[Column])
    requires(Row == 1)
  {
    for (decltype(Column) j{0}; j < Column; ++j) {
      data[0][j] = elements[j];
    }
  }

  inline constexpr explicit matrix(const Type (&elements)[Row])
    requires(Row != 1 && Column == 1)
  {
    for (decltype(Row) i{0}; i < Row; ++i) {
      data[i][0] = elements[i];
    }
  }

  template <typename... Types, std::size_t... Columns>
  matrix(const Types (&...rows)[Columns])
    requires(std::conjunction_v<std::is_same<Type, Types>...> &&
             ((Columns == Column) && ... && true))
  {
    decltype(Row) i{0};
    (
        [&](const auto &row) {
          for (decltype(Column) j{0}; j < Column; ++j) {
            data[i][j] = row[j];
          }
          ++i;
        }(rows),
        ...);
  }

  inline constexpr explicit matrix(
      std::initializer_list<std::initializer_list<Type>> rows) {

    for (decltype(Row) i{0}; const auto &row : rows) {
      for (decltype(Column) j{0}; const auto &element : row) {
        data[i][j] = element;
        ++j;
      }
      ++i;
    }
  }

  [[nodiscard]] inline constexpr explicit(false) operator Type() const
    requires(Row == 1 && Column == 1)
  {
    return data[0][0];
  }

  //! @todo Shorten with self deduced this?
  [[nodiscard]] inline constexpr Type &operator[](std::size_t index)
    requires(Row != 1 && Column == 1)
  {
    return data[index][0];
  }

  [[nodiscard]] inline constexpr const Type &operator[](std::size_t index) const
    requires(Row != 1 && Column == 1)
  {
    return data[index][0];
  }

  [[nodiscard]] inline constexpr Type &operator[](std::size_t index)
    requires(Row == 1)
  {
    return data[0][index];
  }

  [[nodiscard]] inline constexpr const Type &operator[](std::size_t index) const
    requires(Row == 1)
  {
    return data[0][index];
  }

  [[nodiscard]] inline constexpr Type &operator[](std::size_t row,
                                                  std::size_t column) {
    return data[row][column];
  }

  [[nodiscard]] inline constexpr const Type &
  operator[](std::size_t row, std::size_t column) const {
    return data[row][column];
  }

  [[nodiscard]] inline constexpr Type &operator()(std::size_t index)
    requires(Row != 1 && Column == 1)
  {
    return data[index][0];
  }

  [[nodiscard]] inline constexpr const Type &operator()(std::size_t index) const
    requires(Row != 1 && Column == 1)
  {
    return data[index][0];
  }

  [[nodiscard]] inline constexpr Type &operator()(std::size_t index)
    requires(Row == 1)
  {
    return data[0][index];
  }

  [[nodiscard]] inline constexpr const Type &operator()(std::size_t index) const
    requires(Row == 1)
  {
    return data[0][index];
  }

  [[nodiscard]] inline constexpr Type &operator()(std::size_t row,
                                                  std::size_t column) {
    return data[row][column];
  }

  [[nodiscard]] inline constexpr const Type &
  operator()(std::size_t row, std::size_t column) const {
    return data[row][column];
  }

  [[no_unique_address]] Type data[Row][Column]{};
};

//! @brief Row vector.
template <typename Type = double, std::size_t Column = 1>
using row_vector = matrix<Type, decltype(Column){1}, Column>;

//! @brief Column vector.
template <typename Type = double, std::size_t Row = 1>
using column_vector = matrix<Type, Row, decltype(Row){1}>;

//! @}

//! @name Deduction Guides
//! @{

template <typename Type> matrix(Type) -> matrix<Type, 1, 1>;

template <typename Type, std::size_t Row, std::size_t Column>
matrix(const Type (&)[Row][Column]) -> matrix<Type, Row, Column>;

template <typename Type, std::size_t Row>
matrix(const Type (&)[Row]) -> matrix<Type, Row, 1>;

template <typename... Types, std::size_t... Columns>
  requires(std::conjunction_v<std::is_same<first<Types...>, Types>...> &&
           ((Columns == first_v<Columns>) && ... && true))
matrix(const Types (&...rows)[Columns])
    -> matrix<std::remove_cvref_t<first<Types...>>, sizeof...(Columns),
              //! @todo Should this be `first_v<Columns...>` instead?
              (Columns, ...)>;

//! @}

template <typename Type, std::size_t Row, std::size_t Column>
[[nodiscard]] inline constexpr bool
operator==(const matrix<Type, Row, Column> &lhs,
           const matrix<Type, Row, Column> &rhs) {
  for (decltype(Row) i{0}; i < Row; ++i) {
    for (decltype(Column) j{0}; j < Column; ++j) {
      if (lhs.data[i][j] != rhs.data[i][j]) {
        return false;
      }
    }
  }
  return true;
}

template <typename Type, std::size_t Row, std::size_t Column, std::size_t Size>
[[nodiscard]] inline constexpr auto
operator*(const matrix<Type, Row, Size> &lhs,
          const matrix<Type, Size, Column> &rhs) {
  matrix<Type, Row, Column> result;

  for (decltype(Row) i{0}; i < Row; ++i) {
    for (decltype(Column) j{0}; j < Column; ++j) {
      for (decltype(Size) k{0}; k < Size; ++k) {
        result.data[i][j] += lhs.data[i][k] * rhs.data[k][j];
      }
    }
  }

  return result;
}

template <typename Type>
[[nodiscard]] inline constexpr auto operator*(arithmetic auto lhs,
                                              const matrix<Type, 1, 1> rhs) {
  return lhs * rhs.data[0][0];
}

template <typename Type, std::size_t Column>
[[nodiscard]] inline constexpr auto operator*(arithmetic auto lhs,
                                              matrix<Type, 1, Column> rhs) {
  matrix<Type, 1, Column> result;

  for (decltype(Column) j{0}; j < Column; ++j) {
    result.data[0][j] = lhs * rhs.data[0][j];
  }

  return result;
}

template <typename Type, std::size_t Row, std::size_t Column>
inline constexpr auto &operator*=(matrix<Type, Row, Column> &lhs,
                                  arithmetic auto rhs) {
  for (decltype(Row) i{0}; i < Row; ++i) {
    for (decltype(Column) j{0}; j < Column; ++j) {
      lhs.data[i][j] *= rhs;
    }
  }

  return lhs;
}

template <typename Type, std::size_t Row, std::size_t Column>
[[nodiscard]] inline constexpr auto operator*(matrix<Type, Row, Column> lhs,
                                              arithmetic auto rhs) {
  return lhs *= rhs;
}

template <typename Type, std::size_t Row, std::size_t Column>
[[nodiscard]] inline constexpr auto
operator+(const matrix<Type, Row, Column> &lhs,
          const matrix<Type, Row, Column> &rhs) {
  matrix<Type, Row, Column> result{lhs};

  for (decltype(Row) i{0}; i < Row; ++i) {
    for (decltype(Column) j{0}; j < Column; ++j) {
      result.data[i][j] += rhs.data[i][j];
    }
  }

  return result;
}

template <typename Type>
[[nodiscard]] inline constexpr auto operator+(arithmetic auto lhs,
                                              matrix<Type, 1, 1> rhs) {
  return lhs + rhs.data[0][0];
}

template <typename Type>
[[nodiscard]] inline constexpr auto operator+(matrix<Type, 1, 1> lhs,
                                              arithmetic auto rhs) {
  return lhs.data[0][0] + rhs;
}

template <typename Type, std::size_t Row, std::size_t Column>
[[nodiscard]] inline constexpr auto
operator-(const matrix<Type, Row, Column> &lhs,
          const matrix<Type, Row, Column> &rhs) {
  matrix<Type, Row, Column> result{lhs};

  for (decltype(Row) i{0}; i < Row; ++i) {
    for (decltype(Column) j{0}; j < Column; ++j) {
      result.data[i][j] -= rhs.data[i][j];
    }
  }

  return result;
}

template <typename Type>
[[nodiscard]] inline constexpr auto operator-(arithmetic auto lhs,
                                              const matrix<Type, 1, 1> &rhs) {
  return lhs - rhs.data[0][0];
}

template <typename Type, std::size_t Row>
[[nodiscard]] inline constexpr auto operator/(const matrix<Type, Row, 1> &lhs,
                                              arithmetic auto rhs) {
  matrix<Type, Row, 1> result{lhs};

  for (decltype(Row) i{0}; i < Row; ++i) {
    result.data[i][0] /= rhs;
  }

  return result;
}
} // namespace fcarouge::naive

namespace fcarouge {
//! @brief Specialization of the evaluation type.
//!
//! @note Implementation not needed.
template <typename Type, std::size_t Row, std::size_t Column>
struct evaluater<naive::matrix<Type, Row, Column>> {
  [[nodiscard]] inline constexpr auto
  operator()() const -> naive::matrix<Type, Row, Column>;
};

//! @brief Specialization of the transposer.
template <typename Type, std::size_t Row, std::size_t Column>
struct transposer<naive::matrix<Type, Row, Column>> {
  [[nodiscard]] inline constexpr auto
  operator()(const naive::matrix<Type, Row, Column> &value) const {
    naive::matrix<Type, Column, Row> result;

    for (decltype(Row) i{0}; i < Row; ++i) {
      for (decltype(Column) j{0}; j < Column; ++j) {
        result.data[j][i] = value.data[i][j];
      }
    }

    return result;
  }
};

//! @name Algebraic Named Values
//! @{

//! @brief The identity matrix naive specialization.
template <typename Type, std::size_t Row, std::size_t Column>
inline constexpr naive::matrix<Type, Row, Column>
    identity<naive::matrix<Type, Row, Column>>{[] {
      naive::matrix<Type, Row, Column> result;
      std::size_t size{Row < Column ? Row : Column};

      for (decltype(size) k{0}; k < size; ++k) {
        result.data[k][k] = 1.0;
      }

      return result;
    }()};

//! @brief The zero matrix naive specialization.
template <typename Type, std::size_t Row, std::size_t Column>
inline constexpr naive::matrix<Type, Row, Column>
    zero<naive::matrix<Type, Row, Column>>{};

//! @}

} // namespace fcarouge

#endif // FCAROUGE_NAIVE_HPP
