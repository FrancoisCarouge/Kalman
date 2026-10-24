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

#ifndef FCAROUGE_LINALG_HPP
#define FCAROUGE_LINALG_HPP

//! @file
//! @brief Scalar type linear algebra with a freestanding implementation.
//!
//! @details A fixed-size, row-major matrix storage over a standard array,
//! relying only on the freestanding subset of the standard library: no memory
//! allocation, no exception, no input-output. None of the third party linear
//! algebra backends supports a freestanding implementation.

#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <utility>

namespace fcarouge::freestanding {
//! @name Types
//! @{

//! @brief Compile-time sized matrix.
//!
//! @details Supporting the element access, the comparison, the addition, the
//! subtraction, the negation, the matrix multiplication, the scaling by a
//! scalar factor, the transposition, the division, and the one and zero
//! matrices.
//!
//! @tparam Type The matrix element type.
//! @tparam Row The number of rows of the matrix.
//! @tparam Column The number of columns of the matrix.
template <typename Type = double, std::size_t Row = 1, std::size_t Column = 1>
struct matrix {
  //! @brief The row-major, zero-initialized elements.
  std::array<Type, Row * Column> elements{};

  //! @brief Constructs a zero matrix.
  constexpr matrix() = default;

  //! @brief Constructs a vector from its elements, without narrowing.
  template <typename... Types>
    requires(sizeof...(Types) == Row * Column && (Row == 1 || Column == 1)) &&
            requires(const Types &...values) {
              std::array<Type, Row * Column>{values...};
            }
  constexpr explicit matrix(const Types &...values) : elements{values...} {}

  //! @brief Constructs a matrix from its rows.
  //!
  //! @details The support expected by the filter from a backend matrix,
  //! declared from a braced list of rows. Missing elements are zero.
  constexpr explicit matrix(
      std::initializer_list<std::initializer_list<Type>> rows) {
    std::size_t row{0};

    for (const auto &columns : rows) {
      std::size_t column{0};

      for (const auto &value : columns) {
        (*this)[row, column] = value;
        ++column;
      }

      ++row;
    }
  }

  //! @brief The one matrix, with ones on the diagonal.
  [[nodiscard]] static constexpr auto identity() -> matrix {
    matrix result;

    for (std::size_t i{0}; i < Row && i < Column; ++i) {
      result[i, i] = Type{1};
    }

    return result;
  }

  //! @brief The zero matrix.
  [[nodiscard]] static constexpr auto zero() -> matrix { return {}; }

  //! @brief Access the element at the given row and column.
  [[nodiscard]] constexpr auto operator[](this auto &&self, std::size_t row,
                                          std::size_t column) -> auto & {
    return self.elements[(row * Column) + column];
  }

  //! @brief Access the element at the given row-major position.
  [[nodiscard]] constexpr auto operator[](this auto &&self,
                                          std::size_t position) -> auto & {
    return self.elements[position];
  }

  //! @brief The transposed matrix.
  [[nodiscard]] constexpr auto transpose() const -> matrix<Type, Column, Row> {
    matrix<Type, Column, Row> result;

    for (std::size_t i{0}; i < Row; ++i) {
      for (std::size_t j{0}; j < Column; ++j) {
        result[j, i] = (*this)[i, j];
      }
    }

    return result;
  }

  //! @brief Element-wise comparison of two matrices of the same shape.
  [[nodiscard]] friend constexpr auto operator==(const matrix &lhs,
                                                 const matrix &rhs)
      -> bool = default;

  //! @brief Element-wise addition of two matrices of the same shape.
  [[nodiscard]] friend constexpr auto operator+(const matrix &lhs,
                                                const matrix &rhs) -> matrix {
    return transform(std::plus{}, lhs, rhs);
  }

  //! @brief Element-wise subtraction of two matrices of the same shape.
  [[nodiscard]] friend constexpr auto operator-(const matrix &lhs,
                                                const matrix &rhs) -> matrix {
    return transform(std::minus{}, lhs, rhs);
  }

  //! @brief Element-wise negation of a matrix.
  [[nodiscard]] friend constexpr auto operator-(const matrix &value) -> matrix {
    return transform(std::negate{}, value);
  }

  //! @brief Scaling of a matrix by a scalar factor, on the right.
  [[nodiscard]] friend constexpr auto operator*(const matrix &lhs,
                                                const Type &rhs) -> matrix {
    return transform(
        [&rhs](const Type &element) -> Type { return element * rhs; }, lhs);
  }

  //! @brief Scaling of a matrix by a scalar factor, on the left.
  [[nodiscard]] friend constexpr auto operator*(const Type &lhs,
                                                const matrix &rhs) -> matrix {
    return transform(
        [&lhs](const Type &element) -> Type { return lhs * element; }, rhs);
  }

  //! @brief Scaling of a matrix by the inverse of a scalar factor.
  [[nodiscard]] friend constexpr auto operator/(const matrix &lhs,
                                                const Type &rhs) -> matrix {
    return transform(
        [&rhs](const Type &element) -> Type { return element / rhs; }, lhs);
  }

  //! @brief Matrix multiplication.
  template <std::size_t Other>
  [[nodiscard]] friend constexpr auto
  operator*(const matrix &lhs, const matrix<Type, Column, Other> &rhs)
      -> matrix<Type, Row, Other> {
    matrix<Type, Row, Other> result;

    for (std::size_t i{0}; i < Row; ++i) {
      for (std::size_t j{0}; j < Other; ++j) {
        for (std::size_t k{0}; k < Column; ++k) {
          result[i, j] += lhs[i, k] * rhs[k, j];
        }
      }
    }

    return result;
  }

  //! @brief Matrix solution to division.
  //!
  //! @details The matrix `X` solution of `X * rhs = lhs`: dividing an `R1 x C`
  //! matrix by an `R2 x C` matrix results in an `R1 x R2` matrix. This
  //! demonstrator solves `rhsᵀ * Xᵀ = lhsᵀ` by a Gauss-Jordan elimination with
  //! partial pivoting, for a square denominator. Other applications could
  //! select a different solver.
  //!
  //! @warning The denominator must be invertible.
  template <std::size_t Other>
  [[nodiscard]] friend constexpr auto
  operator/(const matrix &lhs, const matrix<Type, Other, Column> &rhs)
      -> matrix<Type, Row, Other> {
    static_assert(Other == Column,
                  "The division requires a square denominator matrix.");

    const auto magnitude{[](const Type &value) -> Type {
      return value < Type{} ? -value : value;
    }};

    auto denominator{rhs.transpose()};
    auto result{lhs.transpose()};

    for (std::size_t pivot{0}; pivot < Column; ++pivot) {
      std::size_t largest{pivot};

      for (std::size_t i{pivot + 1}; i < Column; ++i) {
        if (magnitude(denominator[i, pivot]) >
            magnitude(denominator[largest, pivot])) {
          largest = i;
        }
      }

      for (std::size_t j{0}; j < Column; ++j) {
        std::swap(denominator[pivot, j], denominator[largest, j]);
      }

      for (std::size_t j{0}; j < Row; ++j) {
        std::swap(result[pivot, j], result[largest, j]);
      }

      for (std::size_t i{0}; i < Column; ++i) {
        if (i != pivot) {
          const Type factor{denominator[i, pivot] / denominator[pivot, pivot]};

          for (std::size_t j{0}; j < Column; ++j) {
            denominator[i, j] -= factor * denominator[pivot, j];
          }

          for (std::size_t j{0}; j < Row; ++j) {
            result[i, j] -= factor * result[pivot, j];
          }
        }
      }
    }

    for (std::size_t i{0}; i < Column; ++i) {
      for (std::size_t j{0}; j < Row; ++j) {
        result[i, j] /= denominator[i, i];
      }
    }

    return result.transpose();
  }

private:
  //! @brief The function applied to the elements of the given matrices at the
  //! given position.
  template <std::size_t Position>
  [[nodiscard]] static constexpr auto transform_at(const auto &function,
                                                   const auto &...values)
      -> Type {
    return function(std::get<Position>(values.elements)...);
  }

  //! @brief The matrix of the function applied to the elements of the given
  //! matrices at each position.
  [[nodiscard]] static constexpr auto transform(const auto &function,
                                                const auto &...values)
      -> matrix {
    return [&]<std::size_t... Positions>(
               std::index_sequence<Positions...>) -> matrix {
      matrix result;

      result.elements = {transform_at<Positions>(function, values...)...};

      return result;
    }(std::make_index_sequence<Row * Column>{});
  }
};

//! @brief Compile-time sized row vector.
template <typename Type = double, std::size_t Column = 1>
using row_vector = matrix<Type, 1, Column>;

//! @brief Compile-time sized column vector.
template <typename Type = double, std::size_t Row = 1>
using column_vector = matrix<Type, Row, 1>;

//! @}
} // namespace fcarouge::freestanding

namespace fcarouge {
using namespace freestanding;
} // namespace fcarouge

#endif // FCAROUGE_LINALG_HPP
