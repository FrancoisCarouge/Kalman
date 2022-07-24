/*_  __          _      __  __          _   _
 | |/ /    /\   | |    |  \/  |   /\   | \ | |
 | ' /    /  \  | |    | \  / |  /  \  |  \| |
 |  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
 | . \  / ____ \| |____| |  | |/ ____ \| |\  |
 |_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter for C++
Version 0.1.0
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

#ifndef FCAROUGE_INTERNAL_EIGEN_HPP
#define FCAROUGE_INTERNAL_EIGEN_HPP

//! @file
//! @brief Kalman operation for Eigen 3 types.
//!
//! @details Default customization point objects (CPO).

#include "fcarouge/kalman.hpp"

#include <Eigen/Eigen>

#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>

namespace fcarouge::eigen::internal
{

//! @brief Arithmetic concept.
template <typename Type>
concept arithmetic = std::integral<Type> || std::floating_point<Type>;

//! @brief Function object for performing Eigen matrix transposition.
//!
//! @details Implemented with the Eigen linear algebra library matrices with
//! sizes fixed at compile-time.
struct transpose {
  //! @brief Returns the transpose of `value`.
  //!
  //! @param value Value to compute the transpose of.
  //!
  //! @exception May throw implementation-defined exceptions.
  [[nodiscard]] inline constexpr auto operator()(const auto &value) const
  {
    using type = std::decay_t<decltype(value)>;
    using result_type =
        typename Eigen::Matrix<typename type::Scalar, type::ColsAtCompileTime,
                               type::RowsAtCompileTime>;

    return result_type{ value.transpose() };
  }

  //! @brief Returns the transpose of `value`.
  //!
  //! @param value Value to compute the transpose of.
  //!
  //! @todo Can this be optimized?
  [[nodiscard]] inline constexpr auto
  operator()(const arithmetic auto &value) const
  {
    return value;
  }
};

//! @brief Function object for performing Eigen matrix symmetrization.
//!
//! @details Implemented with the Eigen linear algebra library matrices with
//! sizes fixed at compile-time.
struct symmetrize {
  //! @brief Returns the symmetrized `value`.
  //!
  //! @param value Value to compute the symmetry of.
  //!
  //! @exception May throw implementation-defined exceptions.
  [[nodiscard]] inline constexpr auto operator()(const auto &value) const
  {
    using result_type = std::decay_t<decltype(value)>;

    return result_type{ (value + value.transpose()) / 2 };
  }

  //! @brief Returns the symmetrized `value`.
  //!
  //! @param value Value to compute the symmetry of.
  //!
  //! @todo Can this be optimized?
  [[nodiscard]] inline constexpr auto
  operator()(const arithmetic auto &value) const
  {
    return value;
  }
};

//! @brief Function object for performing Eigen matrix division.
//!
//! @details Implemented with the Eigen linear algebra library matrices with
//! sizes fixed at compile-time.
struct divide {
  //! @brief Returns the quotient of `numerator` and `denominator`.
  //!
  //! @param numerator The dividend matrix of the division. N: m x n
  //! @param denominator The divisor matrix of the division. D: o x n
  //!
  //! @return The quotient matrix. Q: m x o
  //!
  //! @exception May throw implementation-defined exceptions.
  //!
  //! @todo Why compilation fails if we specify the return type in the body of
  //! the function?
  template <typename Numerator, typename Denominator>
  [[nodiscard]] inline constexpr auto
  operator()(const Numerator &numerator, const Denominator &denominator) const
      -> typename Eigen::Matrix<typename std::decay_t<Numerator>::Scalar,
                                std::decay_t<Numerator>::RowsAtCompileTime,
                                std::decay_t<Denominator>::RowsAtCompileTime>
  {
    return denominator.transpose()
        .fullPivHouseholderQr()
        .solve(numerator.transpose())
        .transpose();
  }

  //! @brief Returns the quotient of `numerator` and `denominator`.
  //!
  //! @param numerator The dividend matrix of the division. N: m x 1
  //! @param denominator The divisor value of the division.
  //!
  //! @return The quotient column vector. Q: m x 1
  //!
  //! @exception May throw implementation-defined exceptions.
  //!
  //! @todo Simplify implementation.
  template <typename Numerator>
  [[nodiscard]] inline constexpr auto
  operator()(const Numerator &numerator,
             const arithmetic auto &denominator) const ->
      typename Eigen::Vector<typename std::decay_t<Numerator>::Scalar,
                             std::decay_t<Numerator>::RowsAtCompileTime>
  {
    return Eigen::Matrix<typename std::decay_t<Numerator>::Scalar, 1, 1>{
      denominator
    }
        .transpose()
        .fullPivHouseholderQr()
        .solve(numerator.transpose())
        .transpose();
  }

  //! @brief Returns the quotient of `numerator` and `denominator`.
  //!
  //! @param numerator The dividend value of the division.
  //! @param denominator The divisor matrix of the division. D: o x 1
  //!
  //! @return The quotient row vector. Q: 1 x o
  //!
  //! @exception May throw implementation-defined exceptions.
  //!
  //! @todo Simplify implementation.
  template <typename Denominator>
  [[nodiscard]] inline constexpr auto
  operator()(const arithmetic auto &numerator,
             const Denominator &denominator) const ->
      typename Eigen::RowVector<typename std::decay_t<Denominator>::Scalar,
                                std::decay_t<Denominator>::RowsAtCompileTime>
  {
    return denominator.transpose()
        .fullPivHouseholderQr()
        .solve(Eigen::Matrix<typename std::decay_t<decltype(numerator)>::Scalar,
                             1, 1>{ numerator })
        .transpose();
  }

  //! @brief Returns the quotient of `numerator` and `denominator`.
  //!
  //! @param numerator The dividend value of the division.
  //! @param denominator The divisor value of the division.
  //!
  //! @return The quotient value.
  [[nodiscard]] inline constexpr auto
  operator()(const arithmetic auto &numerator,
             const arithmetic auto &denominator) const
  {
    return numerator / denominator;
  }
};

//! @brief Function object for providing an Eigen identity matrix.
//!
//! @details Implemented with the Eigen linear algebra library matrices with
//! sizes fixed at compile-time.
//!
//! @note Could this function object template be a variable template as proposed
//! in paper P2008R0 entitled "Enabling variable template template parameters"?
struct identity_matrix {
  //! @brief Returns the identity matrix.
  //!
  //! @tparam Type The type template parameter of the matrix.
  //!
  //! @return The identity matrix `diag(1, 1, ..., 1)`.
  //!
  //! @exception May throw implementation-defined exceptions.
  template <typename Type>
  [[nodiscard]] inline constexpr auto operator()() const
  {
    return Type::Identity();
  }

  //! @brief Returns `1`, the 1-by-1 identity matrix equivalent.
  //!
  //! @tparam Type The type template parameter of the value.
  //!
  //! @return The value `1`.
  template <arithmetic Type>
  [[nodiscard]] inline constexpr auto operator()() const noexcept
  {
    return Type{ 1 };
  }
};

//! @todo Improve support and optimize for no input: neither type nor void but
//! an equivalent empty type? Void may be more intuitive, practical for the user
//! although less theoretically correct?
template <typename Type = double, std::size_t State = 1, std::size_t Output = 1,
          std::size_t Input = 1,
          typename UpdateTypes = fcarouge::internal::empty_pack_t,
          typename PredictionTypes = fcarouge::internal::empty_pack_t>
using kalman = fcarouge::kalman<
    std::conditional_t<State == 1, Type, Eigen::Vector<Type, State>>,
    std::conditional_t<Output == 1, Type, Eigen::Vector<Type, Output>>,
    std::conditional_t<
        Input == 0, void,
        std::conditional_t<Input == 1, Type, Eigen::Vector<Type, Input>>>,
    transpose, symmetrize, divide, identity_matrix, UpdateTypes,
    PredictionTypes>;

} // namespace fcarouge::eigen::internal

#endif // FCAROUGE_INTERNAL_EIGEN_HPP
