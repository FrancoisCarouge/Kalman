/*  __          _      __  __          _   _
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

#ifndef FCAROUGE_EIGEN_KALMAN_HPP
#define FCAROUGE_EIGEN_KALMAN_HPP

//! @file
//! @brief Kalman operation for Eigen 3 types.

#include "fcarouge/kalman.hpp"
#include "internal/utility.hpp"

#include <Eigen/Eigen>

namespace fcarouge::eigen {
//! @brief Function object for performing Eigen matrix transposition.
using transpose = internal::transpose;

//! @brief Function object for performing Eigen matrix division.
using divide = internal::divide;

//! @brief Function object for providing an Eigen identity matrix.
using identity_matrix = internal::identity_matrix;

//! @brief Convenience tuple-like empty pack type.
using empty_pack = fcarouge::internal::empty_pack;

//! @brief Convenience tuple-like pack type.
template <typename... Types> using pack = fcarouge::internal::pack<Types...>;

//! @brief Convenience Eigen vector.
template <typename Type, auto Size> using vector = internal::vector<Type, Size>;

//! @brief Convenience Eigen matrix.
template <typename Type, auto RowSize, auto ColumnSize>
using matrix = internal::matrix<Type, RowSize, ColumnSize>;

//! @brief Eigen-based Kalman filter.
//!
//! @details Implemented with the Eigen linear algebra library matrices with
//! sizes fixed at compile-time.
//!
//! @see fcarouge::kalman
template <typename State = double, typename Output = double,
          typename Input = void, typename UpdateTypes = empty_pack,
          typename PredictionTypes = empty_pack>
using kalman = fcarouge::kalman<State, Output, Input, transpose, divide,
                                identity_matrix, UpdateTypes, PredictionTypes>;

} // namespace fcarouge::eigen

#endif // FCAROUGE_EIGEN_KALMAN_HPP
