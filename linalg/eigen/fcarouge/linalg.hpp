/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.2.0
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
//! @brief Linear algebra facade for Eigen3 third party implementation.
//!
//! @details Standardizes matrix, vectors, and algebraic values type names for
//! the library usage.
//!
//! @note The Eigen3 linear algebra is not constexpr-compatible.

#include "fcarouge/internal/utility.hpp"

#include <Eigen/Eigen>

namespace fcarouge {
//! @name Algebraic Types
//! @{
//! @brief Compile-time sized Eigen3 matrix.
template <typename Type = double, auto Row = 1, auto Column = 1>
using matrix = Eigen::Matrix<Type, Row, Column>;

//! @brief Compile-time sized Eigen3 row vector.
template <typename Type = double, auto Column = 1>
using row_vector = Eigen::RowVector<Type, Column>;

//! @brief Compile-time sized Eigen3 column vector.
template <typename Type = double, auto Row = 1>
using column_vector = Eigen::Vector<Type, Row>;
//! @}

//! @name Algebraic Named Values
//! @{
//! @brief The identity matrix.
template <typename Type = double>
inline const auto identity_v{internal::identity_v<Type>};

//! @brief The zero matrix.
template <typename Type = double>
inline const auto zero_v{internal::zero_v<Type>};
//! @}

} // namespace fcarouge

#endif // FCAROUGE_LINALG_HPP
