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

#ifndef FCAROUGE_KALMAN_EIGEN_HPP
#define FCAROUGE_KALMAN_EIGEN_HPP

//! @file
//! @brief Kalman operation for Eigen 3 types.

#include "internal/eigen.hpp"
#include "internal/kalman.hpp"

#include <cstddef>

namespace fcarouge::eigen
{
//! @brief Function object for performing Eigen matrix transposition.
using transpose = internal::transpose;

//! @brief Function object for performing Eigen matrix symmetrization.
using symmetrize = internal::symmetrize;

//! @brief Function object for performing Eigen matrix division.
using divide = internal::divide;

//! @brief Function object for providing an Eigen identity matrix.
using identity_matrix = internal::identity_matrix;

//! @brief Eigen-based Kalman filter.
//!
//! @details Implemented with the Eigen linear algebra library matrices with
//! sizes fixed at compile-time.
//!
//! @tparam Type The type template parameter of the matrices data.
//! @tparam State The non-type template parameter size of the state vector X.
//! @tparam Output The non-type template parameter size of the measurement
//! vector Z.
//! @tparam Input The non-type template parameter size of the control U.
//! @tparam UpdateTypes The additional update function parameter types passed in
//! through a tuple-like parameter type, composing zero or more types.
//! Parameters such as delta times, variances, or linearized values. The
//! parameters are propagated to the function objects used to compute the state
//! observation H and the observation noise R matrices. The parameters are also
//! propagated to the state observation function object h.
//! @tparam PredictionTypes The additional prediction function parameter types
//! passed in through a tuple-like parameter type, composing zero or more types.
//! Parameters such as delta times, variances, or linearized values. The
//! parameters are propagated to the function objects used to compute the
//! process noise Q, the state transition F, and the control transition G
//! matrices. The parameters are also propagated to the state transition
//! function object f.
template <typename Type = double, std::size_t State = 1, std::size_t Output = 1,
          std::size_t Input = 1,
          typename UpdateTypes = fcarouge::internal::empty_pack_t,
          typename PredictionTypes = fcarouge::internal::empty_pack_t>
using kalman =
    internal::kalman<Type, State, Output, Input, UpdateTypes, PredictionTypes>;

} // namespace fcarouge::eigen

#endif // FCAROUGE_KALMAN_EIGEN_HPP
