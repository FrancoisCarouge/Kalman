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

#ifndef FCAROUGE_KALMAN_OPERATOR_HPP
#define FCAROUGE_KALMAN_OPERATOR_HPP

//! @file
//! @brief Kalman operation for standard types.

namespace fcarouge
{
//! @brief Function object for performing matrix transposition.
//!
//! @tparam Type The type template parameter of the matrix.
template <typename Type> struct transpose {
  //! @brief Returns the transpose of `value`.
  //!
  //! @param value Value to compute the transpose of.
  [[nodiscard]] inline constexpr auto
  operator()(const Type &value) const noexcept
  {
    return value;
  }
};

//! @brief Function object for performing matrix symmetrization.
//!
//! @tparam Type The type template parameter of the matrix.
template <typename Type> struct symmetrize {
  //! @brief Returns the symmetrised `value`.
  //!
  //! @param value Value to compute the symmetry of.
  [[nodiscard]] inline constexpr auto
  operator()(const Type &value) const noexcept
  {
    return value;
  }
};

//! @brief Function object for performing matrix division.
//!
//! @tparam Numerator The type template parameter of the dividend.
//! @tparam Denominator The type template parameter of the divisor.
template <typename Numerator, typename Denominator> struct divide {
  //! @brief Returns the quotient of `numerator` and `denominator`.
  //!
  //! @param numerator The dividend of the division.
  //! @param denominator The divisor of the division.
  [[nodiscard]] inline constexpr auto
  operator()(const Numerator &numerator,
             const Denominator &denominator) const noexcept
  {
    return numerator / denominator;
  }
};

//! @brief Function object for providing an identy matrix.
//!
//! @tparam Type The type template parameter of the matrix.
//!
//! @note This function object template should be a variable template. Proposed
//! in paper P2008R0 entitled "Enabling variable template template parameters".
template <typename Type> struct identity {
  //! @brief Returns the identity maxtrix.
  //!
  //! @return The identity matrix `diag(1, 1, ..., 1)`.
  [[nodiscard]] inline constexpr Type operator()() const noexcept
  {
    return 1;
  }
};

} // namespace fcarouge

#endif // FCAROUGE_KALMAN_OPERATOR_HPP
