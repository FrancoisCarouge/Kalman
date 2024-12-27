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

#ifndef FCAROUGE_QUANTITY_LINALG_HPP
#define FCAROUGE_QUANTITY_LINALG_HPP

//! @file
//! @brief Indexed-based linear algebra with mp-units and Eigen implementations.

#include "fcarouge/indexed_linalg.hpp"
#include "fcarouge/linalg.hpp"
#include "fcarouge/unit.hpp"

namespace fcarouge {
//! @name Types
//! @{

//! @brief A quantity index based on the mp-units quantity.
//!
//! @todo Constraint index types with a concept: scalar/underlying, type,
//! conversion members?
template <auto Reference> struct quantity_index {
  using scalar = double;
  using type = quantity<Reference, scalar>;

  //! @todo Can we do without this structure altogether with the help of
  //! https://mpusz.github.io/mp-units/latest/users_guide/use_cases/interoperability_with_other_libraries/
  //! Or can this definition be more generic?
  [[nodiscard]] static constexpr auto convert(const auto &value) -> scalar {
    return value.numerical_value_in(value.unit);
  }
};

//! @brief Quantity column vector with mp-units and Eigen implementations.
template <auto... Reference>
using quantity_vector = indexed_column_vector<
    column_vector<typename quantity_index<first_v<Reference...>>::scalar,
                  sizeof...(Reference)>,
    quantity_index<Reference>...>;

//! @}
} // namespace fcarouge

#endif // FCAROUGE_QUANTITY_LINALG_HPP
