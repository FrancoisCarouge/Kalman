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

#ifndef FCAROUGE_INTERNAL_FORMAT_HPP
#define FCAROUGE_INTERNAL_FORMAT_HPP

//! @file
//! @brief Kalman filter formatting support.

#include "fcarouge/kalman.hpp"

#include <format>

namespace fcarouge
{
template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename UpdateArguments,
          typename PredictionArguments>
class kalman;
} // namespace fcarouge

//! @brief Specialization of the standard formatter for the Kalman filter.
//!
//! @details Defines the formatting rules for the filter. This formatter
//! specialization is usually not directly accessed, but is used through
//! formatting functions.
//!
//! @requirement Formatter
template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename UpdateArguments,
          typename PredictionArguments, typename Char>
struct std::formatter<
    fcarouge::kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide,
                     Identity, UpdateArguments, PredictionArguments>,
    Char> {
  //! @name Public Member Functions
  //! @{

  constexpr auto parse(auto &parse_context)
  {
    return value_formatter.parse(parse_context);
  }

  //! @brief Formats the filter.
  //!
  //! @details Formats the filter and values according to the specifiers.
  //! Writes the output to the context and returns an end iterator of the output
  //! range.
  constexpr auto
  format(const fcarouge::kalman<Type, State, Output, Input, Transpose,
                                Symmetrize, Divide, Identity, UpdateArguments,
                                PredictionArguments> &filter,
         auto &format_context) const
  {
    fmt::format_to(format_context.out(), "{}", "{f:");
    value_formatter.format(filter.f(), format_context);
    fmt::format_to(format_context.out(), "{}", ",g:");
    value_formatter.format(filter.g(), format_context);
    fmt::format_to(format_context.out(), "{}", ",h:");
    value_formatter.format(filter.h(), format_context);
    fmt::format_to(format_context.out(), "{}", ",k:");
    value_formatter.format(filter.k(), format_context);
    fmt::format_to(format_context.out(), "{}", ",p:");
    value_formatter.format(filter.p(), format_context);
    fmt::format_to(format_context.out(), "{}", ",q:");
    value_formatter.format(filter.q(), format_context);
    fmt::format_to(format_context.out(), "{}", ",r:");
    value_formatter.format(filter.r(), format_context);
    fmt::format_to(format_context.out(), "{}", ",s:");
    value_formatter.format(filter.s(), format_context);
    fmt::format_to(format_context.out(), "{}", ",u:");
    value_formatter.format(filter.u(), format_context);
    fmt::format_to(format_context.out(), "{}", ",x:");
    value_formatter.format(filter.x(), format_context);
    fmt::format_to(format_context.out(), "{}", ",y:");
    value_formatter.format(filter.y(), format_context);
    fmt::format_to(format_context.out(), "{}", ",z:");
    value_formatter.format(filter.z(), format_context);
    fmt::format_to(format_context.out(), "{}", '}');

    return format_context.out();
  }
  //! @}

  private:
  //! @name Private Member Variables
  //! @{

  //! @brief Filter value type standard formatter.
  std::formatter<Type, Char> value_formatter;

  //! @}
};

#endif // FCAROUGE_INTERNAL_FORMAT_HPP
