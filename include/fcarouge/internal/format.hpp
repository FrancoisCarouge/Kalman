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

#include <format>

namespace fcarouge
{
template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
class kalman;
} // namespace fcarouge

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes, typename Char>
struct std::formatter<
    fcarouge::kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide,
                     Identity, UpdateTypes, PredictionTypes>,
    Char> {
  //! @todo Support parsing arguments.
  constexpr auto parse(std::basic_format_parse_context<Char> &parse_context)
  {
    return parse_context.begin();
  }

  // @todo How to support different nested types?
  template <typename OutputIt>
  auto format(const fcarouge::kalman<Type, State, Output, Input, Transpose,
                                     Symmetrize, Divide, Identity, UpdateTypes,
                                     PredictionTypes> &filter,
              std::basic_format_context<OutputIt, Char> &format_context)
      -> OutputIt
  {
    return format_to(
        format_context.out(),
        "{{f:{},g:{},h:{},k:{},p:{},q:{},r:{},s:{},u:{},x:{},y:{},z:{}}}",
        filter.f(), filter.g(), filter.h(), filter.k(), filter.p(), filter.q(),
        filter.r(), filter.s(), filter.u(), filter.x(), filter.y(), filter.z());
  }
};

template <typename Type, std::size_t State, std::size_t Output,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename UpdateTypes, typename PredictionTypes,
          typename Char>
struct std::formatter<
    fcarouge::kalman<Type, State, Output, 0, Transpose, Symmetrize, Divide,
                     Identity, UpdateTypes, PredictionTypes>,
    Char> {
  //! @todo Support parsing arguments.
  constexpr auto parse(std::basic_format_parse_context<Char> &parse_context)
  {
    return parse_context.begin();
  }

  template <typename OutputIt>
  auto format(const fcarouge::kalman<Type, State, Output, 0, Transpose,
                                     Symmetrize, Divide, Identity, UpdateTypes,
                                     PredictionTypes> &filter,
              std::basic_format_context<OutputIt, Char> &format_context)
      -> OutputIt
  {
    return format_to(format_context.out(),
                     "{{f:{},h:{},k:{},p:{},q:{},r:{},s:{},x:{},y:{},z:{}}}",
                     filter.f(), filter.h(), filter.k(), filter.p(), filter.q(),
                     filter.r(), filter.s(), filter.x(), filter.y(),
                     filter.z());
  }
};

#endif // FCAROUGE_INTERNAL_FORMAT_HPP
