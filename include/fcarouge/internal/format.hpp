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

#ifndef FCAROUGE_INTERNAL_FORMAT_HPP
#define FCAROUGE_INTERNAL_FORMAT_HPP

#include "utility.hpp"

#include <cstddef>
#include <format>

namespace fcarouge {
template <typename, typename, typename, typename, typename, typename, typename,
          typename, typename>
class kalman;
} // namespace fcarouge

template <typename State, typename Output, typename Input, typename Transpose,
          typename Symmetrize, typename Divide, typename Identity,
          typename UpdateTypes, typename PredictionTypes, typename Char>
// It is allowed to add template specializations for any standard library class
// template to the namespace std only if the declaration depends on at least one
// program-defined type and the specialization satisfies all requirements for
// the original template, except where such specializations are prohibited.
// NOLINTNEXTLINE(cert-dcl58-cpp)
struct std::formatter<
    fcarouge::kalman<State, Output, Input, Transpose, Symmetrize, Divide,
                     Identity, UpdateTypes, PredictionTypes>,
    Char> {
  constexpr auto parse(std::basic_format_parse_context<Char> &parse_context) {
    return parse_context.begin();
  }

  //! @todo P2585 may be useful in simplifying and standardizing the support.
  template <typename OutputIt>
  auto format(const fcarouge::kalman<State, Output, Input, Transpose,
                                     Symmetrize, Divide, Identity, UpdateTypes,
                                     PredictionTypes> &filter,
              std::basic_format_context<OutputIt, Char> &format_context)
      -> OutputIt {
    format_context.advance_to(
        format_to(format_context.out(), R"({{"f": {}, )", filter.f()));

    if constexpr (not std::is_same_v<Input, void>) {
      format_context.advance_to(
          format_to(format_context.out(), R"("g": {}, )", filter.g()));
    }

    format_context.advance_to(format_to(format_context.out(),
                                        R"("h": {}, "k": {}, "p": {}, )",
                                        filter.h(), filter.k(), filter.p()));

    fcarouge::internal::for_constexpr<
        std::size_t{0}, fcarouge::internal::repack_s<PredictionTypes>, 1>(
        [&format_context, &filter](auto position) {
          format_context.advance_to(format_to(
              format_context.out(), R"("prediction_{}": {}, )",
              std::size_t{position}, filter.template predict<position>()));
        });

    format_context.advance_to(format_to(format_context.out(),
                                        R"("q": {}, "r": {}, "s": {}, )",
                                        filter.q(), filter.r(), filter.s()));

    if constexpr (not std::is_same_v<Input, void>) {
      format_context.advance_to(
          format_to(format_context.out(), R"("u": {}, )", filter.u()));
    }

    fcarouge::internal::for_constexpr<
        std::size_t{0}, fcarouge::internal::repack_s<UpdateTypes>, 1>(
        [&format_context, &filter](auto position) {
          format_context.advance_to(format_to(
              format_context.out(), R"("update_{}": {}, )",
              std::size_t{position}, filter.template update<position>()));
        });

    format_context.advance_to(format_to(format_context.out(),
                                        R"("x": {}, "y": {}, "z": {}}})",
                                        filter.x(), filter.y(), filter.z()));

    return format_context.out();
  }
};

#endif // FCAROUGE_INTERNAL_FORMAT_HPP
