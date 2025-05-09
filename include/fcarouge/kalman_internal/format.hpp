/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.1
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

#ifndef FCAROUGE_KALMAN_INTERNAL_FORMAT_HPP
#define FCAROUGE_KALMAN_INTERNAL_FORMAT_HPP

//! @file
//! @brief Formatting support for the Kalman filter.

#include "utility.hpp"

#include <cstddef>
#include <format>
#include <tuple>

//! @brief Specialization of the standard formatter for the Kalman filters.
template <fcarouge::kalman_filter Filter, typename Char>
// It is allowed to add template specializations for any standard library class
// template to the namespace std only if the declaration depends on at least one
// program-defined type and the specialization satisfies all requirements for
// the original template, except where such specializations are prohibited.
// NOLINTNEXTLINE(cert-dcl58-cpp)
struct std::formatter<Filter, Char> {
  constexpr auto parse(std::basic_format_parse_context<Char> &parse_context) {
    return parse_context.begin();
  }

  //! @todo P2585 may be useful in simplifying and standardizing the support.
  template <typename OutputIterator>
  constexpr auto
  format(const Filter &filter,
         std::basic_format_context<OutputIterator, Char> &format_context) const
      -> OutputIterator {

    format_context.advance_to(std::format_to(format_context.out(), R"({{)"));

    if constexpr (fcarouge::kalman_internal::has_state_transition_method<
                      Filter>) {
      format_context.advance_to(
          std::format_to(format_context.out(), R"("f": {}, )", filter.f()));
    }

    if constexpr (fcarouge::kalman_internal::has_input_control_method<Filter>) {
      format_context.advance_to(
          std::format_to(format_context.out(), R"("g": {}, )", filter.g()));
    }

    if constexpr (fcarouge::kalman_internal::has_output_model_method<Filter>) {
      format_context.advance_to(
          std::format_to(format_context.out(), R"("h": {}, )", filter.h()));
    }

    format_context.advance_to(std::format_to(
        format_context.out(), R"("k": {}, "p": {}, )", filter.k(), filter.p()));

    if constexpr (fcarouge::has_prediction_types<Filter>) {
      fcarouge::for_constexpr<
          0, fcarouge::size<typename Filter::prediction_types>, 1>(
          [&format_context, &filter](auto position) {
            format_context.advance_to(std::format_to(
                format_context.out(), R"("prediction_{}": {}, )", position(),
                filter.template predict<position>()));
          });
    }

    if constexpr (fcarouge::kalman_internal::has_process_uncertainty_method<
                      Filter>) {
      format_context.advance_to(
          std::format_to(format_context.out(), R"("q": {}, )", filter.q()));
    }

    if constexpr (fcarouge::has_output_uncertainty<Filter>) {
      format_context.advance_to(
          std::format_to(format_context.out(), R"("r": {}, )", filter.r()));
    }

    format_context.advance_to(
        std::format_to(format_context.out(), R"("s": {}, )", filter.s()));

    //! @todo Generalize out internal method concept when MSVC has better
    //! if-constexpr-requires support.
    if constexpr (fcarouge::kalman_internal::has_input_method<Filter>) {
      format_context.advance_to(
          std::format_to(format_context.out(), R"("u": {}, )", filter.u()));
    }

    //! @todo Inconsistent usage of internal?
    if constexpr (fcarouge::has_update_types<Filter>) {
      fcarouge::for_constexpr<0, fcarouge::size<typename Filter::update_types>,
                              1>([&format_context, &filter](auto position) {
        format_context.advance_to(
            std::format_to(format_context.out(), R"("update_{}": {}, )",
                           position(), filter.template update<position>()));
      });
    }

    format_context.advance_to(
        std::format_to(format_context.out(), R"("x": {}, "y": {}, "z": {}}})",
                       filter.x(), filter.y(), filter.z()));

    return format_context.out();
  }
};

#endif // FCAROUGE_KALMAN_INTERNAL_FORMAT_HPP
