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

#ifndef FCAROUGE_INTERNAL_UTILITY_HPP
#define FCAROUGE_INTERNAL_UTILITY_HPP

#include <type_traits>

namespace fcarouge::internal {

struct empty {
  inline constexpr explicit empty(auto &&...any) noexcept {
    // Constructs from anything for all initializations compatibility.
    (static_cast<void>(any), ...);
  }
};

template <typename...> struct pack {};

using empty_pack = pack<>;

template <typename Type> struct repack {
  using type = Type;
};

template <template <typename...> typename From, typename... Types>
struct repack<From<Types...>> {
  using type = pack<Types...>;
  static inline constexpr auto size{sizeof...(Types)};
};

template <typename From> using repack_t = typename repack<From>::type;

template <typename From> inline constexpr auto repack_s{repack<From>::size};

template <auto Begin, auto End, auto Increment, typename Function>
constexpr void for_constexpr(Function &&function) {
  if constexpr (Begin < End) {
    function(std::integral_constant<decltype(Begin), Begin>());
    for_constexpr<Begin + Increment, End, Increment>(function);
  }
}

} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_UTILITY_HPP