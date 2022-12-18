/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter for C++
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

#ifndef FCAROUGE_INTERNAL_UTILITY_HPP
#define FCAROUGE_INTERNAL_UTILITY_HPP

#include <concepts>
#include <type_traits>

namespace fcarouge::internal {

template <typename Type>
concept arithmetic = std::integral<Type> || std::floating_point<Type>;

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

// USED? ///////////////////////////////////////////////////////////////////////
template <typename From> using repack_t = typename repack<From>::type;

template <typename From> inline constexpr auto repack_s{repack<From>::size};

template <auto Begin, auto End, auto Increment, typename Function>
constexpr void for_constexpr(Function &&function) {
  if constexpr (Begin < End) {
    function(std::integral_constant<decltype(Begin), Begin>());
    for_constexpr<Begin + Increment, End, Increment>(function);
  }
}

template <typename Type>
inline constexpr Type identity_v{
    //! @todo Implement standard, default form.
};

template <arithmetic Arithmetic>
inline constexpr Arithmetic identity_v<Arithmetic>{1};

template <typename Matrix>
  requires requires(Matrix value) { value.Identity(); }
inline const auto identity_v<Matrix>{Matrix::Identity()};

template <typename Type>
inline constexpr Type zero_v{
    //! @todo Implement standard, default form.
};

template <arithmetic Arithmetic>
inline constexpr Arithmetic zero_v<Arithmetic>{0};

template <typename Matrix>
  requires requires(Matrix value) { value.Zero(); }
inline const auto zero_v<Matrix>{Matrix::Zero()};

//! @todo Consider P1169 making all compatible call operators static member
//! functions?
struct transpose final {
  template <arithmetic Arithmetic>
  [[nodiscard]] inline constexpr auto
  operator()(const Arithmetic &value) const {
    return value;
  }

  template <typename Matrix>
    requires requires(Matrix value) { value.transpose(); }
  [[nodiscard]] inline constexpr auto operator()(const Matrix &value) const {
    return value.transpose();
  }
};

//! @todo The dimensional analysis shows the deduction of matrices gives us the
//! correctly sized resulting matrix but the correctness of the units have yet
//! to be proven, nor whether its systematic usage is in fact appropriate.
//! Hypothesis: units are incorrect, usage may be incorrect, for example
//! `state_transition` may actually be unit-less. Note the lhs column size and
//! rhs row size are the resulting type's column and row sizes, respectively:
//! Lhs [m by n] and Rhs [o by n] -> Result [m by o].
//! @todo Is there a better, simpler, canonical, standard way of doing this type
//! deductions?
struct deducer final {
  // Built-in's types deductions.
  template <arithmetic Lhs, arithmetic Rhs>
  [[nodiscard]] inline constexpr auto operator()(const Lhs &lhs,
                                                 const Rhs &rhs) const
      -> decltype(lhs / rhs);

  // Eigen's types deductions.
  template <typename Lhs, typename Rhs>
    requires requires(Lhs lhs, Rhs rhs) {
               typename Lhs::PlainMatrix;
               typename Lhs::PlainMatrix;
             }
  [[nodiscard]] inline constexpr auto operator()(const Lhs &lhs,
                                                 const Rhs &rhs) const ->
      typename decltype(lhs * rhs.transpose())::PlainMatrix;

  template <typename Lhs, arithmetic Rhs>
    requires requires(Lhs lhs) { typename Lhs::PlainMatrix; }
  [[nodiscard]] inline constexpr auto operator()(const Lhs &lhs,
                                                 const Rhs &rhs) const ->
      typename Lhs::PlainMatrix;

  template <arithmetic Lhs, typename Rhs>
    requires requires(Rhs rhs) { typename Rhs::PlainMatrix; }
  [[nodiscard]] inline constexpr auto operator()(const Lhs &lhs,
                                                 const Rhs &rhs) const ->
      typename decltype(rhs.transpose())::PlainMatrix;
};

template <typename Lhs, typename Rhs>
using matrix = std::decay_t<std::invoke_result_t<deducer, Lhs, Rhs>>;

} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_UTILITY_HPP
