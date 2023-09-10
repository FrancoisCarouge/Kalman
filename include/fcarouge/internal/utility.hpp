/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.3.0
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

template <typename Type>
concept algebraic = not arithmetic<Type>;

template <typename Type>
concept eigen = requires { typename Type::PlainMatrix; };

struct empty {
  inline constexpr explicit empty([[maybe_unused]] auto &&...any) noexcept {
    // Constructs from anything for all initializations compatibility.
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

template <typename Type, typename... Types> struct first_type {
  using type = Type;
};

template <typename... Types>
using first_t = typename first_type<Types...>::type;

template <auto Value, auto... Values> struct first_value {
  static constexpr auto value{Value};
};

template <auto... Values>
inline constexpr auto first_v{first_value<Values...>::value};

template <auto Begin, decltype(Begin) End, decltype(Begin) Increment,
          typename Function>
constexpr void for_constexpr(Function &&function) {
  if constexpr (Begin < End) {
    function(std::integral_constant<decltype(Begin), Begin>());
    for_constexpr<Begin + Increment, End, Increment>(function);
  }
}

template <typename Dependent>
constexpr auto type_dependent_false{sizeof(Dependent) != sizeof(Dependent)};

template <typename Type> struct not_implemented {
  static constexpr auto none{type_dependent_false<Type>};
  template <auto Size>
  inline constexpr explicit not_implemented(
      [[maybe_unused]] const char (&message)[Size]) {}
  static_assert(none, "This type is not implemented. See message.");
};

inline constexpr auto adl_transpose{
    [](const auto &value) { return transpose(value); }};

struct transpose final {
  template <arithmetic Arithmetic>
  [[nodiscard]] inline constexpr auto
  operator()(const Arithmetic &value) const {
    return value;
  }

  //! @todo Remove and always rely on ADL and require implementation in linear
  //! algebra support?
  template <typename Matrix>
    requires requires(Matrix value) { value.transpose(); }
  [[nodiscard]] inline constexpr auto operator()(const Matrix &value) const {
    return value.transpose();
  }

  template <typename Matrix>
  [[nodiscard]] inline constexpr auto operator()(const Matrix &value) const {
    return adl_transpose(value);
  }
};

//! @todo The dimensional analysis shows the deduction of matrices gives us the
//! correctly sized resulting matrix but the correctness of the units have yet
//! to be proven, nor whether its systematic usage is in fact appropriate.
//! Hypothesis: units are incorrect, usage may be incorrect, for example
//! `state_transition` may actually be unit-less. Note the `lhs` column size and
//! `rhs` row size are the resulting type's column and row sizes, respectively:
//! Lhs [m by n] and Rhs [o by n] -> Result [m by o].
//! @todo Is there a better, simpler, canonical, standard way of doing this type
//! deduction? For example, by doing it directly from the operation itself?
//! There could be simplicity and performance benefits?
struct deducer final {
  // Built-in, arithmetic, standard division support.
  template <arithmetic Lhs, arithmetic Rhs>
  [[nodiscard]] inline constexpr auto operator()(const Lhs &lhs,
                                                 const Rhs &rhs) const
      -> decltype(lhs / rhs);

  // Type-erased matrix first party linear algebra support.
  template <template <typename, auto, auto> typename Matrix, typename Type,
            auto M, auto N, auto O>
    requires(M > 1 || O > 1)
  [[nodiscard]] inline constexpr auto
  operator()(const Matrix<Type, M, N> &lhs, const Matrix<Type, O, N> &rhs) const
      -> Matrix<Type, M, O>;

  template <template <typename, auto, auto> typename Matrix, typename Type,
            auto N>
  [[nodiscard]] inline constexpr auto
  operator()(const Matrix<Type, 1, N> &lhs, const Matrix<Type, 1, N> &rhs) const
      -> Type;

  template <template <typename, auto, auto> typename Lhs, typename Type, auto M>
  [[nodiscard]] inline constexpr auto operator()(const Lhs<Type, M, 1> &lhs,
                                                 arithmetic auto rhs) const
      -> Lhs<Type, M, 1>;

  //! @todo Coerce type and arithmetic to be the same?
  template <template <typename, auto, auto> typename Rhs, typename Type, auto O>
  [[nodiscard]] inline constexpr auto
  operator()(arithmetic auto lhs, const Rhs<Type, O, 1> &rhs) const
      -> Rhs<Type, 1, O>;

  // Type-erased Eigen third party linear algebra support.
  template <eigen Lhs, eigen Rhs>
  [[nodiscard]] inline constexpr auto operator()(const Lhs &lhs,
                                                 const Rhs &rhs) const ->
      typename decltype(lhs * rhs.transpose())::PlainMatrix;

  template <eigen Lhs, arithmetic Rhs>
  [[nodiscard]] inline constexpr auto operator()(const Lhs &lhs,
                                                 const Rhs &rhs) const ->
      typename Lhs::PlainMatrix;

  template <arithmetic Lhs, eigen Rhs>
  [[nodiscard]] inline constexpr auto operator()(const Lhs &lhs,
                                                 const Rhs &rhs) const ->
      typename decltype(rhs.transpose())::PlainMatrix;
};

//! @todo How to return the `emtpy` type if the deducer would fail to help avoid
//! specialization?
template <typename Numerator, typename Denominator>
using quotient =
    std::remove_cvref_t<std::invoke_result_t<deducer, Numerator, Denominator>>;
} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_UTILITY_HPP
