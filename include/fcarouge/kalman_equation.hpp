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

#ifndef FCAROUGE_KALMAN_EQUATION_HPP
#define FCAROUGE_KALMAN_EQUATION_HPP

//! @file
//! @brief Kalman filter main project header.

#include <type_traits>

namespace fcarouge
{
[[nodiscard]] inline constexpr auto extrapolate_state(const auto &x,
                                                      const auto &f)
{
  using State = std::remove_reference_t<std::remove_cv_t<decltype(x)>>;

  return State{ f * x };
}

[[nodiscard]] inline constexpr auto
extrapolate_state(const auto &x, const auto &f, const auto &g, const auto &u)
{
  using State = std::remove_reference_t<std::remove_cv_t<decltype(x)>>;

  return State{ f * x + g * u };
}

template <template <typename> class Transpose>
[[nodiscard]] inline constexpr auto
extrapolate_covariance(const auto &p, const auto &f, const auto &q)
{
  using estimate_uncertainty_p =
      std::remove_reference_t<std::remove_cv_t<decltype(p)>>;
  using state_transition =
      std::remove_reference_t<std::remove_cv_t<decltype(f)>>;
  Transpose<state_transition> transpose;

  return estimate_uncertainty_p{ f * p * transpose(f) + q };
}

template <template <typename> typename Transpose,
          template <typename> typename Symmetrize>
inline constexpr void predict(auto &x, auto &p, const auto &f, const auto &q)
{
  x = extrapolate_state(x, f);

  using estimate_uncertainty_p =
      std::remove_reference_t<std::remove_cv_t<decltype(p)>>;
  Symmetrize<estimate_uncertainty_p> symmetrize;
  p = symmetrize(extrapolate_covariance<Transpose>(p, f, q));
}

template <template <typename> typename Transpose,
          template <typename> typename Symmetrize>
inline constexpr void predict(auto &x, auto &p, const auto &f, const auto &q,
                              const auto &g, const auto &u)
{
  x = extrapolate_state(x, f, g, u);

  using estimate_uncertainty_p =
      std::remove_reference_t<std::remove_cv_t<decltype(p)>>;
  Symmetrize<estimate_uncertainty_p> symmetrize;
  p = symmetrize(extrapolate_covariance<Transpose>(p, f, q));
}

[[nodiscard]] inline constexpr auto update_state(const auto &x, const auto &k,
                                                 const auto &z, const auto &h)
{
  using State = std::remove_reference_t<std::remove_cv_t<decltype(x)>>;

  return State{ x + k * (z - h * x) };
}

template <template <typename> typename Transpose,
          template <typename> typename Identity>
[[nodiscard]] inline constexpr auto
update_covariance(const auto &p, const auto &k, const auto &h, const auto &r)
{
  using estimate_uncertainty_p =
      std::remove_reference_t<std::remove_cv_t<decltype(p)>>;
  using gain = std::remove_reference_t<std::remove_cv_t<decltype(k)>>;
  Transpose<estimate_uncertainty_p> transpose_p;
  Transpose<gain> transpose_k;
  Identity<estimate_uncertainty_p> i;

  return estimate_uncertainty_p{ (i() - k * h) * p * transpose_p(i() - k * h) +
                                 k * r * transpose_k(k) };
}

template <template <typename> typename Transpose,
          template <typename, typename> typename Divide>
[[nodiscard]] inline constexpr auto weight_gain(const auto &p, const auto &h,
                                                const auto &r)
{
  using observation = std::remove_reference_t<std::remove_cv_t<decltype(h)>>;
  using observation_noise_uncertainty =
      std::remove_reference_t<std::remove_cv_t<decltype(r)>>;
  using gain = std::invoke_result_t<Transpose<observation>, observation>;
  Transpose<observation> transpose_h;
  Divide<gain, observation_noise_uncertainty> divide;

  return gain{ divide(p * transpose_h(h), h * p * transpose_h(h) + r) };
}

template <template <typename> typename Transpose,
          template <typename> typename Symmetrize,
          template <typename, typename> typename Divide,
          template <typename> typename Identity>
inline constexpr void update(auto &x, auto &p, const auto &h, const auto &r,
                             const auto &z)
{
  const auto k{ weight_gain<Transpose, Divide>(p, h, r) };

  x = update_state(x, k, z, h);

  using estimate_uncertainty_p =
      std::remove_reference_t<std::remove_cv_t<decltype(p)>>;
  Symmetrize<estimate_uncertainty_p> symmetrize;
  p = symmetrize(update_covariance<Transpose, Identity>(p, k, h, r));
}

} // namespace fcarouge

#endif // FCAROUGE_KALMAN_EQUATION_HPP
