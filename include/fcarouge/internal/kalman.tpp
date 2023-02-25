/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
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

#ifndef FCAROUGE_INTERNAL_KALMAN_TPP
#define FCAROUGE_INTERNAL_KALMAN_TPP

namespace fcarouge {

template <typename Update, typename Predict>
inline constexpr kalman<Update, Predict>::kalman(
    [[maybe_unused]] const Update &updator,
    [[maybe_unused]] const Predict &predictor)
//: filter{.updator = updator, .predictor = predictor}
{}

template <typename Update, typename Predict>
[[nodiscard("The returned state estimate column vector X is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::x() const -> const state & {
  return filter.x;
}

template <typename Update, typename Predict>
[[nodiscard("The returned state estimate column vector X is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::x() -> state & {
  return filter.x;
}

template <typename Update, typename Predict>
inline constexpr void kalman<Update, Predict>::x(const auto &value,
                                                 const auto &...values) {
  filter.x = std::move(state{value, values...});
}

template <typename Update, typename Predict>
[[nodiscard("The returned observation column vector Z is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::z() const -> const output & {
  return filter.updator.z();
}

template <typename Update, typename Predict>
[[nodiscard("The returned control column vector U is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::u() const
    -> const input &requires(
        not std::is_same_v<typename Predict::input, void>) {
                      return filter.predictor.u();
                    }

template <typename Update, typename Predict>
[[nodiscard("The returned estimated covariance matrix P is unexpectedly "
            "discarded.")]] inline constexpr auto kalman<Update, Predict>::p()
    const -> const estimate_uncertainty & {
  return filter.p;
}

template <typename Update, typename Predict>
[[nodiscard("The returned estimated covariance matrix P is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::p() -> estimate_uncertainty & {
  return filter.p;
}

template <typename Update, typename Predict>
inline constexpr void kalman<Update, Predict>::p(const auto &value,
                                                 const auto &...values) {
  filter.p = std::move(estimate_uncertainty{value, values...});
}

template <typename Update, typename Predict>
[[nodiscard("The returned process noise covariance matrix Q is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::q() const -> const process_uncertainty & {
  return filter.predictor.q();
}

template <typename Update, typename Predict>
[[nodiscard("The returned process noise covariance matrix Q is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::q() -> process_uncertainty & {
  return filter.predictor.q();
}

template <typename Update, typename Predict>
inline constexpr void kalman<Update, Predict>::q(const auto &value,
                                                 const auto &...values) {
  filter.predictor.q(value, values...);
}

template <typename Update, typename Predict>
[[nodiscard("The returned observation noise covariance matrix R is "
            "unexpectedly discarded.")]] inline constexpr auto
kalman<Update, Predict>::r() const -> const output_uncertainty & {
  return filter.updator.r();
}

template <typename Update, typename Predict>
[[nodiscard("The returned observation noise covariance matrix R is "
            "unexpectedly discarded.")]] inline constexpr auto
kalman<Update, Predict>::r() -> output_uncertainty & {
  return filter.updator.r();
}

template <typename Update, typename Predict>
inline constexpr void kalman<Update, Predict>::r(const auto &value,
                                                 const auto &...values) {
  filter.updator.r(value, values...);
}

template <typename Update, typename Predict>
[[nodiscard("The returned state transition matrix F is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::f() const -> const state_transition & {
  return filter.predictor.f();
}

template <typename Update, typename Predict>
[[nodiscard("The returned state transition matrix F is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::f() -> state_transition & {
  return filter.predictor.f();
}

template <typename Update, typename Predict>
inline constexpr void kalman<Update, Predict>::f(const auto &value,
                                                 const auto &...values) {
  filter.predictor.f(value, values...);
}

template <typename Update, typename Predict>
[[nodiscard("The returned observation transition matrix H is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::h() const -> const output_model & {
  return filter.updator.h();
}

template <typename Update, typename Predict>
[[nodiscard("The returned observation transition matrix H is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::h() -> output_model & {
  return filter.updator.h();
}

template <typename Update, typename Predict>
inline constexpr void kalman<Update, Predict>::h(const auto &value,
                                                 const auto &...values) {
  filter.updator.h(value, values...);
}

template <typename Update, typename Predict>
[[nodiscard("The returned control transition matrix G is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::g() const
    -> const input_control &requires(
        not std::is_same_v<typename Predict::input, internal::empty>) {
                              return filter.predictor.g();
                            }

template <typename Update, typename Predict>
[[nodiscard("The returned control transition matrix G is unexpectedly "
            "discarded.")]] inline constexpr auto kalman<Update, Predict>::g()
    -> input_control &requires(
        not std::is_same_v<typename Predict::input, internal::empty>) {
                        return filter.predictor.g();
                      }

template <typename Update, typename Predict>
inline constexpr void kalman<Update, Predict>::g(const auto &value,
                                                 const auto &...values)
  requires(not std::is_same_v<typename Predict::input, internal::empty>)
{
  filter.predictor.g(value, values...);
}

template <typename Update, typename Predict>
[[nodiscard("The returned gain matrix K is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::k() const -> const gain & {
  return filter.updator.k();
}

template <typename Update, typename Predict>
[[nodiscard("The returned innovation column vector Y is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::y() const -> const innovation & {
  return filter.updator.y();
}

template <typename Update, typename Predict>
[[nodiscard("The returned innovation uncertainty matrix S is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::s() const -> const innovation_uncertainty & {
  return filter.updator.s();
}

template <typename Update, typename Predict>
inline constexpr void
kalman<Update, Predict>::transition(const auto &callable) {
  filter.predictor.transition(callable);
}

template <typename Update, typename Predict>
inline constexpr void
kalman<Update, Predict>::observation(const auto &callable) {
  filter.updator.observation(callable);
}

template <typename Update, typename Predict>
inline constexpr void
kalman<Update, Predict>::update(const auto &...arguments) {
  filter.update(arguments...);
}

template <typename Update, typename Predict>
template <std::size_t Position>
[[nodiscard("The returned update argument is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::update() const {
  return filter.updator.template operator()<Position>();
}

template <typename Update, typename Predict>
inline constexpr void
kalman<Update, Predict>::predict(const auto &...arguments) {
  filter.predict(arguments...);
}

template <typename Update, typename Predict>
template <std::size_t Position>
[[nodiscard("The returned prediction argument is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Update, Predict>::predict() const {
  return filter.predictor.template operator()<Position>();
}

} // namespace fcarouge

#endif // FCAROUGE_INTERNAL_KALMAN_TPP
