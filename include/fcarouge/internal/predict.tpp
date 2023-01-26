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

#ifndef FCAROUGE_INTERNAL_PREDICT_TPP
#define FCAROUGE_INTERNAL_PREDICT_TPP

namespace fcarouge {

template <typename Implementation>
[[nodiscard("The returned state estimate column vector X is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<Implementation>::x() const -> const state & {
  return model.x;
}

template <typename Implementation>
[[nodiscard("The returned state estimate column vector X is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<Implementation>::x() -> state & {
  return model.x;
}

template <typename Implementation>
[[nodiscard("The returned control column vector U is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<Implementation>::u() const
    -> const input &requires(not std::is_same_v<typename Implementation::input, void>) {
                      return model.u;
                    }

template <typename Implementation>
[[nodiscard("The returned estimated covariance matrix P is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<Implementation>::p() const -> const estimate_uncertainty & {
  return model.p;
}

template <typename Implementation>
[[nodiscard("The returned estimated covariance matrix P is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<Implementation>::p() -> estimate_uncertainty & {
  return model.p;
}

template <typename Implementation>
[[nodiscard("The returned state transition matrix F is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<Implementation>::f() const
    -> const state_transition & {
  return model.f;
}

template <typename Implementation>
[[nodiscard("The returned state transition matrix F is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<Implementation>::f()
    -> state_transition & {
  return model.f;
}

template <typename Implementation>
inline constexpr void
predict<Implementation>::f(
    const auto &value, const auto &...values) {
  if constexpr (std::is_convertible_v<decltype(value), state_transition>) {
    model.f = std::move(state_transition{value, values...});
  } else {
    using transition_state_function = decltype(model.transition_state_f);
    model.transition_state_f =
        std::move(transition_state_function{value, values...});
  }
}

} // namespace fcarouge

#endif // FCAROUGE_INTERNAL_PREDICT_TPP
