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

#ifndef FCAROUGE_INTERNAL_UPDATE_HPP
#define FCAROUGE_INTERNAL_UPDATE_HPP

#include "utility.hpp"

#include <functional>
#include <tuple>

namespace fcarouge::internal {

template <typename State = double, typename Output = double,
          typename... UpdateTypes>
struct update final {
  using state = State;
  using output = Output;
  using estimate_uncertainty = matrix<state, state>;
  using process_uncertainty = matrix<state, state>;
  using output_uncertainty = matrix<output, output>;
  using state_transition = matrix<state, state>;
  using output_model = matrix<output, state>;
  using gain = matrix<state, output>;
  using innovation = output;
  using innovation_uncertainty = output_uncertainty;
  using observation_state_function =
      std::function<output_model(const state &, const UpdateTypes &...)>;
  using noise_observation_function = std::function<output_uncertainty(
      const state &, const output &, const UpdateTypes &...)>;
  using observation_function =
      std::function<output(const state &, const UpdateTypes &...)>;
  using update_types = std::tuple<UpdateTypes...>;

  // Workaround against C2888 MSVC for generic lambda variable symbol
  // definition in a different namespace.
  update() = default;
  update(state &x_view, estimate_uncertainty &p_view) : x{x_view}, p{p_view} {}

  static inline const auto i{identity_v<matrix<state, state>>};

  state x_storage{zero_v<state>};
  state &x{x_storage};
  estimate_uncertainty p_storage{identity_v<estimate_uncertainty>};
  estimate_uncertainty &p{p_storage};
  process_uncertainty q{zero_v<process_uncertainty>};
  output_uncertainty r{zero_v<output_uncertainty>};
  output_model h{identity_v<output_model>};
  gain k{identity_v<gain>};
  innovation y{zero_v<innovation>};
  innovation_uncertainty s{identity_v<innovation_uncertainty>};
  output z{zero_v<output>};
  update_types update_arguments{};
  transpose t{};

  observation_state_function observation_state_h{
      [&h = h]([[maybe_unused]] const state &state_x,
               [[maybe_unused]] const UpdateTypes &...update_pack)
          -> output_model { return h; }};
  noise_observation_function noise_observation_r{
      [&r = r]([[maybe_unused]] const state &state_x,
               [[maybe_unused]] const output &output_z,
               [[maybe_unused]] const UpdateTypes &...update_pack)
          -> output_uncertainty { return r; }};
  observation_function observation{
      [&h = h](const state &state_x,
               [[maybe_unused]] const UpdateTypes &...update_pack) -> output {
        return h * state_x;
      }};

  template <typename Output0, typename... OutputN>
  inline constexpr void operator()(const UpdateTypes &...update_pack,
                                   const Output0 &output_z,
                                   const OutputN &...outputs_z) {
    update_arguments = {update_pack...};
    z = output{output_z, outputs_z...};
    h = observation_state_h(x, update_pack...);
    r = noise_observation_r(x, z, update_pack...);
    s = innovation_uncertainty{h * p * t(h) + r};
    k = p * t(h) / s;
    y = z - observation(x, update_pack...);
    x = state{x + k * y};
    p = estimate_uncertainty{(i - k * h) * p * t(i - k * h) + k * r * t(k)};
  }
};

} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_UPDATE_HPP
