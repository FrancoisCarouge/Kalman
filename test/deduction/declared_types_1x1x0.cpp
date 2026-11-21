/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.4
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

#include "fcarouge/kalman.hpp"

#include <concepts>
#include <tuple>

namespace fcarouge::test {
namespace {
//! @brief An output model function of the state.
struct model {
  [[nodiscard]] static auto operator()(const float &x) -> float { return x; }
};

//! @brief A transition function of the state and the prediction arguments.
struct propagation {
  [[nodiscard]] static auto operator()(const float &x, const float &step)
      -> float {
    return x + step;
  }
};

//! @brief A state transition function of the state and the prediction
//! arguments.
struct jacobian {
  [[nodiscard]] static auto operator()(const float &x, const float &step)
      -> float {
    return x * step;
  }
};

//! @test Verifies the declared state transition and output model types are
//! part of the filter type, first in the canonical order of its policies,
//! whatever the order of declaration, and are its member types.
using declared_type =
    decltype(kalman{symmetrize, output_model_type<float>, state{0.F},
                    output<float>, estimate_uncertainty{1.F},
                    process_uncertainty{1.F}, output_uncertainty{1.F},
                    output_model{model{}}, state_transition{jacobian{}},
                    transition{propagation{}}, observation{model{}},
                    prediction_types<float>, state_transition_type<float>});

static_assert(
    std::same_as<
        declared_type,
        kalman<kf::x_z_p_q_r_hh_ff_tt_oo_ps<
            float, float, std::tuple<float>, model, jacobian, propagation,
            model,
            kf::policies<kf::state_transition_type_t<float>,
                         kf::output_model_type_t<float>, kf::symmetrize_t>>>>);
static_assert(std::same_as<declared_type::state_transition, float>);
static_assert(std::same_as<declared_type::output_model, float>);
} // namespace
} // namespace fcarouge::test
