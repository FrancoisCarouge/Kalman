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

#include <tuple>

namespace fcarouge::test {
namespace {
//! @brief A placeholder callable: storage is the property under test.
struct model {};
struct jacobian {};
struct propagation {};
struct measurement {};
struct noise {};
struct control {};

//! @brief The filter stores update arguments.
template <typename Filter>
concept stores_update_arguments =
    requires(Filter filter) { filter.update_arguments; };

//! @brief The filter stores prediction arguments.
template <typename Filter>
concept stores_prediction_arguments =
    requires(Filter filter) { filter.prediction_arguments; };

//! @test Verifies the filters store only their non-empty argument lists, and
//! that a filter without update nor prediction arguments is smaller than the
//! filter storing both empty lists.
static_assert(
    stores_update_arguments<kf::x_z_p_q_r_hh_f_tt_oo_us<
        double, double, std::tuple<int>, model, propagation, measurement>>);
static_assert(
    not stores_prediction_arguments<kf::x_z_p_q_r_hh_f_tt_oo_us<
        double, double, std::tuple<int>, model, propagation, measurement>>);
static_assert(
    not stores_update_arguments<kf::x_z_p_q_r_hh_f_tt_oo_ps<
        double, double, std::tuple<float>, model, propagation, measurement>>);
static_assert(
    stores_prediction_arguments<kf::x_z_p_q_r_hh_f_tt_oo_ps<
        double, double, std::tuple<float>, model, propagation, measurement>>);
static_assert(not stores_update_arguments<kf::x_z_p_q_r_hh_f_tt_oo<
                  double, double, model, propagation, measurement>>);
static_assert(not stores_prediction_arguments<kf::x_z_p_q_r_hh_f_tt_oo<
                  double, double, model, propagation, measurement>>);
static_assert(
    sizeof(kalman<kf::x_z_p_q_r_hh_f_tt_oo<double, double, model, propagation,
                                           measurement>>) <
    sizeof(kalman<
           kf::x_z_p_q_r_hh_f_us_ps<double, double, std::tuple<>, std::tuple<>,
                                    model, propagation, measurement>>));
static_assert(
    stores_update_arguments<
        kf::x_z_p_q_r_hh_ff_tt_oo_us<double, double, std::tuple<int>, model,
                                     jacobian, propagation, measurement>>);
static_assert(
    not stores_prediction_arguments<
        kf::x_z_p_q_r_hh_ff_tt_oo_us<double, double, std::tuple<int>, model,
                                     jacobian, propagation, measurement>>);
static_assert(
    not stores_update_arguments<
        kf::x_z_p_q_r_hh_ff_tt_oo_ps<double, double, std::tuple<float>, model,
                                     jacobian, propagation, measurement>>);
static_assert(
    stores_prediction_arguments<
        kf::x_z_p_q_r_hh_ff_tt_oo_ps<double, double, std::tuple<float>, model,
                                     jacobian, propagation, measurement>>);
static_assert(not stores_update_arguments<kf::x_z_p_q_r_hh_ff_tt_oo<
                  double, double, model, jacobian, propagation, measurement>>);
static_assert(not stores_prediction_arguments<kf::x_z_p_q_r_hh_ff_tt_oo<
                  double, double, model, jacobian, propagation, measurement>>);
static_assert(
    sizeof(kalman<kf::x_z_p_q_r_hh_ff_tt_oo<double, double, model, jacobian,
                                            propagation, measurement>>) <
    sizeof(kalman<kf::x_z_p_q_r_hh_ff_us_ps<double, double, std::tuple<>,
                                            std::tuple<>, model, jacobian,
                                            propagation, measurement>>));
static_assert(
    stores_update_arguments<
        kf::x_z_p_qq_r_hh_f_tt_oo_us<double, double, std::tuple<int>, noise,
                                     model, propagation, measurement>>);
static_assert(
    not stores_prediction_arguments<
        kf::x_z_p_qq_r_hh_f_tt_oo_us<double, double, std::tuple<int>, noise,
                                     model, propagation, measurement>>);
static_assert(
    not stores_update_arguments<
        kf::x_z_p_qq_r_hh_f_tt_oo_ps<double, double, std::tuple<float>, noise,
                                     model, propagation, measurement>>);
static_assert(
    stores_prediction_arguments<
        kf::x_z_p_qq_r_hh_f_tt_oo_ps<double, double, std::tuple<float>, noise,
                                     model, propagation, measurement>>);
static_assert(not stores_update_arguments<kf::x_z_p_qq_r_hh_f_tt_oo<
                  double, double, noise, model, propagation, measurement>>);
static_assert(not stores_prediction_arguments<kf::x_z_p_qq_r_hh_f_tt_oo<
                  double, double, noise, model, propagation, measurement>>);
static_assert(
    sizeof(kalman<kf::x_z_p_qq_r_hh_f_tt_oo<double, double, noise, model,
                                            propagation, measurement>>) <
    sizeof(kalman<
           kf::x_z_p_qq_r_hh_f_us_ps<double, double, std::tuple<>, std::tuple<>,
                                     noise, model, propagation, measurement>>));
static_assert(stores_update_arguments<kf::x_z_p_qq_r_hh_ff_tt_oo_us<
                  double, double, std::tuple<int>, noise, model, jacobian,
                  propagation, measurement>>);
static_assert(not stores_prediction_arguments<kf::x_z_p_qq_r_hh_ff_tt_oo_us<
                  double, double, std::tuple<int>, noise, model, jacobian,
                  propagation, measurement>>);
static_assert(not stores_update_arguments<kf::x_z_p_qq_r_hh_ff_tt_oo_ps<
                  double, double, std::tuple<float>, noise, model, jacobian,
                  propagation, measurement>>);
static_assert(stores_prediction_arguments<kf::x_z_p_qq_r_hh_ff_tt_oo_ps<
                  double, double, std::tuple<float>, noise, model, jacobian,
                  propagation, measurement>>);
static_assert(
    not stores_update_arguments<kf::x_z_p_qq_r_hh_ff_tt_oo<
        double, double, noise, model, jacobian, propagation, measurement>>);
static_assert(
    not stores_prediction_arguments<kf::x_z_p_qq_r_hh_ff_tt_oo<
        double, double, noise, model, jacobian, propagation, measurement>>);
static_assert(
    sizeof(kalman<
           kf::x_z_p_qq_r_hh_ff_tt_oo<double, double, noise, model, jacobian,
                                      propagation, measurement>>) <
    sizeof(kalman<kf::x_z_p_qq_r_hh_ff_us_ps<
               double, double, std::tuple<>, std::tuple<>, noise, model,
               jacobian, propagation, measurement>>));
static_assert(
    stores_update_arguments<
        kf::x_z_u_p_q_r_h_f_g_us<double, double, double, std::tuple<int>>>);
static_assert(
    not stores_prediction_arguments<
        kf::x_z_u_p_q_r_h_f_g_us<double, double, double, std::tuple<int>>>);
static_assert(
    not stores_update_arguments<
        kf::x_z_u_p_q_r_h_f_g_ps<double, double, double, std::tuple<float>>>);
static_assert(
    stores_prediction_arguments<
        kf::x_z_u_p_q_r_h_f_g_ps<double, double, double, std::tuple<float>>>);
static_assert(
    not stores_update_arguments<kf::x_z_u_p_q_r_h_f_g<double, double, double>>);
static_assert(not stores_prediction_arguments<
              kf::x_z_u_p_q_r_h_f_g<double, double, double>>);
static_assert(sizeof(kalman<kf::x_z_u_p_q_r_h_f_g<double, double, double>>) <
              sizeof(kalman<kf::x_z_u_p_q_r_h_f_g_us_ps<
                         double, double, double, std::tuple<>, std::tuple<>>>));
static_assert(not stores_update_arguments<kf::x_z_u_p_qq_r_ff_gg<
                  double, double, double, noise, jacobian, control>>);
static_assert(not stores_prediction_arguments<kf::x_z_u_p_qq_r_ff_gg<
                  double, double, double, noise, jacobian, control>>);
static_assert(
    sizeof(kalman<kf::x_z_u_p_qq_r_ff_gg<double, double, double, noise,
                                         jacobian, control>>) <
    sizeof(kalman<
           kf::x_z_u_p_qq_r_ff_gg_ps<double, double, double, std::tuple<>,
                                     std::tuple<>, noise, jacobian, control>>));
static_assert(
    not stores_update_arguments<
        kf::x_z_p_q_r_hh_ff_oo<double, double, model, control, measurement>>);
static_assert(
    not stores_prediction_arguments<
        kf::x_z_p_q_r_hh_ff_oo<double, double, model, control, measurement>>);
static_assert(
    sizeof(kalman<kf::x_z_p_q_r_hh_ff_oo<double, double, model, control,
                                         measurement>>) <
    sizeof(kalman<kf::x_z_p_q_r_hh_ff_ps<double, double, std::tuple<>, model,
                                         control, measurement>>));
static_assert(stores_update_arguments<kf::x_z_p_q_r_hh_f_oo_us<
                  double, double, std::tuple<int>, model, measurement>>);
static_assert(not stores_prediction_arguments<kf::x_z_p_q_r_hh_f_oo_us<
                  double, double, std::tuple<int>, model, measurement>>);
static_assert(not stores_update_arguments<
              kf::x_z_p_q_r_hh_f_oo<double, double, model, measurement>>);
static_assert(not stores_prediction_arguments<
              kf::x_z_p_q_r_hh_f_oo<double, double, model, measurement>>);
static_assert(
    sizeof(kalman<kf::x_z_p_q_r_hh_f_oo<double, double, model, measurement>>) <
    sizeof(kalman<
           kf::x_z_p_q_r_hh_f_us_ps<double, double, std::tuple<>, std::tuple<>,
                                    model, propagation, measurement>>));
} // namespace
} // namespace fcarouge::test
