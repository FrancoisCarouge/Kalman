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
#include "fcarouge/linalg.hpp"

#include <cassert>
#include <cmath>

namespace fcarouge::sample {
namespace {
template <auto Size> using vector = column_vector<float, Size>;
template <auto Row, auto Column> using matrix = matrix<float, Row, Column>;
using state = fcarouge::state<vector<4>>;

//! @brief ArduPilot plane soaring.
//!
//! @copyright This example is transcribed from the ArduPilot Soaring Plane
//! copyright ArduPilot Dev Team.
//!
//! @see https://ardupilot.org/plane/docs/soaring.html
//! @see https://arxiv.org/abs/1802.08215
//!
//! @details The autonomous soaring functionality in ArduPilot allows the plane
//! to respond to rising air current in order to extend endurance and gain
//! altitude with minimal use of the motor. The full technical description is
//! available in S. Tabor, I. Guilliard, A. Kolobov. ArduSoar: an Open-Source
//! Thermalling Controller for Resource-Constrained Autopilots. International
//! Conference on Intelligent Robots and Systems (IROS), 2018.
//! Estimating the parameters of a Wharington et al thermal model state X: [W,
//! R, x, y] with the speed or strength W [m.s^-1] at the center of the thermal
//! of radius R [m] with the center distance x north of the sUAV and y east of
//! the sUAV.
//!
//! @example ekf_4x1x0_thermal_soaring.cpp
[[maybe_unused]] const auto sample{[] -> int {
  const float trigger_strength{0};
  const float thermal_radius{80};
  const float thermal_position_x{5};
  const float thermal_position_y{0};
  const float strength_covariance{0.0049F};
  const float radius_covariance{400};
  const float position_covariance{400};
  const float strength_noise{std::pow(0.001F, 2.F)};
  const float distance_noise{std::pow(0.03F, 2.F)};
  const float measure_noise{std::pow(0.45F, 2.F)};

  // 4x1 extended filter with additional parameter for prediction: driftX [m],
  // driftY [m]. Constant time step.
  kalman filter{
      // The state X:
      state{trigger_strength, thermal_radius, thermal_position_x,
            thermal_position_y},
      // The output Z:
      output<float>,
      // The estimate uncertainty P:
      estimate_uncertainty{{strength_covariance, 0.F, 0.F, 0.F},
                           {0.F, radius_covariance, 0.F, 0.F},
                           {0.F, 0.F, position_covariance, 0.F},
                           {0.F, 0.F, 0.F, position_covariance}},
      // The process uncertainty Q:
      process_uncertainty{{strength_noise, 0.F, 0.F, 0.F},
                          {0.F, distance_noise, 0.F, 0.F},
                          {0.F, 0.F, distance_noise, 0.F},
                          {0.F, 0.F, 0.F, distance_noise}},
      // The output uncertainty R:
      output_uncertainty{measure_noise},
      // No process dynamics: the state transition F = ∂f/∂X = I4
      // Default. The additional parameters for update.
      // See the ArduSoar paper for the equation for H = ∂h/∂X:
      output_model{[](const vector<4> &x, const float &position_x,
                      const float &position_y) -> matrix<1, 4> {
        const float expon{std::exp(-(std::pow(x[2] - position_x, 2.F) +
                                     std::pow(x[3] - position_y, 2.F)) /
                                   std::pow(x[1], 2.F))};
        const matrix<1, 4> h{
            expon,
            2 * x(0) *
                ((std::pow(x(2) - position_x, 2.F) +
                  std::pow(x(3) - position_y, 2.F)) /
                 std::pow(x(1), 3.F)) *
                expon,
            -2 * (x(0) * (x(2) - position_x) / std::pow(x(1), 2.F)) * expon,
            -2 * (x(0) * (x(3) - position_y) / std::pow(x(1), 2.F)) * expon};

        return h;
      }},
      transition{[](const vector<4> &x, const float &drift_x,
                    const float &drift_y) -> vector<4> {
        //! In production, make sure that x[1] stays positive, greater than 40.
        const vector<4> drifts{0.F, 0.F, drift_x, drift_y};
        return x + drifts;
      }},
      // Observation Z: [w] vertical air velocity w at the aircraft’s
      // position w.r.t. the thermal center [m.s^-1].
      observation{[](const vector<4> &x, const float &position_x,
                     const float &position_y) -> float {
        return x(0) * std::exp(-(std::pow(x[2] - position_x, 2.F) +
                                 std::pow(x[3] - position_y, 2.F)) /
                               std::pow(x[1], 2.F));
      }},
      update_types<float, float>,
      // The additional parameters for prediction.
      prediction_types<float, float>};

  struct data {
    float drift_x;
    float drift_y;
    float position_x;
    float position_y;
    float variometer;
  };

  // A hundred randomly generated data point.
  constexpr data measured[]{{.drift_x = 0.0756891F,
                             .drift_y = 0.749786F,
                             .position_x = 0.878827F,
                             .position_y = 0.806808F,
                             .variometer = 0.155487F},
                            {.drift_x = 0.506366F,
                             .drift_y = 0.261469F,
                             .position_x = 0.886986F,
                             .position_y = 0.332883F,
                             .variometer = 0.434406F},
                            {.drift_x = 0.249769F,
                             .drift_y = 0.242154F,
                             .position_x = 0.616454F,
                             .position_y = 0.672545F,
                             .variometer = 0.24927F},
                            {.drift_x = 0.358587F,
                             .drift_y = 0.556206F,
                             .position_x = 0.909985F,
                             .position_y = 0.370336F,
                             .variometer = 0.553264F},
                            {.drift_x = 0.370579F,
                             .drift_y = 0.368003F,
                             .position_x = 0.491917F,
                             .position_y = 0.635429F,
                             .variometer = 0.73594F},
                            {.drift_x = 0.82946F,
                             .drift_y = 0.0221123F,
                             .position_x = 0.461047F,
                             .position_y = 0.940697F,
                             .variometer = 0.987409F},
                            {.drift_x = 0.462132F,
                             .drift_y = 0.708865F,
                             .position_x = 0.941915F,
                             .position_y = 0.122432F,
                             .variometer = 0.911597F},
                            {.drift_x = 0.888334F,
                             .drift_y = 0.542419F,
                             .position_x = 0.773781F,
                             .position_y = 0.116075F,
                             .variometer = 0.917592F},
                            {.drift_x = 0.229376F,
                             .drift_y = 0.174244F,
                             .position_x = 0.972009F,
                             .position_y = 0.509611F,
                             .variometer = 0.37637F},
                            {.drift_x = 0.887738F,
                             .drift_y = 0.707866F,
                             .position_x = 0.90959F,
                             .position_y = 0.430274F,
                             .variometer = 0.242523F},
                            {.drift_x = 0.40713F,
                             .drift_y = 0.0696747F,
                             .position_x = 0.456659F,
                             .position_y = 0.979656F,
                             .variometer = 0.11167F},
                            {.drift_x = 0.77115F,
                             .drift_y = 0.183994F,
                             .position_x = 0.944587F,
                             .position_y = 0.467626F,
                             .variometer = 0.0219546F},
                            {.drift_x = 0.137442F,
                             .drift_y = 0.316077F,
                             .position_x = 0.660742F,
                             .position_y = 0.828009F,
                             .variometer = 0.852228F},
                            {.drift_x = 0.128113F,
                             .drift_y = 0.0757587F,
                             .position_x = 0.742959F,
                             .position_y = 0.360531F,
                             .variometer = 0.3932F},
                            {.drift_x = 0.161107F,
                             .drift_y = 0.709262F,
                             .position_x = 0.690847F,
                             .position_y = 0.161165F,
                             .variometer = 0.237205F},
                            {.drift_x = 0.664184F,
                             .drift_y = 0.658516F,
                             .position_x = 0.972067F,
                             .position_y = 0.465567F,
                             .variometer = 0.807259F},
                            {.drift_x = 0.669789F,
                             .drift_y = 0.236436F,
                             .position_x = 0.341701F,
                             .position_y = 0.430546F,
                             .variometer = 0.229097F},
                            {.drift_x = 0.159471F,
                             .drift_y = 0.122824F,
                             .position_x = 0.975034F,
                             .position_y = 0.833685F,
                             .variometer = 0.78011F},
                            {.drift_x = 0.284848F,
                             .drift_y = 0.917524F,
                             .position_x = 0.358084F,
                             .position_y = 0.82927F,
                             .variometer = 0.0983398F},
                            {.drift_x = 0.209027F,
                             .drift_y = 0.573124F,
                             .position_x = 0.428336F,
                             .position_y = 0.106116F,
                             .variometer = 0.17974F},
                            {.drift_x = 0.861987F,
                             .drift_y = 0.110099F,
                             .position_x = 0.0994602F,
                             .position_y = 0.208052F,
                             .variometer = 0.0545667F},
                            {.drift_x = 0.483002F,
                             .drift_y = 0.707016F,
                             .position_x = 0.189368F,
                             .position_y = 0.0626376F,
                             .variometer = 0.992816F},
                            {.drift_x = 0.588928F,
                             .drift_y = 0.644143F,
                             .position_x = 0.763512F,
                             .position_y = 0.444366F,
                             .variometer = 0.251652F},
                            {.drift_x = 0.419946F,
                             .drift_y = 0.338175F,
                             .position_x = 0.286543F,
                             .position_y = 0.97232F,
                             .variometer = 0.908061F},
                            {.drift_x = 0.0625373F,
                             .drift_y = 0.855109F,
                             .position_x = 0.763831F,
                             .position_y = 0.622934F,
                             .variometer = 0.364608F},
                            {.drift_x = 0.55833F,
                             .drift_y = 0.505803F,
                             .position_x = 0.600797F,
                             .position_y = 0.342724F,
                             .variometer = 0.735087F},
                            {.drift_x = 0.664873F,
                             .drift_y = 0.224638F,
                             .position_x = 0.385409F,
                             .position_y = 0.892807F,
                             .variometer = 0.695F},
                            {.drift_x = 0.255295F,
                             .drift_y = 0.0264766F,
                             .position_x = 0.229274F,
                             .position_y = 0.723291F,
                             .variometer = 0.552242F},
                            {.drift_x = 0.412129F,
                             .drift_y = 0.856404F,
                             .position_x = 0.395075F,
                             .position_y = 0.261842F,
                             .variometer = 0.947885F},
                            {.drift_x = 0.468212F,
                             .drift_y = 0.849367F,
                             .position_x = 0.00615251F,
                             .position_y = 0.842904F,
                             .variometer = 0.700869F},
                            {.drift_x = 0.311582F,
                             .drift_y = 0.293401F,
                             .position_x = 0.299637F,
                             .position_y = 0.567025F,
                             .variometer = 0.659598F},
                            {.drift_x = 0.695464F,
                             .drift_y = 0.941376F,
                             .position_x = 0.21219F,
                             .position_y = 0.27813F,
                             .variometer = 0.289406F},
                            {.drift_x = 0.000397467F,
                             .drift_y = 0.301337F,
                             .position_x = 0.71608F,
                             .position_y = 0.296278F,
                             .variometer = 0.718923F},
                            {.drift_x = 0.36314F,
                             .drift_y = 0.263077F,
                             .position_x = 0.193163F,
                             .position_y = 0.295399F,
                             .variometer = 0.0523569F},
                            {.drift_x = 0.128381F,
                             .drift_y = 0.572157F,
                             .position_x = 0.971297F,
                             .position_y = 0.516492F,
                             .variometer = 0.921166F},
                            {.drift_x = 0.596215F,
                             .drift_y = 0.909239F,
                             .position_x = 0.133898F,
                             .position_y = 0.506903F,
                             .variometer = 0.0335569F},
                            {.drift_x = 0.444556F,
                             .drift_y = 0.997721F,
                             .position_x = 0.348369F,
                             .position_y = 0.644847F,
                             .variometer = 0.80885F},
                            {.drift_x = 0.891465F,
                             .drift_y = 0.0797467F,
                             .position_x = 0.85753F,
                             .position_y = 0.369457F,
                             .variometer = 0.418543F},
                            {.drift_x = 0.861948F,
                             .drift_y = 0.520583F,
                             .position_x = 0.900797F,
                             .position_y = 0.153884F,
                             .variometer = 0.080031F},
                            {.drift_x = 0.169696F,
                             .drift_y = 0.981169F,
                             .position_x = 0.406729F,
                             .position_y = 0.292696F,
                             .variometer = 0.831505F},
                            {.drift_x = 0.172591F,
                             .drift_y = 0.349291F,
                             .position_x = 0.782213F,
                             .position_y = 0.534652F,
                             .variometer = 0.214628F},
                            {.drift_x = 0.875081F,
                             .drift_y = 0.746097F,
                             .position_x = 0.0806311F,
                             .position_y = 0.15685F,
                             .variometer = 0.357471F},
                            {.drift_x = 0.519389F,
                             .drift_y = 0.007303F,
                             .position_x = 0.18117F,
                             .position_y = 0.370993F,
                             .variometer = 0.427305F},
                            {.drift_x = 0.961372F,
                             .drift_y = 0.218945F,
                             .position_x = 0.486608F,
                             .position_y = 0.618755F,
                             .variometer = 0.168813F},
                            {.drift_x = 0.537862F,
                             .drift_y = 0.451312F,
                             .position_x = 0.384422F,
                             .position_y = 0.540216F,
                             .variometer = 0.525636F},
                            {.drift_x = 0.494387F,
                             .drift_y = 0.162124F,
                             .position_x = 0.0136825F,
                             .position_y = 0.127037F,
                             .variometer = 0.803511F},
                            {.drift_x = 0.409087F,
                             .drift_y = 0.991167F,
                             .position_x = 0.276877F,
                             .position_y = 0.188698F,
                             .variometer = 0.155701F},
                            {.drift_x = 0.851474F,
                             .drift_y = 0.54778F,
                             .position_x = 0.133586F,
                             .position_y = 0.37391F,
                             .variometer = 0.137362F},
                            {.drift_x = 0.0148137F,
                             .drift_y = 0.97396F,
                             .position_x = 0.945259F,
                             .position_y = 0.297432F,
                             .variometer = 0.260494F},
                            {.drift_x = 0.906864F,
                             .drift_y = 0.13484F,
                             .position_x = 0.214258F,
                             .position_y = 0.924681F,
                             .variometer = 0.618572F},
                            {.drift_x = 0.141742F,
                             .drift_y = 0.563986F,
                             .position_x = 0.502602F,
                             .position_y = 0.416297F,
                             .variometer = 0.97038F},
                            {.drift_x = 0.698555F,
                             .drift_y = 0.406929F,
                             .position_x = 0.558199F,
                             .position_y = 0.875364F,
                             .variometer = 0.736008F},
                            {.drift_x = 0.175105F,
                             .drift_y = 0.270328F,
                             .position_x = 0.332957F,
                             .position_y = 0.145101F,
                             .variometer = 0.765857F},
                            {.drift_x = 0.68083F,
                             .drift_y = 0.125673F,
                             .position_x = 0.922594F,
                             .position_y = 0.831683F,
                             .variometer = 0.457214F},
                            {.drift_x = 0.520728F,
                             .drift_y = 0.26214F,
                             .position_x = 0.458674F,
                             .position_y = 0.306454F,
                             .variometer = 0.783164F},
                            {.drift_x = 0.780442F,
                             .drift_y = 0.472245F,
                             .position_x = 0.125185F,
                             .position_y = 0.460146F,
                             .variometer = 0.0847598F},
                            {.drift_x = 0.360083F,
                             .drift_y = 0.0686402F,
                             .position_x = 0.328997F,
                             .position_y = 0.799852F,
                             .variometer = 0.818809F},
                            {.drift_x = 0.71546F,
                             .drift_y = 0.717884F,
                             .position_x = 0.253842F,
                             .position_y = 0.812915F,
                             .variometer = 0.0141433F},
                            {.drift_x = 0.441185F,
                             .drift_y = 0.171204F,
                             .position_x = 0.0432966F,
                             .position_y = 0.739241F,
                             .variometer = 0.448679F},
                            {.drift_x = 0.399117F,
                             .drift_y = 0.148854F,
                             .position_x = 0.743042F,
                             .position_y = 0.0230124F,
                             .variometer = 0.378786F},
                            {.drift_x = 0.841239F,
                             .drift_y = 0.292533F,
                             .position_x = 0.391296F,
                             .position_y = 0.734326F,
                             .variometer = 0.0597166F},
                            {.drift_x = 0.350847F,
                             .drift_y = 0.519149F,
                             .position_x = 0.808508F,
                             .position_y = 0.113644F,
                             .variometer = 0.673261F},
                            {.drift_x = 0.229909F,
                             .drift_y = 0.814871F,
                             .position_x = 0.118688F,
                             .position_y = 0.612729F,
                             .variometer = 0.354682F},
                            {.drift_x = 0.734755F,
                             .drift_y = 0.675693F,
                             .position_x = 0.646155F,
                             .position_y = 0.0296504F,
                             .variometer = 0.405621F},
                            {.drift_x = 0.121731F,
                             .drift_y = 0.231111F,
                             .position_x = 0.47879F,
                             .position_y = 0.733299F,
                             .variometer = 0.270893F},
                            {.drift_x = 0.732981F,
                             .drift_y = 0.813999F,
                             .position_x = 0.597652F,
                             .position_y = 0.455436F,
                             .variometer = 0.691262F},
                            {.drift_x = 0.10297F,
                             .drift_y = 0.534613F,
                             .position_x = 0.553605F,
                             .position_y = 0.777385F,
                             .variometer = 0.553588F},
                            {.drift_x = 0.441429F,
                             .drift_y = 0.974205F,
                             .position_x = 0.120671F,
                             .position_y = 0.279931F,
                             .variometer = 0.624484F},
                            {.drift_x = 0.531836F,
                             .drift_y = 0.697762F,
                             .position_x = 0.274009F,
                             .position_y = 0.827927F,
                             .variometer = 0.741129F},
                            {.drift_x = 0.745307F,
                             .drift_y = 0.085542F,
                             .position_x = 0.473629F,
                             .position_y = 0.286912F,
                             .variometer = 0.175756F},
                            {.drift_x = 0.758466F,
                             .drift_y = 0.268705F,
                             .position_x = 0.108006F,
                             .position_y = 0.291002F,
                             .variometer = 0.559732F},
                            {.drift_x = 0.632262F,
                             .drift_y = 0.733193F,
                             .position_x = 0.919653F,
                             .position_y = 0.165692F,
                             .variometer = 0.84716F},
                            {.drift_x = 0.0107621F,
                             .drift_y = 0.694084F,
                             .position_x = 0.35781F,
                             .position_y = 0.793076F,
                             .variometer = 0.0818898F},
                            {.drift_x = 0.17388F,
                             .drift_y = 0.333606F,
                             .position_x = 0.867638F,
                             .position_y = 0.969285F,
                             .variometer = 0.887633F},
                            {.drift_x = 0.255376F,
                             .drift_y = 0.180532F,
                             .position_x = 0.737631F,
                             .position_y = 0.869954F,
                             .variometer = 0.875926F},
                            {.drift_x = 0.525821F,
                             .drift_y = 0.882517F,
                             .position_x = 0.224126F,
                             .position_y = 0.906093F,
                             .variometer = 0.557676F},
                            {.drift_x = 0.516693F,
                             .drift_y = 0.986614F,
                             .position_x = 0.644313F,
                             .position_y = 0.00903489F,
                             .variometer = 0.207868F},
                            {.drift_x = 0.00175451F,
                             .drift_y = 0.49772F,
                             .position_x = 0.436713F,
                             .position_y = 0.0418148F,
                             .variometer = 0.63547F},
                            {.drift_x = 0.559954F,
                             .drift_y = 0.192099F,
                             .position_x = 0.0787102F,
                             .position_y = 0.976933F,
                             .variometer = 0.552542F},
                            {.drift_x = 0.983202F,
                             .drift_y = 0.165426F,
                             .position_x = 0.136735F,
                             .position_y = 0.467933F,
                             .variometer = 0.626612F},
                            {.drift_x = 0.520497F,
                             .drift_y = 0.593702F,
                             .position_x = 0.0155549F,
                             .position_y = 0.791301F,
                             .variometer = 0.635127F},
                            {.drift_x = 0.934924F,
                             .drift_y = 0.0663795F,
                             .position_x = 0.513404F,
                             .position_y = 0.791586F,
                             .variometer = 0.68594F},
                            {.drift_x = 0.977299F,
                             .drift_y = 0.682359F,
                             .position_x = 0.0689664F,
                             .position_y = 0.769369F,
                             .variometer = 0.169862F},
                            {.drift_x = 0.681586F,
                             .drift_y = 0.900795F,
                             .position_x = 0.312534F,
                             .position_y = 0.854568F,
                             .variometer = 0.113097F},
                            {.drift_x = 0.0783791F,
                             .drift_y = 0.340692F,
                             .position_x = 0.23686F,
                             .position_y = 0.5932F,
                             .variometer = 0.38193F},
                            {.drift_x = 0.430041F,
                             .drift_y = 0.401364F,
                             .position_x = 0.88266F,
                             .position_y = 0.226286F,
                             .variometer = 0.514185F},
                            {.drift_x = 0.422123F,
                             .drift_y = 0.713778F,
                             .position_x = 0.813105F,
                             .position_y = 0.960577F,
                             .variometer = 0.794308F},
                            {.drift_x = 0.0531423F,
                             .drift_y = 0.930818F,
                             .position_x = 0.913336F,
                             .position_y = 0.382305F,
                             .variometer = 0.372521F},
                            {.drift_x = 0.91698F,
                             .drift_y = 0.128078F,
                             .position_x = 0.901849F,
                             .position_y = 0.0860355F,
                             .variometer = 0.432365F},
                            {.drift_x = 0.749259F,
                             .drift_y = 0.198112F,
                             .position_x = 0.538301F,
                             .position_y = 0.739992F,
                             .variometer = 0.909026F},
                            {.drift_x = 0.903781F,
                             .drift_y = 0.206122F,
                             .position_x = 0.743227F,
                             .position_y = 0.700662F,
                             .variometer = 0.784729F},
                            {.drift_x = 0.914658F,
                             .drift_y = 0.625943F,
                             .position_x = 0.697374F,
                             .position_y = 0.333459F,
                             .variometer = 0.213769F},
                            {.drift_x = 0.313091F,
                             .drift_y = 0.0485961F,
                             .position_x = 0.625018F,
                             .position_y = 0.916347F,
                             .variometer = 0.363119F},
                            {.drift_x = 0.455916F,
                             .drift_y = 0.982769F,
                             .position_x = 0.245987F,
                             .position_y = 0.555492F,
                             .variometer = 0.938798F},
                            {.drift_x = 0.0737146F,
                             .drift_y = 0.324519F,
                             .position_x = 0.325405F,
                             .position_y = 0.677491F,
                             .variometer = 0.148078F},
                            {.drift_x = 0.918677F,
                             .drift_y = 0.537612F,
                             .position_x = 0.917458F,
                             .position_y = 0.611973F,
                             .variometer = 0.965844F},
                            {.drift_x = 0.832977F,
                             .drift_y = 0.466222F,
                             .position_x = 0.528761F,
                             .position_y = 0.348765F,
                             .variometer = 0.472975F},
                            {.drift_x = 0.784042F,
                             .drift_y = 0.866144F,
                             .position_x = 0.00524178F,
                             .position_y = 0.217837F,
                             .variometer = 0.145246F},
                            {.drift_x = 0.308576F,
                             .drift_y = 0.993283F,
                             .position_x = 0.0244056F,
                             .position_y = 0.543786F,
                             .variometer = 0.575841F},
                            {.drift_x = 0.285113F,
                             .drift_y = 0.12198F,
                             .position_x = 0.74075F,
                             .position_y = 0.834888F,
                             .variometer = 0.561457F},
                            {.drift_x = 0.635992F,
                             .drift_y = 0.590228F,
                             .position_x = 0.629378F,
                             .position_y = 0.112457F,
                             .variometer = 0.78253F}};

  for (const auto &measure : measured) {
    filter.predict(measure.drift_x, measure.drift_y);
    filter.update(measure.position_x, measure.position_y, measure.variometer);
  }

  assert(std::abs(1 - (filter.x()[0] / 0.347191F)) < 0.0001F &&
         std::abs(1 - (filter.x()[1] / 91.8926F)) < 0.0001F &&
         std::abs(1 - (filter.x()[2] / 22.9656F)) < 0.0001F &&
         std::abs(1 - (filter.x()[3] / 20.6146F)) < 0.0001F &&
         "The estimated states expected to meet ArduPilot soaring plane "
         "implementation at 0.01% accuracy.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
