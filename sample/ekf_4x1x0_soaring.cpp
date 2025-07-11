/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.3
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
//! @example ekf_4x1x0_soaring.cpp
[[maybe_unused]] auto sample{[] {
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
  constexpr data measured[]{
      {0.0756891F, 0.749786F, 0.878827F, 0.806808F, 0.155487F},
      {0.506366F, 0.261469F, 0.886986F, 0.332883F, 0.434406F},
      {0.249769F, 0.242154F, 0.616454F, 0.672545F, 0.24927F},
      {0.358587F, 0.556206F, 0.909985F, 0.370336F, 0.553264F},
      {0.370579F, 0.368003F, 0.491917F, 0.635429F, 0.73594F},
      {0.82946F, 0.0221123F, 0.461047F, 0.940697F, 0.987409F},
      {0.462132F, 0.708865F, 0.941915F, 0.122432F, 0.911597F},
      {0.888334F, 0.542419F, 0.773781F, 0.116075F, 0.917592F},
      {0.229376F, 0.174244F, 0.972009F, 0.509611F, 0.37637F},
      {0.887738F, 0.707866F, 0.90959F, 0.430274F, 0.242523F},
      {0.40713F, 0.0696747F, 0.456659F, 0.979656F, 0.11167F},
      {0.77115F, 0.183994F, 0.944587F, 0.467626F, 0.0219546F},
      {0.137442F, 0.316077F, 0.660742F, 0.828009F, 0.852228F},
      {0.128113F, 0.0757587F, 0.742959F, 0.360531F, 0.3932F},
      {0.161107F, 0.709262F, 0.690847F, 0.161165F, 0.237205F},
      {0.664184F, 0.658516F, 0.972067F, 0.465567F, 0.807259F},
      {0.669789F, 0.236436F, 0.341701F, 0.430546F, 0.229097F},
      {0.159471F, 0.122824F, 0.975034F, 0.833685F, 0.78011F},
      {0.284848F, 0.917524F, 0.358084F, 0.82927F, 0.0983398F},
      {0.209027F, 0.573124F, 0.428336F, 0.106116F, 0.17974F},
      {0.861987F, 0.110099F, 0.0994602F, 0.208052F, 0.0545667F},
      {0.483002F, 0.707016F, 0.189368F, 0.0626376F, 0.992816F},
      {0.588928F, 0.644143F, 0.763512F, 0.444366F, 0.251652F},
      {0.419946F, 0.338175F, 0.286543F, 0.97232F, 0.908061F},
      {0.0625373F, 0.855109F, 0.763831F, 0.622934F, 0.364608F},
      {0.55833F, 0.505803F, 0.600797F, 0.342724F, 0.735087F},
      {0.664873F, 0.224638F, 0.385409F, 0.892807F, 0.695F},
      {0.255295F, 0.0264766F, 0.229274F, 0.723291F, 0.552242F},
      {0.412129F, 0.856404F, 0.395075F, 0.261842F, 0.947885F},
      {0.468212F, 0.849367F, 0.00615251F, 0.842904F, 0.700869F},
      {0.311582F, 0.293401F, 0.299637F, 0.567025F, 0.659598F},
      {0.695464F, 0.941376F, 0.21219F, 0.27813F, 0.289406F},
      {0.000397467F, 0.301337F, 0.71608F, 0.296278F, 0.718923F},
      {0.36314F, 0.263077F, 0.193163F, 0.295399F, 0.0523569F},
      {0.128381F, 0.572157F, 0.971297F, 0.516492F, 0.921166F},
      {0.596215F, 0.909239F, 0.133898F, 0.506903F, 0.0335569F},
      {0.444556F, 0.997721F, 0.348369F, 0.644847F, 0.80885F},
      {0.891465F, 0.0797467F, 0.85753F, 0.369457F, 0.418543F},
      {0.861948F, 0.520583F, 0.900797F, 0.153884F, 0.080031F},
      {0.169696F, 0.981169F, 0.406729F, 0.292696F, 0.831505F},
      {0.172591F, 0.349291F, 0.782213F, 0.534652F, 0.214628F},
      {0.875081F, 0.746097F, 0.0806311F, 0.15685F, 0.357471F},
      {0.519389F, 0.007303F, 0.18117F, 0.370993F, 0.427305F},
      {0.961372F, 0.218945F, 0.486608F, 0.618755F, 0.168813F},
      {0.537862F, 0.451312F, 0.384422F, 0.540216F, 0.525636F},
      {0.494387F, 0.162124F, 0.0136825F, 0.127037F, 0.803511F},
      {0.409087F, 0.991167F, 0.276877F, 0.188698F, 0.155701F},
      {0.851474F, 0.54778F, 0.133586F, 0.37391F, 0.137362F},
      {0.0148137F, 0.97396F, 0.945259F, 0.297432F, 0.260494F},
      {0.906864F, 0.13484F, 0.214258F, 0.924681F, 0.618572F},
      {0.141742F, 0.563986F, 0.502602F, 0.416297F, 0.97038F},
      {0.698555F, 0.406929F, 0.558199F, 0.875364F, 0.736008F},
      {0.175105F, 0.270328F, 0.332957F, 0.145101F, 0.765857F},
      {0.68083F, 0.125673F, 0.922594F, 0.831683F, 0.457214F},
      {0.520728F, 0.26214F, 0.458674F, 0.306454F, 0.783164F},
      {0.780442F, 0.472245F, 0.125185F, 0.460146F, 0.0847598F},
      {0.360083F, 0.0686402F, 0.328997F, 0.799852F, 0.818809F},
      {0.71546F, 0.717884F, 0.253842F, 0.812915F, 0.0141433F},
      {0.441185F, 0.171204F, 0.0432966F, 0.739241F, 0.448679F},
      {0.399117F, 0.148854F, 0.743042F, 0.0230124F, 0.378786F},
      {0.841239F, 0.292533F, 0.391296F, 0.734326F, 0.0597166F},
      {0.350847F, 0.519149F, 0.808508F, 0.113644F, 0.673261F},
      {0.229909F, 0.814871F, 0.118688F, 0.612729F, 0.354682F},
      {0.734755F, 0.675693F, 0.646155F, 0.0296504F, 0.405621F},
      {0.121731F, 0.231111F, 0.47879F, 0.733299F, 0.270893F},
      {0.732981F, 0.813999F, 0.597652F, 0.455436F, 0.691262F},
      {0.10297F, 0.534613F, 0.553605F, 0.777385F, 0.553588F},
      {0.441429F, 0.974205F, 0.120671F, 0.279931F, 0.624484F},
      {0.531836F, 0.697762F, 0.274009F, 0.827927F, 0.741129F},
      {0.745307F, 0.085542F, 0.473629F, 0.286912F, 0.175756F},
      {0.758466F, 0.268705F, 0.108006F, 0.291002F, 0.559732F},
      {0.632262F, 0.733193F, 0.919653F, 0.165692F, 0.84716F},
      {0.0107621F, 0.694084F, 0.35781F, 0.793076F, 0.0818898F},
      {0.17388F, 0.333606F, 0.867638F, 0.969285F, 0.887633F},
      {0.255376F, 0.180532F, 0.737631F, 0.869954F, 0.875926F},
      {0.525821F, 0.882517F, 0.224126F, 0.906093F, 0.557676F},
      {0.516693F, 0.986614F, 0.644313F, 0.00903489F, 0.207868F},
      {0.00175451F, 0.49772F, 0.436713F, 0.0418148F, 0.63547F},
      {0.559954F, 0.192099F, 0.0787102F, 0.976933F, 0.552542F},
      {0.983202F, 0.165426F, 0.136735F, 0.467933F, 0.626612F},
      {0.520497F, 0.593702F, 0.0155549F, 0.791301F, 0.635127F},
      {0.934924F, 0.0663795F, 0.513404F, 0.791586F, 0.68594F},
      {0.977299F, 0.682359F, 0.0689664F, 0.769369F, 0.169862F},
      {0.681586F, 0.900795F, 0.312534F, 0.854568F, 0.113097F},
      {0.0783791F, 0.340692F, 0.23686F, 0.5932F, 0.38193F},
      {0.430041F, 0.401364F, 0.88266F, 0.226286F, 0.514185F},
      {0.422123F, 0.713778F, 0.813105F, 0.960577F, 0.794308F},
      {0.0531423F, 0.930818F, 0.913336F, 0.382305F, 0.372521F},
      {0.91698F, 0.128078F, 0.901849F, 0.0860355F, 0.432365F},
      {0.749259F, 0.198112F, 0.538301F, 0.739992F, 0.909026F},
      {0.903781F, 0.206122F, 0.743227F, 0.700662F, 0.784729F},
      {0.914658F, 0.625943F, 0.697374F, 0.333459F, 0.213769F},
      {0.313091F, 0.0485961F, 0.625018F, 0.916347F, 0.363119F},
      {0.455916F, 0.982769F, 0.245987F, 0.555492F, 0.938798F},
      {0.0737146F, 0.324519F, 0.325405F, 0.677491F, 0.148078F},
      {0.918677F, 0.537612F, 0.917458F, 0.611973F, 0.965844F},
      {0.832977F, 0.466222F, 0.528761F, 0.348765F, 0.472975F},
      {0.784042F, 0.866144F, 0.00524178F, 0.217837F, 0.145246F},
      {0.308576F, 0.993283F, 0.0244056F, 0.543786F, 0.575841F},
      {0.285113F, 0.12198F, 0.74075F, 0.834888F, 0.561457F},
      {0.635992F, 0.590228F, 0.629378F, 0.112457F, 0.78253F}};

  for (const auto &measure : measured) {
    filter.predict(measure.drift_x, measure.drift_y);
    filter.update(measure.position_x, measure.position_y, measure.variometer);
  }

  assert(std::abs(1 - filter.x()[0] / 0.347191F) < 0.0001F &&
         std::abs(1 - filter.x()[1] / 91.8926F) < 0.0001F &&
         std::abs(1 - filter.x()[2] / 22.9656F) < 0.0001F &&
         std::abs(1 - filter.x()[3] / 20.6146F) < 0.0001F &&
         "The estimated states expected to meet ArduPilot soaring plane "
         "implementation at 0.01% accuracy.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
