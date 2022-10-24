#include "fcarouge/eigen/kalman.hpp"

#include <Eigen/Eigen>

#include <cassert>
#include <cmath>

namespace fcarouge::eigen::sample {
namespace {

template <typename Type, auto Size> using vector = Eigen::Vector<Type, Size>;

//! @brief ArduPilot Plane Soaring
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
//!
//! @todo Add a data set and assert for correctness of results.
[[maybe_unused]] auto ekf_4x1x0_soaring{[] {
  // 4x1 extended filter with additional parameter for prediction: driftX [m],
  // driftY [m]. Constant time step.
  using kalman = kalman<vector<float, 4>, float, void, std::tuple<float, float>,
                        std::tuple<float, float>>;

  kalman filter;

  // Initialization
  const float trigger_strength{0};
  const float thermal_radius{80};
  const float thermal_position_x{5};
  const float thermal_position_y{0};
  filter.x(trigger_strength, thermal_radius, thermal_position_x,
           thermal_position_y);

  const float strength_covariance{0.0049f};
  const float radius_covariance{400};
  const float position_covariance{400};
  filter.p(kalman::estimate_uncertainty{{strength_covariance, 0, 0, 0},
                                        {0, radius_covariance, 0, 0},
                                        {0, 0, position_covariance, 0},
                                        {0, 0, 0, position_covariance}});

  // No process dynamics: F = ∂f/∂X = I4 Default.

  filter.transition([](const kalman::state &x, const float &drift_x,
                       const float &drift_y) -> kalman::state {
    //! @todo Could make sure that x[1] stays positive, greater than 40.
    const kalman::state drifts{0, 0, drift_x, drift_y};
    return x + drifts;
  });

  const float strength_noise{std::pow(0.001f, 2.f)};
  const float distance_noise{std::pow(0.03f, 2.f)};
  filter.q(kalman::process_uncertainty{{strength_noise, 0, 0, 0},
                                       {0, distance_noise, 0, 0},
                                       {0, 0, distance_noise, 0},
                                       {0, 0, 0, distance_noise}});

  const float measure_noise{std::pow(0.45f, 2.f)};
  filter.r(kalman::output_uncertainty{measure_noise});

  // Observation Z: [w] vertical air velocity w at the aircraft’s
  // position w.r.t. the thermal center [m.s^-1].
  filter.observation([](const kalman::state &x, const float &position_x,
                        const float &position_y) -> kalman::output {
    return kalman::output{x(0) * std::exp(-(std::pow(x[2] - position_x, 2.f) +
                                            std::pow(x[3] - position_y, 2.f)) /
                                          std::pow(x[1], 2.f))};
  });

  // See the ArduSoar paper for the equation for H = ∂h/∂X:
  filter.h([](const kalman::state &x, const float &position_x,
              const float &position_y) -> kalman::output_model {
    const float expon{std::exp(
        -(std::pow(x[2] - position_x, 2.f) + std::pow(x[3] - position_y, 2.f)) /
        std::pow(x[1], 2.f))};

    const kalman::output_model h{
        expon,
        2 * x(0) *
            ((std::pow(x(2) - position_x, 2.f) +
              std::pow(x(3) - position_y, 2.f)) /
             std::pow(x(1), 3.f)) *
            expon,
        -2 * (x(0) * (x(2) - position_x) / std::pow(x(1), 2.f)) * expon,
        -2 * (x(0) * (x(3) - position_y) / std::pow(x(1), 2.f)) * expon};

    return h;
  });

  struct data {
    float drift_x;
    float drift_y;
    float position_x;
    float position_y;
    float variometer;
  };

  // A hundred randomly generated data point.
  constexpr data measured[]{
      {0.0756891f, 0.749786f, 0.878827f, 0.806808f, 0.155487f},
      {0.506366f, 0.261469f, 0.886986f, 0.332883f, 0.434406f},
      {0.249769f, 0.242154f, 0.616454f, 0.672545f, 0.24927f},
      {0.358587f, 0.556206f, 0.909985f, 0.370336f, 0.553264f},
      {0.370579f, 0.368003f, 0.491917f, 0.635429f, 0.73594f},
      {0.82946f, 0.0221123f, 0.461047f, 0.940697f, 0.987409f},
      {0.462132f, 0.708865f, 0.941915f, 0.122432f, 0.911597f},
      {0.888334f, 0.542419f, 0.773781f, 0.116075f, 0.917592f},
      {0.229376f, 0.174244f, 0.972009f, 0.509611f, 0.37637f},
      {0.887738f, 0.707866f, 0.90959f, 0.430274f, 0.242523f},
      {0.40713f, 0.0696747f, 0.456659f, 0.979656f, 0.11167f},
      {0.77115f, 0.183994f, 0.944587f, 0.467626f, 0.0219546f},
      {0.137442f, 0.316077f, 0.660742f, 0.828009f, 0.852228f},
      {0.128113f, 0.0757587f, 0.742959f, 0.360531f, 0.3932f},
      {0.161107f, 0.709262f, 0.690847f, 0.161165f, 0.237205f},
      {0.664184f, 0.658516f, 0.972067f, 0.465567f, 0.807259f},
      {0.669789f, 0.236436f, 0.341701f, 0.430546f, 0.229097f},
      {0.159471f, 0.122824f, 0.975034f, 0.833685f, 0.78011f},
      {0.284848f, 0.917524f, 0.358084f, 0.82927f, 0.0983398f},
      {0.209027f, 0.573124f, 0.428336f, 0.106116f, 0.17974f},
      {0.861987f, 0.110099f, 0.0994602f, 0.208052f, 0.0545667f},
      {0.483002f, 0.707016f, 0.189368f, 0.0626376f, 0.992816f},
      {0.588928f, 0.644143f, 0.763512f, 0.444366f, 0.251652f},
      {0.419946f, 0.338175f, 0.286543f, 0.97232f, 0.908061f},
      {0.0625373f, 0.855109f, 0.763831f, 0.622934f, 0.364608f},
      {0.55833f, 0.505803f, 0.600797f, 0.342724f, 0.735087f},
      {0.664873f, 0.224638f, 0.385409f, 0.892807f, 0.695f},
      {0.255295f, 0.0264766f, 0.229274f, 0.723291f, 0.552242f},
      {0.412129f, 0.856404f, 0.395075f, 0.261842f, 0.947885f},
      {0.468212f, 0.849367f, 0.00615251f, 0.842904f, 0.700869f},
      {0.311582f, 0.293401f, 0.299637f, 0.567025f, 0.659598f},
      {0.695464f, 0.941376f, 0.21219f, 0.27813f, 0.289406f},
      {0.000397467f, 0.301337f, 0.71608f, 0.296278f, 0.718923f},
      {0.36314f, 0.263077f, 0.193163f, 0.295399f, 0.0523569f},
      {0.128381f, 0.572157f, 0.971297f, 0.516492f, 0.921166f},
      {0.596215f, 0.909239f, 0.133898f, 0.506903f, 0.0335569f},
      {0.444556f, 0.997721f, 0.348369f, 0.644847f, 0.80885f},
      {0.891465f, 0.0797467f, 0.85753f, 0.369457f, 0.418543f},
      {0.861948f, 0.520583f, 0.900797f, 0.153884f, 0.080031f},
      {0.169696f, 0.981169f, 0.406729f, 0.292696f, 0.831505f},
      {0.172591f, 0.349291f, 0.782213f, 0.534652f, 0.214628f},
      {0.875081f, 0.746097f, 0.0806311f, 0.15685f, 0.357471f},
      {0.519389f, 0.007303f, 0.18117f, 0.370993f, 0.427305f},
      {0.961372f, 0.218945f, 0.486608f, 0.618755f, 0.168813f},
      {0.537862f, 0.451312f, 0.384422f, 0.540216f, 0.525636f},
      {0.494387f, 0.162124f, 0.0136825f, 0.127037f, 0.803511f},
      {0.409087f, 0.991167f, 0.276877f, 0.188698f, 0.155701f},
      {0.851474f, 0.54778f, 0.133586f, 0.37391f, 0.137362f},
      {0.0148137f, 0.97396f, 0.945259f, 0.297432f, 0.260494f},
      {0.906864f, 0.13484f, 0.214258f, 0.924681f, 0.618572f},
      {0.141742f, 0.563986f, 0.502602f, 0.416297f, 0.97038f},
      {0.698555f, 0.406929f, 0.558199f, 0.875364f, 0.736008f},
      {0.175105f, 0.270328f, 0.332957f, 0.145101f, 0.765857f},
      {0.68083f, 0.125673f, 0.922594f, 0.831683f, 0.457214f},
      {0.520728f, 0.26214f, 0.458674f, 0.306454f, 0.783164f},
      {0.780442f, 0.472245f, 0.125185f, 0.460146f, 0.0847598f},
      {0.360083f, 0.0686402f, 0.328997f, 0.799852f, 0.818809f},
      {0.71546f, 0.717884f, 0.253842f, 0.812915f, 0.0141433f},
      {0.441185f, 0.171204f, 0.0432966f, 0.739241f, 0.448679f},
      {0.399117f, 0.148854f, 0.743042f, 0.0230124f, 0.378786f},
      {0.841239f, 0.292533f, 0.391296f, 0.734326f, 0.0597166f},
      {0.350847f, 0.519149f, 0.808508f, 0.113644f, 0.673261f},
      {0.229909f, 0.814871f, 0.118688f, 0.612729f, 0.354682f},
      {0.734755f, 0.675693f, 0.646155f, 0.0296504f, 0.405621f},
      {0.121731f, 0.231111f, 0.47879f, 0.733299f, 0.270893f},
      {0.732981f, 0.813999f, 0.597652f, 0.455436f, 0.691262f},
      {0.10297f, 0.534613f, 0.553605f, 0.777385f, 0.553588f},
      {0.441429f, 0.974205f, 0.120671f, 0.279931f, 0.624484f},
      {0.531836f, 0.697762f, 0.274009f, 0.827927f, 0.741129f},
      {0.745307f, 0.085542f, 0.473629f, 0.286912f, 0.175756f},
      {0.758466f, 0.268705f, 0.108006f, 0.291002f, 0.559732f},
      {0.632262f, 0.733193f, 0.919653f, 0.165692f, 0.84716f},
      {0.0107621f, 0.694084f, 0.35781f, 0.793076f, 0.0818898f},
      {0.17388f, 0.333606f, 0.867638f, 0.969285f, 0.887633f},
      {0.255376f, 0.180532f, 0.737631f, 0.869954f, 0.875926f},
      {0.525821f, 0.882517f, 0.224126f, 0.906093f, 0.557676f},
      {0.516693f, 0.986614f, 0.644313f, 0.00903489f, 0.207868f},
      {0.00175451f, 0.49772f, 0.436713f, 0.0418148f, 0.63547f},
      {0.559954f, 0.192099f, 0.0787102f, 0.976933f, 0.552542f},
      {0.983202f, 0.165426f, 0.136735f, 0.467933f, 0.626612f},
      {0.520497f, 0.593702f, 0.0155549f, 0.791301f, 0.635127f},
      {0.934924f, 0.0663795f, 0.513404f, 0.791586f, 0.68594f},
      {0.977299f, 0.682359f, 0.0689664f, 0.769369f, 0.169862f},
      {0.681586f, 0.900795f, 0.312534f, 0.854568f, 0.113097f},
      {0.0783791f, 0.340692f, 0.23686f, 0.5932f, 0.38193f},
      {0.430041f, 0.401364f, 0.88266f, 0.226286f, 0.514185f},
      {0.422123f, 0.713778f, 0.813105f, 0.960577f, 0.794308f},
      {0.0531423f, 0.930818f, 0.913336f, 0.382305f, 0.372521f},
      {0.91698f, 0.128078f, 0.901849f, 0.0860355f, 0.432365f},
      {0.749259f, 0.198112f, 0.538301f, 0.739992f, 0.909026f},
      {0.903781f, 0.206122f, 0.743227f, 0.700662f, 0.784729f},
      {0.914658f, 0.625943f, 0.697374f, 0.333459f, 0.213769f},
      {0.313091f, 0.0485961f, 0.625018f, 0.916347f, 0.363119f},
      {0.455916f, 0.982769f, 0.245987f, 0.555492f, 0.938798f},
      {0.0737146f, 0.324519f, 0.325405f, 0.677491f, 0.148078f},
      {0.918677f, 0.537612f, 0.917458f, 0.611973f, 0.965844f},
      {0.832977f, 0.466222f, 0.528761f, 0.348765f, 0.472975f},
      {0.784042f, 0.866144f, 0.00524178f, 0.217837f, 0.145246f},
      {0.308576f, 0.993283f, 0.0244056f, 0.543786f, 0.575841f},
      {0.285113f, 0.12198f, 0.74075f, 0.834888f, 0.561457f},
      {0.635992f, 0.590228f, 0.629378f, 0.112457f, 0.78253f}};

  for (const auto &output : measured) {
    filter.predict(output.drift_x, output.drift_y);
    filter.update(output.position_x, output.position_y, output.variometer);
  }

  assert(std::abs(1 - filter.x()[0] / 0.347191f) < 0.0001f &&
         std::abs(1 - filter.x()[1] / 91.8926f) < 0.0001f &&
         std::abs(1 - filter.x()[2] / 22.9656f) < 0.0001f &&
         std::abs(1 - filter.x()[3] / 20.6146f) < 0.0001f &&
         "The estimated states expected to meet ArduPilot soaring plane "
         "implementation at 0.01% accuracy.");

  return 0;
}()};

} // namespace
} // namespace fcarouge::eigen::sample
