#include "fcarouge/eigen/kalman.hpp"

#include <cassert>
#include <cmath>

namespace fcarouge::eigen::sample {
namespace {

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
//! @example ekf_4x1x0_ardupilot_soaring.cpp
//!
//! @todo Add a data set and assert for correctness of results.
[[maybe_unused]] auto ekf_4x1x0_ardupilot_soaring{[] {
  // 4x1 extended filter with additional parameter for prediction: driftX [m],
  // driftY [m]. Constant time step.
  using kalman = kalman<vector<float, 4>, float, void, std::tuple<float, float>,
                        std::tuple<float, float>>;

  kalman k;

  // Initialization
  const float trigger_strength{0};
  const float thermal_radius{80};
  const float thermal_position_x{5};
  const float thermal_position_y{0};
  k.x(trigger_strength, thermal_radius, thermal_position_x, thermal_position_y);

  const float strength_covariance{0.0049};
  const float radius_covariance{400};
  const float position_covariance{400};
  k.p(kalman::estimate_uncertainty{{strength_covariance, 0, 0, 0},
                                   {0, radius_covariance, 0, 0},
                                   {0, 0, position_covariance, 0},
                                   {0, 0, 0, position_covariance}});

  // No process dynamics: F = ∂f/∂X = I4 Default.

  k.transition([](const kalman::state &x, const float &drift_x,
                  const float &drift_y) -> kalman::state {
    //! @todo Could make sure that x[1] stays positive, greater than 40.
    const kalman::state drifts{0, 0, drift_x, drift_y};
    return x + drifts;
  });

  const float strength_noise{std::pow(0.001f, 2.f)};
  const float distance_noise{std::pow(0.03f, 2.f)};
  k.q(kalman::process_uncertainty{{strength_noise, 0, 0, 0},
                                  {0, distance_noise, 0, 0},
                                  {0, 0, distance_noise, 0},
                                  {0, 0, 0, distance_noise}});

  const float measure_noise{std::pow(0.45f, 2.f)};
  k.r(kalman::output_uncertainty{measure_noise});

  // Observation Z: [w] vertical air velocity w at the aircraft’s
  // position w.r.t. the thermal center [m.s^-1].
  k.observation([](const kalman::state &x, const float &position_x,
                   const float &position_y) -> kalman::output {
    return x(0) * std::exp(-(std::pow(x[2] - position_x, 2.f) +
                             std::pow(x[3] - position_y, 2.f)) /
                           std::pow(x[1], 2.f));
  });

  // See the ArduSoar paper for the equation for H = ∂h/∂X:
  k.h([](const kalman::state &x, const float &position_x,
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
      {0.0756891, 0.749786, 0.878827, 0.806808, 0.155487},
      {0.506366, 0.261469, 0.886986, 0.332883, 0.434406},
      {0.249769, 0.242154, 0.616454, 0.672545, 0.24927},
      {0.358587, 0.556206, 0.909985, 0.370336, 0.553264},
      {0.370579, 0.368003, 0.491917, 0.635429, 0.73594},
      {0.82946, 0.0221123, 0.461047, 0.940697, 0.987409},
      {0.462132, 0.708865, 0.941915, 0.122432, 0.911597},
      {0.888334, 0.542419, 0.773781, 0.116075, 0.917592},
      {0.229376, 0.174244, 0.972009, 0.509611, 0.37637},
      {0.887738, 0.707866, 0.90959, 0.430274, 0.242523},
      {0.40713, 0.0696747, 0.456659, 0.979656, 0.11167},
      {0.77115, 0.183994, 0.944587, 0.467626, 0.0219546},
      {0.137442, 0.316077, 0.660742, 0.828009, 0.852228},
      {0.128113, 0.0757587, 0.742959, 0.360531, 0.3932},
      {0.161107, 0.709262, 0.690847, 0.161165, 0.237205},
      {0.664184, 0.658516, 0.972067, 0.465567, 0.807259},
      {0.669789, 0.236436, 0.341701, 0.430546, 0.229097},
      {0.159471, 0.122824, 0.975034, 0.833685, 0.78011},
      {0.284848, 0.917524, 0.358084, 0.82927, 0.0983398},
      {0.209027, 0.573124, 0.428336, 0.106116, 0.17974},
      {0.861987, 0.110099, 0.0994602, 0.208052, 0.0545667},
      {0.483002, 0.707016, 0.189368, 0.0626376, 0.992816},
      {0.588928, 0.644143, 0.763512, 0.444366, 0.251652},
      {0.419946, 0.338175, 0.286543, 0.97232, 0.908061},
      {0.0625373, 0.855109, 0.763831, 0.622934, 0.364608},
      {0.55833, 0.505803, 0.600797, 0.342724, 0.735087},
      {0.664873, 0.224638, 0.385409, 0.892807, 0.695},
      {0.255295, 0.0264766, 0.229274, 0.723291, 0.552242},
      {0.412129, 0.856404, 0.395075, 0.261842, 0.947885},
      {0.468212, 0.849367, 0.00615251, 0.842904, 0.700869},
      {0.311582, 0.293401, 0.299637, 0.567025, 0.659598},
      {0.695464, 0.941376, 0.21219, 0.27813, 0.289406},
      {0.000397467, 0.301337, 0.71608, 0.296278, 0.718923},
      {0.36314, 0.263077, 0.193163, 0.295399, 0.0523569},
      {0.128381, 0.572157, 0.971297, 0.516492, 0.921166},
      {0.596215, 0.909239, 0.133898, 0.506903, 0.0335569},
      {0.444556, 0.997721, 0.348369, 0.644847, 0.80885},
      {0.891465, 0.0797467, 0.85753, 0.369457, 0.418543},
      {0.861948, 0.520583, 0.900797, 0.153884, 0.080031},
      {0.169696, 0.981169, 0.406729, 0.292696, 0.831505},
      {0.172591, 0.349291, 0.782213, 0.534652, 0.214628},
      {0.875081, 0.746097, 0.0806311, 0.15685, 0.357471},
      {0.519389, 0.007303, 0.18117, 0.370993, 0.427305},
      {0.961372, 0.218945, 0.486608, 0.618755, 0.168813},
      {0.537862, 0.451312, 0.384422, 0.540216, 0.525636},
      {0.494387, 0.162124, 0.0136825, 0.127037, 0.803511},
      {0.409087, 0.991167, 0.276877, 0.188698, 0.155701},
      {0.851474, 0.54778, 0.133586, 0.37391, 0.137362},
      {0.0148137, 0.97396, 0.945259, 0.297432, 0.260494},
      {0.906864, 0.13484, 0.214258, 0.924681, 0.618572},
      {0.141742, 0.563986, 0.502602, 0.416297, 0.97038},
      {0.698555, 0.406929, 0.558199, 0.875364, 0.736008},
      {0.175105, 0.270328, 0.332957, 0.145101, 0.765857},
      {0.68083, 0.125673, 0.922594, 0.831683, 0.457214},
      {0.520728, 0.26214, 0.458674, 0.306454, 0.783164},
      {0.780442, 0.472245, 0.125185, 0.460146, 0.0847598},
      {0.360083, 0.0686402, 0.328997, 0.799852, 0.818809},
      {0.71546, 0.717884, 0.253842, 0.812915, 0.0141433},
      {0.441185, 0.171204, 0.0432966, 0.739241, 0.448679},
      {0.399117, 0.148854, 0.743042, 0.0230124, 0.378786},
      {0.841239, 0.292533, 0.391296, 0.734326, 0.0597166},
      {0.350847, 0.519149, 0.808508, 0.113644, 0.673261},
      {0.229909, 0.814871, 0.118688, 0.612729, 0.354682},
      {0.734755, 0.675693, 0.646155, 0.0296504, 0.405621},
      {0.121731, 0.231111, 0.47879, 0.733299, 0.270893},
      {0.732981, 0.813999, 0.597652, 0.455436, 0.691262},
      {0.10297, 0.534613, 0.553605, 0.777385, 0.553588},
      {0.441429, 0.974205, 0.120671, 0.279931, 0.624484},
      {0.531836, 0.697762, 0.274009, 0.827927, 0.741129},
      {0.745307, 0.085542, 0.473629, 0.286912, 0.175756},
      {0.758466, 0.268705, 0.108006, 0.291002, 0.559732},
      {0.632262, 0.733193, 0.919653, 0.165692, 0.84716},
      {0.0107621, 0.694084, 0.35781, 0.793076, 0.0818898},
      {0.17388, 0.333606, 0.867638, 0.969285, 0.887633},
      {0.255376, 0.180532, 0.737631, 0.869954, 0.875926},
      {0.525821, 0.882517, 0.224126, 0.906093, 0.557676},
      {0.516693, 0.986614, 0.644313, 0.00903489, 0.207868},
      {0.00175451, 0.49772, 0.436713, 0.0418148, 0.63547},
      {0.559954, 0.192099, 0.0787102, 0.976933, 0.552542},
      {0.983202, 0.165426, 0.136735, 0.467933, 0.626612},
      {0.520497, 0.593702, 0.0155549, 0.791301, 0.635127},
      {0.934924, 0.0663795, 0.513404, 0.791586, 0.68594},
      {0.977299, 0.682359, 0.0689664, 0.769369, 0.169862},
      {0.681586, 0.900795, 0.312534, 0.854568, 0.113097},
      {0.0783791, 0.340692, 0.23686, 0.5932, 0.38193},
      {0.430041, 0.401364, 0.88266, 0.226286, 0.514185},
      {0.422123, 0.713778, 0.813105, 0.960577, 0.794308},
      {0.0531423, 0.930818, 0.913336, 0.382305, 0.372521},
      {0.91698, 0.128078, 0.901849, 0.0860355, 0.432365},
      {0.749259, 0.198112, 0.538301, 0.739992, 0.909026},
      {0.903781, 0.206122, 0.743227, 0.700662, 0.784729},
      {0.914658, 0.625943, 0.697374, 0.333459, 0.213769},
      {0.313091, 0.0485961, 0.625018, 0.916347, 0.363119},
      {0.455916, 0.982769, 0.245987, 0.555492, 0.938798},
      {0.0737146, 0.324519, 0.325405, 0.677491, 0.148078},
      {0.918677, 0.537612, 0.917458, 0.611973, 0.965844},
      {0.832977, 0.466222, 0.528761, 0.348765, 0.472975},
      {0.784042, 0.866144, 0.00524178, 0.217837, 0.145246},
      {0.308576, 0.993283, 0.0244056, 0.543786, 0.575841},
      {0.285113, 0.12198, 0.74075, 0.834888, 0.561457},
      {0.635992, 0.590228, 0.629378, 0.112457, 0.78253}};

  for (const auto &output : measured) {
    k.predict(output.drift_x, output.drift_y);
    k.update(output.position_x, output.position_y, output.variometer);
  }

  assert(std::abs(1 - k.x()[0] / 0.347191f) < 0.0001f &&
         std::abs(1 - k.x()[1] / 91.8926f) < 0.0001f &&
         std::abs(1 - k.x()[2] / 22.9656f) < 0.0001f &&
         std::abs(1 - k.x()[3] / 20.6146f) < 0.0001f &&
         "The estimated states expected to meet ArduPilot soaring plane "
         "implementation at 0.01% accuracy.");

  return 0;
}()};

} // namespace
} // namespace fcarouge::eigen::sample
