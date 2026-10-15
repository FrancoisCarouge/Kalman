## François Carouge / Kalman

The Kalman filter is a Bayesian filter that uses multivariate Gaussians, a recursive state estimator, a linear quadratic estimator (LQE), and an Infinite Impulse Response (IIR) filter. It is a control theory tool applicable to signal estimation, sensor fusion, or data assimilation problems. The filter is applicable for unimodal and uncorrelated uncertainties. The filter assumes white noise, propagation and measurement functions are differentiable, and that the uncertainty stays centered on the state estimate. The filter is the optimal linear filter under assumptions. The filter updates estimates by multiplying Gaussians rather than integrating differential equations. The filter predicts estimates by adding Gaussians. The filter maintains an estimate of the state and its uncertainty over the sequential estimation process. The filter is named after Rudolf E. Kálmán, who was one of the primary developers of its theory in 1960.

This library supports various simple and extended filters. The implementation is independent from linear algebra backends. Arbitrary parameters can be added to the prediction and update stages to participate in gain-scheduling or linear parameter varying (LPV) systems. The default filter type is a generalized, customizable, and extended filter. The default type parameters implement a one-state, one-output, and double-precision floating-point type filter. The default update equation uses the Joseph form. Examples illustrate various usages and implementation tradeoffs. A standard formatter specialization is included for representation of the filter states. Filters with `state x output x input` dimensions as 1x1x1 and 1x1x0 (no input) are supported through vanilla C++. Higher dimension filters require a linear algebra backend. Customization points and type injections allow for implementation tradeoffs.

```cpp
kalman filter{state{60.}, output<double>,
              estimate_uncertainty{225.}, output_uncertainty{25.}};

filter.update(48.54);

std::println("{}", filter);

// {"k": 0.9, "p": 22.5, "r": 25, "s": 250, "x": 49.686, "y": -11.46, "z": 48.54}
```

[![Pipeline](https://github.com/FrancoisCarouge/Kalman/actions/workflows/pipeline.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/pipeline.yml)
[![Freestanding](https://github.com/FrancoisCarouge/Kalman/actions/workflows/freestanding.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/freestanding.yml)
[![Sanitizer](https://github.com/FrancoisCarouge/Kalman/actions/workflows/sanitizer.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/sanitizer.yml)
[![Format](https://github.com/FrancoisCarouge/Kalman/actions/workflows/format.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/format.yml)
[![ClangTidy](https://github.com/FrancoisCarouge/Kalman/actions/workflows/clang_tidy.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/clang_tidy.yml)
[![CppCheck](https://github.com/FrancoisCarouge/Kalman/actions/workflows/cppcheck.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/cppcheck.yml)
[![Doxygen](https://github.com/FrancoisCarouge/Kalman/actions/workflows/doxygen.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/doxygen.yml)
[![Valgrind](https://github.com/FrancoisCarouge/Kalman/actions/workflows/valgrind_memory.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/valgrind_memory.yml)
[![Public Domain](https://img.shields.io/badge/License-Public%20Domain%20%F0%9F%94%97-brightgreen)](https://raw.githubusercontent.com/francoiscarouge/Kalman/master/LICENSE.txt)
[![License Scan](https://app.fossa.com/api/projects/git%2Bgithub.com%2FFrancoisCarouge%2FKalman.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2FFrancoisCarouge%2FKalman?ref=badge_shield)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8933/badge)](https://www.bestpractices.dev/projects/8933)
[![Deploy Unit Test Code Coverage](https://github.com/FrancoisCarouge/Kalman/actions/workflows/deploy_coverage.yml/badge.svg)](https://francoiscarouge.github.io/Kalman/unit_test_coverage.xhtml)
[![Deploy Doxygen](https://github.com/FrancoisCarouge/Kalman/actions/workflows/deploy_doxygen.yml/badge.svg)](https://francoiscarouge.github.io/Kalman/index.xhtml)
[![Sponsor](https://img.shields.io/badge/Support-Sponsor%20%F0%9F%94%97-brightgreen)](http://paypal.me/francoiscarouge)
[![Discussions](https://img.shields.io/github/discussions/FrancoisCarouge/Kalman)](https://github.com/FrancoisCarouge/Kalman/discussions)

## Installation & Usage

See [INSTALL.md](https://github.com/FrancoisCarouge/Kalman/blob/master/INSTALL.md) for installation instructions.

## Reference

### Layers

![Where Kalman sits in the technical stack: below the application, composing an optionally typed linear algebra backend and adapting strong element types.](https://francoiscarouge.github.io/Kalman/kalman/layers.svg)

The library is the state estimation layer between the application and an established linear algebra backend.

### CMake

Link against the CMake target `fcarouge-kalman::kalman`.

### Include

Include the library header in your sources:

```cpp
#include "fcarouge/kalman.hpp"
```

For forward declarations only, include instead:

```cpp
#include "fcarouge/kalman_forward.hpp"
```

### Namespace

The public API lives in the `fcarouge` namespace. Import it with a using-declaration or namespace alias as your project's style prefers.

### Linear Algebra Backends

Filters with `state x output x input` dimensions as 1x1x1 and 1x1x0 (no input) are supported through vanilla C++. Add a backend for higher dimension filters:

| Integration | Example Backend |
| --- | --- |
| Built-in Types | No backend needed. |
| Eigen | [Example backend: `support/eigen`](https://github.com/FrancoisCarouge/Kalman/tree/master/support/eigen). |
| Eigen + TypedLinearAlgebra | [Example backend: `support/eigexed`](https://github.com/FrancoisCarouge/Kalman/tree/master/support/eigexed). |
| Eigen + TypedLinearAlgebra + mp-units | [Example backend: `support/mp_units_eigen`](https://github.com/FrancoisCarouge/Kalman/tree/master/support/mp_units_eigen). |
| Freestanding | [Example backend: `support/freestanding`](https://github.com/FrancoisCarouge/Kalman/tree/master/support/freestanding). |
| mp-units | [Example backend: `support/mp_units`](https://github.com/FrancoisCarouge/Kalman/tree/master/support/mp_units). |

### Samples

Designing a filter is as much art as science, with the following recipe. Model the real world in state-space notation. Then, compute and select the fundamental matrices, select the states *X*, *P*, the processes *F*, *Q*, the measurements *Z*, *R*, the measurement function *H*, and if the system has control inputs *U*, *G*. Evaluate the performance and iterate.

| Filter | Sample |
| --- | --- |
| 1x1x0 constant system dynamic model filter of the building height. | [Usage example: `sample/kf_1x1x0_building_height/kf_1x1x0_building_height.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_1x1x0_building_height/kf_1x1x0_building_height.cpp), [with units: `sample/kf_1x1x0_building_height_unit/kf_1x1x0_building_height_unit.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_1x1x0_building_height_unit/kf_1x1x0_building_height_unit.cpp). |
| 1x1x0 constant system dynamic model filter of the temperature of a liquid in a tank. | [Usage example: `sample/kf_1x1x0_liquid_temperature/kf_1x1x0_liquid_temperature.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_1x1x0_liquid_temperature/kf_1x1x0_liquid_temperature.cpp), [with units: `sample/kf_1x1x0_liquid_temperature_unit/kf_1x1x0_liquid_temperature_unit.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_1x1x0_liquid_temperature_unit/kf_1x1x0_liquid_temperature_unit.cpp). |
| 1x1x0 nonlinear output model extended filter of the terrain height below a vehicle, from a range finder and an optical flow sensor, with consistency tests and a constraint applied outside of the filter. | [Usage example: `sample/ekf_1x1x0_terrain_estimation/ekf_1x1x0_terrain_estimation.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/ekf_1x1x0_terrain_estimation/ekf_1x1x0_terrain_estimation.cpp). |
| 1x1x1 constant velocity dynamic model filter of the 1-dimension position of a dog. | [Usage example: `sample/kf_1x1x1_1d_dog_position/kf_1x1x1_1d_dog_position.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_1x1x1_1d_dog_position/kf_1x1x1_1d_dog_position.cpp), [with units: `sample/kf_1x1x1_1d_dog_position_unit/kf_1x1x1_1d_dog_position_unit.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_1x1x1_1d_dog_position_unit/kf_1x1x1_1d_dog_position_unit.cpp). |
| 2x1x0 dynamic linear regression filter of the pairs trading hedge ratio and intercept of two stocks, with a daily output model, for mean reversion trading. | [Usage example with units: `sample/kf_2x1x0_pairs_hedge_ratio_unit/kf_2x1x0_pairs_hedge_ratio_unit.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_2x1x0_pairs_hedge_ratio_unit/kf_2x1x0_pairs_hedge_ratio_unit.cpp). |
| 2x1x0 local linear trend structural time series model filter of the short-term forecast of a daily stock closing price. | [Usage example with units: `sample/kf_2x1x0_stock_price_unit/kf_2x1x0_stock_price_unit.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_2x1x0_stock_price_unit/kf_2x1x0_stock_price_unit.cpp). |
| 2x1x1 constant acceleration dynamic model filter of the 1-dimension position and velocity of a rocket altitude. | [Usage example: `sample/kf_2x1x1_1d_rocket_altitude/kf_2x1x1_1d_rocket_altitude.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_2x1x1_1d_rocket_altitude/kf_2x1x1_1d_rocket_altitude.cpp). |
| 3x1x0 nonlinear output model extended filter of the 3-dimension position of a receiver from its ranges to beacons at known positions, with consistency tests after an alignment, as in the ArduPilot autopilot. | [Usage example with units: `sample/ekf_3x1x0_beacon_position_unit/ekf_3x1x0_beacon_position_unit.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/ekf_3x1x0_beacon_position_unit/ekf_3x1x0_beacon_position_unit.cpp). |
| 4x1x0 nonlinear dynamic model extended filter of the thermal, current of warm air, strength, radius, and location. | [Usage example: `sample/ekf_4x1x0_thermal_soaring/ekf_4x1x0_thermal_soaring.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/ekf_4x1x0_thermal_soaring/ekf_4x1x0_thermal_soaring.cpp). |
| 6x2x0 constant acceleration dynamic model filter of the 2-dimension vehicle location, velocity, and acceleration. | [Usage example: `sample/kf_6x2x0_2d_vehicle_position/kf_6x2x0_2d_vehicle_position.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_6x2x0_2d_vehicle_position/kf_6x2x0_2d_vehicle_position.cpp), [with units: `sample/kf_6x2x0_2d_vehicle_position_unit/kf_6x2x0_2d_vehicle_position_unit.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_6x2x0_2d_vehicle_position_unit/kf_6x2x0_2d_vehicle_position_unit.cpp). |
| 6x4x0 extended filter of the 3-dimension position and velocity of the NASA Apollo lunar module abort guidance system for spacecraft rendezvous approaching the command/service module. | [Usage example: `sample/ekf_6x4x0_3d_apollo_position/ekf_6x4x0_3d_apollo_position.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/ekf_6x4x0_3d_apollo_position/ekf_6x4x0_3d_apollo_position.cpp). |
| 8x4x0 constant velocity dynamic model filter of the 2-dimension position and velocity of the center, aspect ratio, and height of a bounding box. | [Usage example: `sample/kf_8x4x0_2d_bounding_box/kf_8x4x0_2d_bounding_box.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_8x4x0_2d_bounding_box/kf_8x4x0_2d_bounding_box.cpp). |
| 9x4x0 error-state extended filter of the attitude, velocity, and gyroscope bias of a camera gimbal, from its inertial sensors, a zero velocity, and a compass heading, as in the ArduPilot autopilot. | [Usage example with units: `sample/ekf_9x4x0_gimbal_attitude_unit/ekf_9x4x0_gimbal_attitude_unit.cpp`](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/ekf_9x4x0_gimbal_attitude_unit/ekf_9x4x0_gimbal_attitude_unit.cpp). |

#### 6x2 Constant Acceleration Dynamic Model Filter

Example from the [2-dimension vehicle location, velocity, and acceleration vehicle estimation](https://francoiscarouge.github.io/Kalman/kf_6x2x0_2d_vehicle_position_8cpp-example.xhtml) sample. Six estimated states and two observed outputs filter.

```cpp
  kalman filter{
    state{0., 0., 0., 0., 0., 0.},
    output<vector<2>>,
    estimate_uncertainty{{500., 0., 0., 0., 0., 0.},
                         {0., 500., 0., 0., 0., 0.},
                         {0., 0., 500., 0., 0., 0.},
                         {0., 0., 0., 500., 0., 0.},
                         {0., 0., 0., 0., 500., 0.},
                         {0., 0., 0., 0., 0., 500.}},
    process_uncertainty{0.2 * 0.2 *
                        matrix<6, 6>{{0.25, 0.5, 0.5, 0., 0., 0.},
                                      {0.5, 1., 1., 0., 0., 0.},
                                      {0.5, 1., 1., 0., 0., 0.},
                                      {0., 0., 0., 0.25, 0.5, 0.5},
                                      {0., 0., 0., 0.5, 1., 1.},
                                      {0., 0., 0., 0.5, 1., 1.}}},
    output_uncertainty{{9., 0.}, {0., 9.}},
    output_model{{1., 0., 0., 0., 0., 0.},
                 {0., 0., 0., 1., 0., 0.}},
    state_transition{{1., 1., 0.5, 0., 0., 0.},
                     {0., 1., 1., 0., 0., 0.},
                     {0., 0., 1., 0., 0., 0.},
                     {0., 0., 0., 1., 1., 0.5},
                     {0., 0., 0., 0., 1., 1.},
                     {0., 0., 0., 0., 0., 1.}}};

filter.predict();
filter.update(-393.66, 300.4);
```

[full sample code](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_6x2x0_2d_vehicle_position/kf_6x2x0_2d_vehicle_position.cpp)

#### 4x1 Nonlinear Dynamic Model Extended Filter

Example from the [thermal, current of warm air, strength, radius, and location estimation](https://francoiscarouge.github.io/Kalman/ekf_4x1x0_thermal_soaring_8cpp-example.xhtml) sample. Four estimated states and one observed output extended filter with two additional prediction arguments and two additional update arguments.

```cpp
  kalman filter{
    state{trigger_strength, thermal_radius, thermal_position_x,
          thermal_position_y},
    output<float>,
    estimate_uncertainty{{strength_covariance, 0.F, 0.F, 0.F},
                         {0.F, radius_covariance, 0.F, 0.F},
                         {0.F, 0.F, position_covariance, 0.F},
                         {0.F, 0.F, 0.F, position_covariance}},
    process_uncertainty{{strength_noise, 0.F, 0.F, 0.F},
                        {0.F, distance_noise, 0.F, 0.F},
                        {0.F, 0.F, distance_noise, 0.F},
                        {0.F, 0.F, 0.F, distance_noise}},
    output_uncertainty{measure_noise},
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
      const vector<4> drifts{0.F, 0.F, drift_x, drift_y};
      return x + drifts;
    }},
    observation{[](const vector<4> &x, const float &position_x,
                   const float &position_y) -> float {
      return x(0) * std::exp(-(std::pow(x[2] - position_x, 2.F) +
                               std::pow(x[3] - position_y, 2.F)) /
                             std::pow(x[1], 2.F));
    }},
    update_types<float, float>,
    prediction_types<float, float>};

filter.predict(drift_x, drift_y);
filter.update(position_x, position_y, variometer);
```

[full sample code](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/ekf_4x1x0_thermal_soaring/ekf_4x1x0_thermal_soaring.cpp)

### Class Kalman

Also documented in the [fcarouge/kalman.hpp](https://github.com/FrancoisCarouge/Kalman/tree/master/include/fcarouge/kalman.hpp) header.

#### Declaration

```cpp
template <typename Filter>
class kalman final : public kf::conditional_member_types<Filter>
```

#### Template Parameters

| Template Parameter | Definition |
| --- | --- |
| `Filter` | Exposition only. The deduced internal filter template parameter. Class template argument deduction (CTAD) figures out the filter type based on the declared configuration. See deduction guide. |

#### Member Types

| Member Type | Dimensions | Definition | Also Known As |
| --- | --- | --- | --- |
| `estimate_uncertainty` | x by x | Type of the estimated, hidden covariance matrix `p`. | *P*, *Σ* |
| `gain` | x by z | Type of the gain matrix `k`. | *K*, *L* |
| `innovation_uncertainty` | z by z | Type of the innovation uncertainty matrix `s`. | *S* |
| `innovation` | z by 1 | Type of the innovation column vector `y`. | *Y* |
| `input_control` | x by u | Type of the control transition matrix `g`. This member type is defined only if the filter supports input control. | *G*, *B* |
| `input` | u by 1 | Type of the control column vector `u`. This member type is defined only if the filter supports input. | *U* |
| `output_model` | z by x | Type of the observation transition matrix `h`. This member type is defined only if the filter supports output model. | *H*, *C* |
| `output_uncertainty` | z by z | Type of the observation, measurement noise covariance matrix `r`. | *R* |
| `output` | z by 1 | Type of the observation column vector `z`. | *Z*, *Y*, *O* |
| `process_uncertainty` | x by x | Type of the process noise covariance matrix `q`. | *Q* |
| `state_transition` | x by x | Type of the state transition matrix `f`. | *F*, *Φ*, *A* |
| `state` | x by 1 | Type of the state estimate, hidden column vector `x`. | *X* |

The member types are optionally present according to the filter configuration.

#### Member Functions

| Member Function | Definition |
| --- | --- |
| `(constructor)` | Constructs a Kalman filter from its declared configuration. |
| `(move constructor)` | Constructs the filter, default. |
| `(move assignment operator)` | Assigns values to the filter, default. |
| `(destructor)` | Destructs the filter. |

##### Characteristics

| Characteristic | Definition |
| --- | --- |
| `f` | Manages the state transition matrix *F*. Gets or sets the value. Configures the callable object of expression `state_transition(const input &, const PredictionTypes &...)`, `state_transition(const PredictionTypes &...)`, or `state_transition(const state &, const PredictionTypes &...)` evaluated at the state estimate before the prediction, of the type configured at construction, to compute the value. The default value is the matrix with all its diagonal elements equal to ones, and zeroes everywhere else. |
| `g` | Manages the control transition matrix *G*. Gets or sets the value. Configures the callable object of expression `input_control(const PredictionTypes &...)`, of the type configured at construction, to compute the value. The default value is the matrix with all its diagonal elements equal to ones, and zeroes everywhere else. This member function is defined only if the filter supports input control. |
| `h` | Manages the observation transition matrix *H*. Gets or sets the value. Configures the callable object of expression `output_model(const state &, const UpdateTypes &...)`, of the type configured at construction, to compute the value. The default value is the matrix with all its diagonal elements equal to ones, and zeroes everywhere else. This member function is defined only if the filter supports output model. |
| `k` | Manages the gain matrix *K*. Gets the value last computed during the update. The default value is the matrix with all its diagonal elements equal to ones, and zeroes everywhere else. |
| `p` | Manages the estimated covariance matrix *P*. Gets or sets the value. The default value is the matrix with all its diagonal elements equal to ones, and zeroes everywhere else. |
| `q` | Manages the process noise covariance matrix *Q* from the process noise *w* expected value *E[wwᵀ]* and its variance *σ²* found by measuring, tuning, educated guesses of the noise. Gets or sets the value. Configures the callable object of expression `process_uncertainty(const state &, const PredictionTypes &...)`, of the type configured at construction, to compute the value. The default value is the null matrix. |
| `r` | Manages the observation, measurement noise covariance matrix *R* from the measurement noise *v* expected value *E[vvᵀ]* and its variance *σ²* found by measuring, tuning, educated guesses of the noise. Gets or sets the value. Configures the callable object of expression `output_uncertainty(const state &, const output &, const UpdateTypes &...)`, of the type configured at construction, to compute the value. The default value is the null matrix. |
| `s` | Manages the innovation uncertainty matrix *S*. Gets the value last computed during the update. The default value is the matrix with all its diagonal elements equal to ones, and zeroes everywhere else. |
| `u` | Manages the control column vector *U*. Gets the value last used in prediction. This member function is defined only if the filter supports input. |
| `x` | Manages the state estimate column vector *X*. Gets or sets the value. The default value is the null column vector. |
| `y` | Manages the innovation column vector *Y*. Gets the value last computed during the update. The default value is the null column vector. |
| `z` | Manages the observation column vector *Z*. Gets the value last used during the update. The default value is the null column vector. |

The characteristics are optionally present according to the filter configuration. A characteristic is set from values that direct-list-initialize it, without narrowing of the characteristic nor of its elements, through a modifiable filter, or replaced by a callable of its configured type. A characteristic is read as a constant reference from an lvalue filter, valid for the lifetime of the filter and reflecting its later changes, and as a moved value from an rvalue filter.

##### Modifiers

| Modifier | Definition |
| --- | --- |
| `predict` | Produces estimates of the state variables and uncertainties. |
| `update` | Updates the estimates with the outcome of a measurement. |

The prediction and the update take the arguments of their declared types followed by values that direct-list-initialize the input and the output, respectively.

##### Policies

Declare zero, one, or more policies among the characteristics of a filter, in any order, to select its update form or to condition its estimate uncertainty. A policy is part of the filter type, selected at compile-time, at no runtime cost. An omitted policy selects the default filter.

```cpp
kalman filter{state{0., 0.}, output<double>, simple_form, symmetrize, floors{0.1, 0.2}};
kalman constant{state{0., 0.}, output<double>, floors{cw<0.1>, cw<0.2>}};
```

| Policy | Definition |
| --- | --- |
| `joseph_form` | Updates the estimate uncertainty as *P = (I - KH)P(I - KH)ᵀ + KRKᵀ*, the default, declared explicitly. Valid for any gain, the most robust and the most expensive form, evaluated at a quadratic rather than a cubic cost in the state dimension for a scalar output. |
| `information_form` | Updates the estimate uncertainty as *P = (P⁻¹ + HᵀR⁻¹H)⁻¹*. Inverts state-sized rather than output-sized matrices: cheaper only for outputs much larger than the state. Requires invertible estimate and output uncertainties. |
| `simple_form` | Updates the estimate uncertainty as *P = (I - KH)P*, the cheapest form, evaluated at a quadratic rather than a cubic cost in the state dimension for a scalar output. Rounding loses the symmetry of the estimate uncertainty and may lose its positive semi-definiteness. |
| `symmetric_form` | Updates the estimate uncertainty as *P = P - KSKᵀ*, symmetric in exact arithmetic. Subtracting nearly equal matrices of a precise output may lose the positive semi-definiteness of the estimate uncertainty. |
| `symmetrize` | Averages the estimate uncertainty with its transpose after each update and prediction, *P = (P + Pᵀ) / 2*, restoring its symmetry lost to rounding. |
| `floors{σ...}` | Keeps each state variance at least *σ²* after each update and prediction, one standard deviation *σ* per state in its units: a runtime value, or a compile-time constant `cw<σ>` without storage. Useful when a variance would otherwise collapse, an overconfident filter ignoring new outputs, for example under model errors or for weakly observable states. Requires an estimate uncertainty with assignable elements. |

Without an update form policy, the filter updates the estimate uncertainty in the Joseph form: declaring `joseph_form` is the default filter, of the same type. The update forms agree in exact arithmetic since the filter computes the optimal gain: they trade numerical robustness for cost. A filter declares at most one update form, one symmetrization, and one floors. The symmetrization keeps the diagonal of the estimate uncertainty, which the floors change alone: their order does not matter.

### Format

A specialization of the standard formatter is provided for the filter. Use `std::format` to store a formatted representation of all of the characteristics of the filter in a new string. Standard format parameters to be supported. Hosted implementation only.

```cpp
kalman filter;

std::println("{}", filter);
// {"f": 1, "k": 1, "p": 1, "r": 0, "s": 1, "x": 0, "y": 0, "z": 0}
// The characteristics are optionally present according to the filter configuration.
```

### Decorators

Pipe one or more decorators to the filter declaration to attach additional responsibilities. A decorator takes ownership of the filter: decorate an existing filter by moving it, `std::move(filter) | print`.

```cpp
auto filter{kalman{...} | print};
```

| Decorator | Definition |
| --- | --- |
| `print` | Print filter activities to the standard output. Hosted implementation only. |

## More

### Motivations

Kalman filters can be difficult to learn, use, and implement. Users often need fair algebra, domain, and software knowledge. Inadequacy leads to incorrectness, underperformance, and a big ball of mud.

This package explores what could be a Kalman filter implementation a la standard library. The following concerns are considered:
- Separation of the application domain and integration needs.
- Separation of the mathematical concepts and linear algebra implementation.
- Generalization and specialization of modern language and library support.

### Selected Tradeoffs

In theory there is no difference between theory and practice, while in practice there is. The following engineering tradeoffs have been selected for this library implementation:

- Update and prediction additional arguments are stored in the filter at the costs of memory and performance for the benefits of consistent data access and records.
- The default floating point data type for the filter is `double` with about 16 significant digits to reduce loss of information compared to `float`.
- A filter declaration has a single arithmetic type, so that the filter equations never narrow a result into a characteristic, at the cost of converting mixed precision values, for example a `float` measurement of a `double` state, before passing them to the filter.
- The ergonomics and precision of the default filter takes precedence over performance.
- The update form and the conditioning of the estimate uncertainty are compile-time policies of the filter type rather than runtime options, for no runtime cost, at the cost of a distinct filter type per combination of policies.
- The library is freestanding compatible, for embedded and bare-metal targets: it uses no hosted-only standard header, no exception, no run-time type information, and no memory allocation. The formatting and printing support, built on the hosted-only `<format>` and `<print>` headers, are omitted from a freestanding implementation rather than reimplemented. The same header serves both implementations without configuration.
- Callable characteristics are stored by value, their types part of the deduced filter type, without type erasure, for real-time compatibility: constructing, copying, or reconfiguring a filter does not allocate memory, and the callables may be inlined. The costs are a filter type specific to its callables, a callable characteristic replaced only by a callable of the same type, and filter assignment available only for assignable callables, which a lambda expression with captures is not. Configuring a function pointer or a `std::function` restores the replacement by any compatible callable, at the cost of an indirection, and of allocations for `std::function`.
- A characteristic is read through a constant reference from an lvalue filter, without copying it, and written only through its accessor's setting overloads, which constrain the values. Reading from an rvalue filter moves the characteristic out instead, since a reference into the filter would dangle past the full-expression.

### Lessons Learned

The filter's state, output, and input column vectors should be strongly typed parameters to allow the filter to participate in full compile-time safeties verification.

There exist Kalman filters with hundreds of state variables.

The `float` data type has about seven significant digits. Floating point error is a loss of information to account for in design.

The units of useful matrices are factorizable, i.e. the unit of an element is expressed as the product of the row and column indexed units. The deduced units result type of a matrix product collapses and folds the inner indexed units merely returning the outer units.

Safe physical linear algebra not only includes types and units safety, but also coordinate axes and frames reference.

**A specialized filter is an optimization of the general equations, not a new model.** A filter declared without a characteristic is the general filter as-if that characteristic were at its default value, with the unused algebra removed. The meaning of a prediction or an update is a property of the model alone, never of how the filter is declared. Each specialized filter should be tested to agree with the general filter.

### Performance

The [benchmarks](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark) share some performance information. Custom specializations and implementations can outperform this library. Custom optimizations may include: using a different covariance estimation update formula; removing symmetry support; using a different matrix inversion formula; removing unused or identity model dynamics supports; implementing a generated, unrolled filter algebra expressions; or running on accelerator hardware.

![Eigen Update](https://raw.githubusercontent.com/FrancoisCarouge/Kalman/master/benchmark/image/eigen_update.svg)
![Float](https://raw.githubusercontent.com/FrancoisCarouge/Kalman/master/benchmark/image/float.svg)

### Definitions

| Term | Definition |
| --- | --- |
| EKF | The Extended Kalman Filter is the nonlinear version of the Kalman filter. Useful for nonlinear dynamics systems. This filter linearizes the model about an estimate working point of the current mean and covariance. |
| ESKF | The Error State Kalman Filter is the error estimation version of the Kalman filter. Useful for linear error state dynamics systems. This filter estimates the errors rather than the states.
| MSCKF | The Multi-State Constraint Kalman Filter is an EKF-based approach that leverages constraints between multiple and varied measurements to improve pose estimation accuracy and robustness. |
| SR-UKF | The Square-Root Unscented Kalman Filter handles non-linearity with covariance factorization of the unscented transformations (sigma points) to maintain numerical stability. |
| UKF | The Unscented Kalman Filter is the sampled version of the Extended Kalman Filter. Useful for highly nonlinear dynamics systems. This filter samples unscented transformations (sigma points) about an estimate working point of the current mean using an Unscented Transformation technique. |

Further terms should be defined and demonstrated for completeness: CKF, EKF-IMM, EnKF, Euler-KF, Fading-Memory, Finite/Fixed-Memory, Forward-Backward, FKF, IEKF, Joseph, KF, Linearized, MEKF, MRP-EKF, MRP-UKF, QSR-UKF, SKF, Smoother, UKF-GSF, UKF-IMM, USQUE, UDU, and UT.

### Projects

The library is used in projects:

- [GstKalman](https://github.com/FrancoisCarouge/GstKalman): A GStreamer Kalman filter video plugin.

*Your project link here!*

### Resources

- [A New Approach to Linear Filtering and Prediction Problems](https://www.cs.unc.edu/~welch/kalman/kalmanPaper.html) by Kalman, Rudolph Emil in Transactions of the ASME - Journal of Basic Engineering, Volume 82, Series D, pp 35-45, 1960 - Transcription by John Lukesh.
- [KalmanFilter.NET](https://www.kalmanfilter.net) by Alex Becker.
- [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) by Roger Labbe.
- [How Kalman Filters Work](https://www.anuncommonlab.com/articles/how-kalman-filters-work) by Tucker McClure of An Uncommon Lab.
- [Wikipedia Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) by Wikipedia, the free encyclopedia.
- [Applications of Kalman Filtering in Aerospace 1960 to the Present](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5466132) by Mohinder S. Grewal and Angus P. Andrews. IEEE 2010.
- [Taking Static Type-Safety to the Next Level - Physical Units for Matrices](https://www.youtube.com/watch?v=aF3samjRzD4) by Daniel Withopf at CppCon 2022 [[slides](https://meetingcpp.com/mcpp/slides/2021/Physical-units-for-matrices6397.pdf)].
- [Units Libraries and Autonomous Vehicles: Lessons from the Trenches](https://www.youtube.com/watch?v=5dhFtSu3wCo) by Chip Hogg at CppCon 2021.
- [R-649: The Apollo Rendezvous Navigation Filter Theory, Description and Performance, Volume 1 of 2](https://archive.org/details/R649Volume1) by Eugene S. Muller, Jr., and Peter M. Kachmar.
- [ArduPilot](https://ardupilot.org) by the ArduPilot Dev Team.
- [InertialNav](https://github.com/priseborough/InertialNav) by Paul Riseborough, the derivations of the ArduPilot navigation filters, including its range beacon filter.
- [Estimation with Applications to Tracking and Navigation](https://doi.org/10.1002/0471221279) by Yaakov Bar-Shalom, X. Rong Li, and Thiagalingam Kirubarajan. John Wiley & Sons 2001.
- [Quaternion Kinematics for the Error-State Kalman Filter](https://arxiv.org/abs/1711.02508) by Joan Solà. 2017.

### Third Party Acknowledgement

The library is designed, developed, and tested with the help of third-party tools and services acknowledged and thanked here:

- [actions-gh-pages](https://github.com/peaceiris/actions-gh-pages) to upload the documentation to GitHub pages.
- [Clang](https://clang.llvm.org) for compilation and code sanitizers.
- [clang-format](https://clang.llvm.org/docs/ClangFormat.html) for code formatting.
- [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) for static analysis.
- [CMake](https://cmake.org) for build automation.
- [cmakefmt](https://cmakefmt.dev) for pretty CMake list files.
- [cppcheck](https://cppcheck.sourceforge.io) for static analysis.
- [Doxygen](https://doxygen.nl) for documentation generation.
- [Doxygen Awesome](https://github.com/jothepro/doxygen-awesome-css) for pretty documentation.
- [Eigen](https://eigen.tuxfamily.org/) for linear algebra.
- [GCC](https://gcc.gnu.org) for compilation and code sanitizers.
- [gitleaks](https://github.com/gitleaks/gitleaks) for secret scanning.
- [Google Benchmark](https://github.com/google/benchmark) to implement the benchmarks.
- [gsl-lite](https://github.com/gsl-lite/gsl-lite) for guidelines support library.
- [lcov](http://ltp.sourceforge.net/coverage/lcov.php) to process coverage information.
- [Matplot++](https://github.com/alandefreitas/matplotplusplus) for plotting.
- [mp-units](https://github.com/mpusz/mp-units) the quantities and units library for C++.
- [MSVC](https://docs.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist) for compilation and code sanitizers.
- [pre-commit](https://pre-commit.com) for repository hygiene hooks.
- [TypedLinearAlgebra](https://github.com/FrancoisCarouge/TypedLinearAlgebra) for typed linear algebra.
- [Valgrind](https://valgrind.org) to check for correct memory management.

### Contributors

Thank you to everyone who has contributed code, tests, or documentation!

- [François Carouge](https://github.com/FrancoisCarouge) — creator and maintainer.

See [CONTRIBUTORS.md](https://github.com/FrancoisCarouge/Kalman/blob/master/CONTRIBUTORS.md) for the full, credited list, [CHANGELOG.md](https://github.com/FrancoisCarouge/Kalman/blob/master/CHANGELOG.md) for notable changes, and [CONTRIBUTING.md](https://github.com/FrancoisCarouge/Kalman/blob/master/CONTRIBUTING.md) to join.

### Sponsors

Become a sponsor today! Support this project with coffee and infrastructure!

[![Sponsor](https://img.shields.io/badge/Support-Sponsor-brightgreen)](http://paypal.me/francoiscarouge)

#### Corporations & Institutions

*Your group logo and link here!*

#### Individuals

*Your name and link here!*

Thanks everyone!


## License

<img align="right" src="http://opensource.org/trademarks/opensource/OSI-Approved-License-100x137.png">

Kalman Filter is public domain:

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

For more information, please refer to <https://unlicense.org>
