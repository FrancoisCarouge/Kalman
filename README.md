# Kalman Filter

The Kalman filter is a Bayesian filter that uses multivariate Gaussians, a recursive state estimator, a linear quadratic estimator (LQE), and an Infinite Impulse Response (IIR) filter. It is a control theory tool applicable to signal estimation, sensor fusion, or data assimilation problems. The filter is applicable for unimodal and uncorrelated uncertainties. The filter assumes white noise, propagation and measurement functions are differentiable, and that the uncertainty stays centered on the state estimate. The filter is the optimal linear filter under assumptions. The filter updates estimates by multiplying Gaussians rather than integrating differential equations. The filter predicts estimates by adding Gaussians. The filter maintains an estimate of the state and its uncertainty over the sequential estimation process. The filter is named after Rudolf E. Kálmán, who was one of the primary developers of its theory in 1960.

Designing a filter is as much art as science, with the following recipe. Model the real world in state-space notation. Then, compute and select the fundamental matrices, select the states *X*, *P*, the processes *F*, *Q*, the measurements *Z*, *R*, the measurement function *H*, and if the system has control inputs *U*, *G*. Evaluate the performance and iterate.

This library supports various simple and extended filters. The implementation is independent from linear algebra backends. Arbitrary parameters can be added to the prediction and update stages to participate in gain-scheduling or linear parameter varying (LPV) systems. The default filter type is a generalized, customizable, and extended filter. The default type parameters implement a one-state, one-output, and double-precision floating-point type filter. The default update equation uses the Joseph form. Examples illustrate various usages and implementation tradeoffs. A standard formatter specialization is included for representation of the filter states. Filters with `state x output x input` dimensions as 1x1x1 and 1x1x0 (no input) are supported through vanilla C++. Higher dimension filters require a linear algebra backend. Customization points and type injections allow for implementation tradeoffs.

# Examples

## 1x1 Constant System Dynamic Model Filter

Example from the [building height estimation](https://francoiscarouge.github.io/Kalman/kf_1x1x0_building_height_8cpp-example.xhtml) sample. One estimated state and one observed output filter.

```cpp
  kalman filter{
    state{60.},
    output<double>,
    estimate_uncertainty{225.},
    output_uncertainty{25.}
  };

filter.update(48.54);
```

[full sample code](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_1x1x0_building_height.cpp)

## 6x2 Constant Acceleration Dynamic Model Filter

Example from the [2-dimension vehicle location, velocity, and acceleration vehicle estimation](https://francoiscarouge.github.io/Kalman/kf_6x2x0_vehicle_location_8cpp-example.xhtml) sample. Six estimated states and two observed outputs filter.

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

[full sample code](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_6x2x0_vehicle_location.cpp)

## 4x1 Nonlinear Dynamic Model Extended Filter

Example from the [thermal, current of warm air, strength, radius, and location estimation](https://francoiscarouge.github.io/Kalman/ekf_4x1x0_soaring_8cpp-example.xhtml) sample. Four estimated states and one observed output extended filter with two additional prediction arguments and two additional update arguments.

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

[full sample code](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/ekf_4x1x0_soaring.cpp)

## Other Examples

- 1x1 constant system dynamic model filter of the [temperature of a liquid in a tank](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_1x1x0_building_height.cpp).
- 1x1x1 constant velocity dynamic model filter of the [1-dimension position of a dog](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_1x1x1_dog_position.cpp).
- 2x1x1 constant acceleration dynamic model filter of the [1-dimension position and velocity of a rocket altitude](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_2x1x1_rocket_altitude.cpp).
- 8x4 constant velocity dynamic model filter of the [2-dimension position and velocity of the center, aspect ratio, and height of a bounding box](https://github.com/FrancoisCarouge/Kalman/tree/master/sample/kf_8x4x0_deep_sort_bounding_box.cpp).

# Installation

Example of installation commands in Shell:

```shell
git clone --depth 1 "https://github.com/FrancoisCarouge/kalman"
cmake -S "kalman" -B "build"
cmake --build "build" --parallel
sudo cmake --install "build"
```

Another variation for your CMake infrastructure via fetch content:

```cmake
include(FetchContent)

FetchContent_Declare(
  fcarouge-kalman
  GIT_REPOSITORY "https://github.com/FrancoisCarouge/kalman"
  FIND_PACKAGE_ARGS NAMES fcarouge-kalman)
FetchContent_MakeAvailable(fcarouge-kalman)

target_link_libraries(your_target PRIVATE fcarouge-kalman::kalman)
```

[For more, see installation instructions](https://github.com/FrancoisCarouge/Kalman/tree/master/INSTALL.md).

# Reference

## Class kalman

Also documented in the [fcarouge/kalman.hpp](https://github.com/FrancoisCarouge/Kalman/tree/master/include/fcarouge/kalman.hpp) header.

### Declaration

```cpp
template <typename Filter>
class kalman final : public internal::conditional_member_types<Filter>
```

### Template Parameters

| Template Parameter | Definition |
| --- | --- |
| `Filter` | Exposition only. The deduced internal filter template parameter. Class template argument deduction (CTAD) figures out the filter type based on the declared configuration. See deduction guide. |

### Member Types

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

### Member Functions

| Member Function | Definition |
| --- | --- |
| `(constructor)` | Constructs the filter. Configures the filter the deduction guides. |
| `(move constructor)` | Constructs the filter, default. |
| `(move assignment operator)` | Assigns values to the filter, default. |
| `(destructor)` | Destructs the filter. |

#### Characteristics

| Characteristic | Definition |
| --- | --- |
| `f` | Manages the state transition matrix *F*. Gets or sets the value. Configures the callable object of expression `state_transition(const state &, const input &, const PredictionTypes &...)` to compute the value. The default value is the matrix with all its diagonal elements equal to ones, and zeroes everywhere else. |
| `g` | Manages the control transition matrix *G*. Gets or sets the value. Configures the callable object of expression `input_control(const PredictionTypes &...)` to compute the value. The default value is the matrix with all its diagonal elements equal to ones, and zeroes everywhere else. This member function is defined only if the filter supports input control. |
| `h` | Manages the observation transition matrix *H*. Gets or sets the value. Configures the callable object of expression `output_model(const state &, const UpdateTypes &...)` to compute the value. The default value is the matrix with all its diagonal elements equal to ones, and zeroes everywhere else. This member function is defined only if the filter supports output model. |
| `k` | Manages the gain matrix *K*. Gets the value last computed during the update. The default value is the matrix with all its diagonal elements equal to ones, and zeroes everywhere else. |
| `p` | Manages the estimated covariance matrix *P*. Gets or sets the value. The default value is the matrix with all its diagonal elements equal to ones, and zeroes everywhere else. |
| `q` | Manages the process noise covariance matrix *Q* from the process noise *w* expected value *E[wwᵀ]* and its variance *σ²* found by measuring, tuning, educated guesses of the noise. Gets or sets the value. Configures the callable object of expression `process_uncertainty(const state &, const PredictionTypes &...)` to compute the value. The default value is the null matrix. |
| `r` | Manages the observation, measurement noise covariance matrix *R* from the measurement noise *v* expected value *E[vvᵀ]* and its variance *σ²* found by measuring, tuning, educated guesses of the noise. Gets or sets the value. Configures the callable object of expression `output_uncertainty(const state &, const output &, const UpdateTypes &...)` to compute the value. The default value is the null matrix. |
| `s` | Manages the innovation uncertainty matrix *S*. Gets the value last computed during the update. The default value is the matrix with all its diagonal elements equal to ones, and zeroes everywhere else. |
| `u` | Manages the control column vector *U*. Gets the value last used in prediction. This member function is defined only if the filter supports input. |
| `x` | Manages the state estimate column vector *X*. Gets or sets the value. The default value is the null column vector. |
| `y` | Manages the innovation column vector *Y*. Gets the value last computed during the update. The default value is the null column vector. |
| `z` | Manages the observation column vector *Z*. Gets the value last used during the update. The default value is the null column vector. |

The characteristics are optionally present according to the filter configuration.

#### Modifiers

| Modifier | Definition |
| --- | --- |
| `predict` | Produces estimates of the state variables and uncertainties. |
| `update` | Updates the estimates with the outcome of a measurement. |

## Format

A specialization of the standard formatter is provided for the filter. Use `std::format` to store a formatted representation of all of the characteristics of the filter in a new string. Standard format parameters to be supported.

```cpp
kalman filter;

std::println("{}", filter);
// {"f": 1, "k": 1, "p": 1, "r": 0, "s": 1, "x": 0, "y": 0, "z": 0}
// The characteristics are optionally present according to the filter configuration.
```

# Considerations

## Motivations

Kalman filters can be difficult to learn, use, and implement. Users often need fair algebra, domain, and software knowledge. Inadequacy leads to incorrectness, underperformance, and a big ball of mud.

This package explores what could be a Kalman filter implementation a la standard library. The following concerns are considered:
- Separation of the application domain and integration needs.
- Separation of the mathematical concepts and linear algebra implementation.
- Generalization and specialization of modern language and library support.

## Selected Tradeoffs

In theory there is no difference between theory and practice, while in practice there is. The following engineering tradeoffs have been selected for this library implementation:

- Update and prediction additional arguments are stored in the filter at the costs of memory and performance for the benefits of consistent data access and records.
- The default floating point data type for the filter is `double` with about 16 significant digits to reduce loss of information compared to `float`.
- The ergonomics and precision of the default filter takes precedence over performance.

## Lessons Learned

Design, development, and testing uncovered unexpected facets of the projects:

- The filter's state, output, and input column vectors should be strongly typed parameters to allow the filter to participate in full compile-time safeties verification.
- There exist Kalman filters with hundreds of state variables.
- The `float` data type has about seven significant digits. Floating point error is a loss of information to account for in design.
- The units of useful matrices are factorizable, i.e. the unit of an element is expressed as the product of the row and column indexed units. The deduced units result type of a matrix product collapses and folds the inner indexed units merely returning the outer units.
- Safe physical linear algebra not only includes types and units safety, but also coordinate axes and frames reference.

## Performance

The [benchmarks](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark) share some performance information. Custom specializations and implementations can outperform this library. Custom optimizations may include: using a different covariance estimation update formula; removing symmetry support; using a different matrix inversion formula; removing unused or identity model dynamics supports; implementing a generated, unrolled filter algebra expressions; or running on accelerator hardware.

![Eigen Update](https://raw.githubusercontent.com/FrancoisCarouge/Kalman/master/benchmark/image/eigen_update.svg)
![Float](https://raw.githubusercontent.com/FrancoisCarouge/Kalman/master/benchmark/image/float.svg)

# Resources

## Definitions

| Term | Definition |
| --- | --- |
| EKF | The Extended Kalman Filter is the nonlinear version of the Kalman filter. Useful for nonlinear dynamics systems. This filter linearizes the model about an estimate working point of the current mean and covariance. |
| ESKF | The Error State Kalman Filter is the error estimation version of the Kalman filter. Useful for linear error state dynamics systems. This filter estimates the errors rather than the states. 
| UKF | The Unscented Kalman Filter is the sampled version of the Extended Kalman Filter. Useful for highly nonlinear dynamics systems. This filter samples sigma points about an estimate working point of the current mean using an Unscented Transformation technique. |

Further terms should be defined and demonstrated for completeness: CKF, EKF-IMM, EnKF, Euler-KF, Fading-Memory, Finite/Fixed-Memory, Forward-Backward, FKF, IEKF, Joseph, KF, Linearized, MEKF, MRP-EKF, MRP-UKF, MSCKF, SKF, Smoother, UKF-GSF, UKF-IMM, USQUE, UDU, and UT.

## Related Resources

- [A New Approach to Linear Filtering and Prediction Problems](https://www.cs.unc.edu/~welch/kalman/kalmanPaper.html) by Kalman, Rudolph Emil in Transactions of the ASME - Journal of Basic Engineering, Volume 82, Series D, pp 35-45, 1960 - Transcription by John Lukesh.
- [KalmanFilter.NET](https://www.kalmanfilter.net) by Alex Becker.
- [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) by Roger Labbe.
- [How Kalman Filters Work](https://www.anuncommonlab.com/articles/how-kalman-filters-work) by Tucker McClure of An Uncommon Lab.
- [Wikipedia Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) by Wikipedia, the free encyclopedia.
- [Applications of Kalman Filtering in Aerospace 1960 to the Present](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5466132) by Mohinder S. Grewal and Angus P. Andrews. IEEE 2010.
- [Taking Static Type-Safety to the Next Level - Physical Units for Matrices](https://www.youtube.com/watch?v=aF3samjRzD4) by Daniel Withopf at CppCon 2022 [[slides](https://meetingcpp.com/mcpp/slides/2021/Physical-units-for-matrices6397.pdf)].
- [Units Libraries and Autonomous Vehicles: Lessons from the Trenches](https://www.youtube.com/watch?v=5dhFtSu3wCo) by Chip Hogg at CppCon 2021.

## Projects

The library is used in projects:

- [GstKalman](https://github.com/FrancoisCarouge/GstKalman): A GStreamer Kalman filter video plugin.

*Your project link here!*

## Third Party Acknowledgement

The library is designed, developed, and tested with the help of third-party tools and services acknowledged and thanked here:

- [actions-gh-pages](https://github.com/peaceiris/actions-gh-pages) to upload the documentation to GitHub pages.
- [Clang](https://clang.llvm.org) for compilation and code sanitizers.
- [CMake](https://cmake.org) for build automation.
- [cmakelang](https://pypi.org/project/cmakelang) for pretty CMake list files.
- [cppcheck](https://cppcheck.sourceforge.io) for static analysis.
- [Doxygen](https://doxygen.nl) for documentation generation.
- [Doxygen Awesome](https://github.com/jothepro/doxygen-awesome-css) for pretty documentation.
- [Eigen](https://eigen.tuxfamily.org/) for linear algebra.
- [GCC](https://gcc.gnu.org) for compilation and code sanitizers.
- [Google Benchmark](https://github.com/google/benchmark) to implement the benchmarks.
- [lcov](http://ltp.sourceforge.net/coverage/lcov.php) to process coverage information.
- [mp-units](https://github.com/mpusz/mp-units) the quantities and units library for C++.
- [MSVC](https://docs.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist) for compilation and code sanitizers.
- [Valgrind](https://valgrind.org) to check for correct memory management.

## Sponsors

Become a sponsor today! Support this project with coffee and infrastructure!

[![Sponsor](https://img.shields.io/badge/Support-Sponsor-brightgreen)](http://paypal.me/francoiscarouge)

### Corporations & Institutions

*Your group logo and link here!*

### Individuals

*Your name and link here!*

Thanks everyone!

# Continuous Integration & Deployment Actions

[![Code Repository](https://img.shields.io/badge/Repository-GitHub%20%F0%9F%94%97-brightgreen)](https://github.com/FrancoisCarouge/Kalman)
<br>
<br>
[![Pipeline](https://github.com/FrancoisCarouge/Kalman/actions/workflows/pipeline.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/pipeline.yml)
<br>
<br>
[![Sanitizer](https://github.com/FrancoisCarouge/Kalman/actions/workflows/sanitizer.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/sanitizer.yml)
<br>
[![Format](https://github.com/FrancoisCarouge/Kalman/actions/workflows/format.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/format.yml)
<br>
[![ClangTidy](https://github.com/FrancoisCarouge/Kalman/actions/workflows/clang_tidy.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/clang_tidy.yml)
<br>
[![CppCheck](https://github.com/FrancoisCarouge/Kalman/actions/workflows/cppcheck.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/cppcheck.yml)
<br>
[![Doxygen](https://github.com/FrancoisCarouge/Kalman/actions/workflows/doxygen.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/doxygen.yml)
<br>
[![Valgrind](https://github.com/FrancoisCarouge/Kalman/actions/workflows/memory_valgrind.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/memory_valgrind.yml)
<br>
<br>
[![Public Domain](https://img.shields.io/badge/License-Public%20Domain%20%F0%9F%94%97-brightgreen)](https://raw.githubusercontent.com/francoiscarouge/Kalman/master/LICENSE.txt)
<br>
[![License Scan](https://app.fossa.com/api/projects/git%2Bgithub.com%2FFrancoisCarouge%2FKalman.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2FFrancoisCarouge%2FKalman?ref=badge_shield)
<br>
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8933/badge)](https://www.bestpractices.dev/projects/8933)
<br>
<br>
[![Deploy Unit Test Code Coverage](https://github.com/FrancoisCarouge/Kalman/actions/workflows/deploy_coverage.yml/badge.svg)](https://francoiscarouge.github.io/Kalman/unit_test_coverage.xhtml)
<br>
[![Deploy Doxygen](https://github.com/FrancoisCarouge/Kalman/actions/workflows/deploy_doxygen.yml/badge.svg)](https://francoiscarouge.github.io/Kalman/index.xhtml)
<br>
<br>
[![Sponsor](https://img.shields.io/badge/Support-Sponsor%20%F0%9F%94%97-brightgreen)](http://paypal.me/francoiscarouge)

# License

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
