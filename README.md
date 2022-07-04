# Kalman Filter for C++

A generic Kalman filter.

The library supports simple and extended filters. The update equation uses the Joseph form. Control input is accepted. Customization point objects allow for using different linear algebra backends for which standard or Eigen3 implementation is provided.

- [Kalman Filter for C++](#kalman-filter-for-c)
- [Examples](#examples)
  - [1x1 Constant System Dynamic Model](#1x1-constant-system-dynamic-model)
  - [6x2 Constant Acceleration Dynamic Model](#6x2-constant-acceleration-dynamic-model)
  - [4x1 Non-Linear Dynamic Model](#4x1-non-linear-dynamic-model)
- [Continuous Integration & Deployment Actions](#continuous-integration--deployment-actions)
- [Motivation](#motivation)
- [Usage](#usage)
  - [System Installation](#system-installation)
- [Class kalman](#class-kalman)
  - [Template Parameters](#template-parameters)
  - [Member Types](#member-types)
  - [Member Functions](#member-functions)
    - [Characteristics](#characteristics)
    - [Modifiers](#modifiers)
- [Resources](#resources)
- [License](#license)

# Examples

## 1x1 Constant System Dynamic Model

Example from the building height estimation sample. One estimated state and one observed output filter.

```cpp
fcarouge::kalman k;

k.x(60.);
k.p(225.);
k.r(25.);

k(48.54);
```

## 6x2 Constant Acceleration Dynamic Model

Example from the 2-dimension vehicle location estimation sample. Six estimated states and two observed outputs filter.

```cpp
using kalman = fcarouge::eigen::kalman<double, 6, 2, 0>;

kalman k;

k.x(0., 0., 0., 0., 0., 0.);
k.p(kalman::estimate_uncertainty{ { 500, 0, 0, 0, 0, 0 },
                                  { 0, 500, 0, 0, 0, 0 },
                                  { 0, 0, 500, 0, 0, 0 },
                                  { 0, 0, 0, 500, 0, 0 },
                                  { 0, 0, 0, 0, 500, 0 },
                                  { 0, 0, 0, 0, 0, 500 } });
k.q(0.2 * 0.2 * kalman::process_uncertainty{ { 0.25, 0.5, 0.5, 0, 0, 0 },
                                             { 0.5, 1, 1, 0, 0, 0 },
                                             { 0.5, 1, 1, 0, 0, 0 },
                                             { 0, 0, 0, 0.25, 0.5, 0.5 },
                                             { 0, 0, 0, 0.5, 1, 1 },
                                             { 0, 0, 0, 0.5, 1, 1 } });
k.f(kalman::state_transition{ { 1, 1, 0.5, 0, 0, 0 },
                              { 0, 1, 1, 0, 0, 0 },
                              { 0, 0, 1, 0, 0, 0 },
                              { 0, 0, 0, 1, 1, 0.5 },
                              { 0, 0, 0, 0, 1, 1 },
                              { 0, 0, 0, 0, 0, 1 } });
k.h(kalman::output_model{ { 1, 0, 0, 0, 0, 0 },
                          { 0, 0, 0, 1, 0, 0 } });
k.r(kalman::output_uncertainty{ { 9, 0 }, { 0, 9 } });

k(-375.93, 301.78);
```

## 4x1 Non-Linear Dynamic Model

Example from the thermal, current of warm air, strength, radius, and location estimation sample. Four estimated states and one observed output extended filter with two additional prediction arguments and two additional update arguments.

```cpp
using kalman = fcarouge::eigen::kalman<float, 4, 1, 0, std::tuple<float, float>,
                                        std::tuple<float, float>>;

kalman k;

k.x(1 / 4.06, 80, 0, 0);
k.p(kalman::estimate_uncertainty{ { 0.0049, 0, 0, 0 },
                                  { 0, 400, 0, 0 },
                                  { 0, 0, 400, 0 },
                                  { 0, 0, 0, 400 } });
k.transition([](const kalman::state &x, const float &drift_x,
                const float &drift_y) -> kalman::state {
  return x + kalman::state{ 0, 0, -drift_x, -drift_y };
});
k.q(kalman::process_uncertainty{ { 0.000001, 0, 0, 0 },
                                 { 0, 0.0009, 0, 0 },
                                 { 0, 0, 0.0009, 0 },
                                 { 0, 0, 0, 0.0009 } });
k.r(0.2025);
k.observation([](const kalman::state &x, const float &position_x,
                 const float &position_y) -> kalman::output {
  return kalman::output{ x(0) *
    std::exp(-((x(2) - position_x)*(x(2) - position_x) +
    (x(3) - position_y) * (x(3) - position_y)) / x(1) * x(1)) };
k.h([](const kalman::state &x, const float &position_x,
       const float &position_y) -> kalman::output_model {
  const auto exp{ std::exp(-((x(2) - position_x) * (x(2) - position_x) +
    (x(3) - position_y) * (x(3) - position_y)) / (x(1) * x(1))) };
  const kalman::output_model h{
    exp,
    2 * x(0) * (((x(2) - position_x) * (x(2) - position_x) +
    (x(3) - position_y) * (x(3) - position_y)) / (x(1) * x(1))) * exp,
    -2 * (x(0) * (x(2) - position_x) / (x(1) * x(1))) * exp,
    -2 * (x(0) * (x(3) - position_y) / (x(1) * x(1))) * exp
  };
  return h;
});

k(drift_x, drift_y, position_x, position_y, variometer);
```

# Continuous Integration & Deployment Actions

[![Code Repository](https://img.shields.io/badge/Repository-GitHub%20%F0%9F%94%97-brightgreen)](https://github.com/FrancoisCarouge/Kalman)
<br>
[![Test: Ubuntu 22.04 GCC Trunk](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_ubuntu-22-04_gcc-trunk.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_ubuntu-22-04_gcc-trunk.yml)
<br>
[![Test: Windows 2019 MSVC](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_windows-2019_msvc.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_windows-2019_msvc.yml)
<br>
[![Test: Ubuntu 22.04 Clang Trunk](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_ubuntu-22-04_clang-trunk.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_ubuntu-22-04_clang-trunk.yml)
<br>
<br>
[![Test Undefined Behavior: Sanitizer](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_sanitizer_undefined_behavior.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_sanitizer_undefined_behavior.yml)
<br>
[![Test Thread: Sanitizer](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_sanitizer_thread.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_sanitizer_thread.yml)
<br>
[![Test Static Analysis: CppCheck](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_code_static_analysis_cppcheck.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_code_static_analysis_cppcheck.yml)
<br>
[![Test Static Analysis: ClangTidy](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_code_static_analysis_tidy.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_code_static_analysis_tidy.yml)
<br>
[![Test Memory: Valgrind](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_memory_valgrind.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_memory_valgrind.yml)
<br>
[![Test Leak: Sanitizer](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_sanitizer_leak.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_sanitizer_leak.yml)
<br>
[![Test Code Style: ClangFormat](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_code_style_format.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_code_style_format.yml)
<br>
[![Test Address: Sanitizer](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_sanitizer_address.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_sanitizer_address.yml)
<br>
[![Coverage Status](https://coveralls.io/repos/github/FrancoisCarouge/Kalman/badge.svg?branch=develop)](https://coveralls.io/github/FrancoisCarouge/Kalman?branch=develop)
<br>
<br>
[![Test Documentation: Doxygen](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_documentation_doxygen.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_documentation_doxygen.yml)
<br>
[![Public Domain](https://img.shields.io/badge/License-Public%20Domain%20%F0%9F%94%97-brightgreen)](https://raw.githubusercontent.com/francoiscarouge/Kalman/develop/LICENSE.txt)
<br>
[![License Scan](https://app.fossa.com/api/projects/git%2Bgithub.com%2FFrancoisCarouge%2FKalman.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2FFrancoisCarouge%2FKalman?ref=badge_shield)
<br>
[![Sponsor](https://img.shields.io/badge/Sponsor-%EF%BC%84%EF%BC%84%EF%BC%84%20%F0%9F%94%97-brightgreen)](http://paypal.me/francoiscarouge)
<br>
<br>
[![Deploy Documentation: Doxygen GitHub Pages](https://github.com/FrancoisCarouge/Kalman/actions/workflows/deploy_documentation_doxygen.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/deploy_documentation_doxygen.yml)
<br>
[![Deploy Code Coverage: Coveralls](https://github.com/FrancoisCarouge/Kalman/actions/workflows/deploy_test_coverage_coveralls.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/deploy_test_coverage_coveralls.yml)


# Motivation

Kalman filters can be difficult to learn, use, and implement. Users often need fair algebra, domain, and software knowledge. Inadequacy leads to incorrectness, underperformance, and a big ball of mud.

This package explores what could be a Kalman filter implementation a la standard library. The following concerns and tradeoffs are considered:
- Separation of the application domain.
- Separation of the algebra implementation.
- Generalization of the support.

# Usage

## System Installation

Clone the repository, from within the cloned folder, run:

```
mkdir build && cd build && cmake .. && sudo make install
```

# Class kalman

Defined in header [fcarouge/kalman.hpp](include/fcarouge/kalman.hpp)

```cpp
template <
  typename Type,
  typename State,
  typename Output,
  typename Input,
  typename Transpose,
  typename Symmetrize,
  typename Divide,
  typename Identity,
  typename... UpdateArguments,
  typename... PredictionArguments>
class kalman<
  Type,
  State,
  Output,
  Input,
  Transpose,
  Symmetrize,
  Divide,
  Identity,
  std::tuple<UpdateArguments...>,
  std::tuple<PredictionArguments...>>
```

## Template Parameters

| Template Parameter | Definition |
| --- | --- |
| `Type` | The type template parameter of the value type of the filter. |
| `State` | The type template parameter of the state vector x. State variables can be observed (measured), or hidden variables (inferred). This is the the mean of the multivariate Gaussian. |
| `Output` | The type template parameter of the measurement vector z. |
| `Input` | The type template parameter of the control u. |
| `Transpose` | The customization point object template parameter of the matrix transpose functor. |
| `Symmetrize` | The customization point object template parameter of the matrix symmetrization functor. |
| `Divide` | The customization point object template parameter of the matrix division functor. |
| `Identity` | The customization point object template parameter of the matrix identity functor. |
| `UpdateArguments...` | The variadic type template parameter for additional update function parameters. Parameters such as delta times, variances, or linearized values. The parameters are propagated to the function objects used to compute the state observation H and the observation noise R matrices. The parameters are also propagated to the state observation function object h. |
| `PredictionArguments...` | The variadic type template parameter for additional prediction function parameters. Parameters such as delta times, variances, or linearized values. The parameters are propagated to the function objects used to compute the process noise Q, the state transition F, and the control transition G matrices. The parameters are also propagated to the state transition function object f. |

## Member Types

| Member Type | Definition | Dimensions |
| --- | --- | --- |
| `estimate_uncertainty` | Type of the estimated covariance matrix P, also known as Σ. | x by z |
| `gain` | Type of the gain matrix K. | x by z |
| `innovation_uncertainty` | Type of the innovation uncertainty matrix S. | z by z |
| `innovation` | Type of the innovation vector Y. | z by 1 |
| `input_control` | Type of the control transition matrix G, also known as B. | x by u |
| `input` | Type of the control vector U. | u by 1 |
| `output_model` | Type of the observation transition matrix H, also known as C. | z by x |
| `output_uncertainty` | Type of the observation, measurement noise covariance matrix R. | z by z |
| `output` | Type of the observation vector Z, also known as Y or O. | z by 1 |
| `process_uncertainty` | Type of the process noise covariance matrix Q. | x by x |
| `state_transition` | Type of the state transition matrix F, also known as Φ or A. | x by x |
| `state` | Type of the state estimate vector X. | x by 1 |

## Member Functions

| Member Function | Definition |
| --- | --- |
| `(constructor)` | Constructs the filter. |
| `(destructor)` | Destructs the filter. |
| `operator=` | Assigns values to the filter. |

### Characteristics

| Characteristic | Definition |
| --- | --- |
| `f` | Manages the state transition matrix F. Gets the value. Initializes and sets the value. Configures the callable object to compute the value. The default value is the identity matrix. |
| `g` | Manages the control transition matrix G. Gets the value. Initializes and sets the value. Configures the callable object to compute the value. The default value is the identity matrix. |
| `h` | Manages the observation transition matrix H. Gets the value. Initializes and sets the value. Configures the callable object to compute the value. The default value is the identity matrix. |
| `k` | Manages the gain matrix K. Gets the value last computed during the update. The default value is the identity matrix. |
| `p` | Manages the estimated covariance matrix P. Gets the value. Initializes and sets the value. The default value is the identity matrix. |
| `q` | Manages the process noise covariance matrix Q. Gets the value. Initializes and sets the value. Configures the callable object to compute the value. The default value is the null matrix. |
| `r` | Manages the observation, measurement noise covariance matrix R. Gets the value. Initializes and sets the value. Configures the callable object to compute the value. The default value is the null matrix. |
| `s` | Manages the innovation uncertainty matrix S. Gets the value last computed during the update. The default value is the identity matrix. |
| `u` | Manages the control vector U. Gets the value last used in prediction. |
| `x` | Manages the state estimate vector X. Gets the value. Initializes and sets the value. The default value is the null vector. |
| `y` | Manages the innovation vector Y. Gets the value last computed during the update. The default value is the null vector. |
| `z` | Manages the observation vector Z. Gets the value last used during the update. The default value is the null vector. |
| `transition` | Manages the state transition function object f. Configures the callable object to compute the transition state value. The default value is the equivalent to `f(x) = F * X`. The default function is suitable for linear systems. For extended filters `transition` is a linearization of the state transition while F is the Jacobian of the transition function: `F = ∂f/∂X = ∂fj/∂xi` that is each row i contains the derivatives of the state transition function for every element j in the state vector X. |
| `observation` | Manages the state observation function object h. Configures the callable object to compute the observation state value. The default value is the equivalent to `h(x) = H * X`. The default function is suitable for linear systems. For extended filters `observation` is a linearization of the state observation while H is the Jacobian of the observation function: `H = ∂h/∂X = ∂hj/∂xi` that is each row i contains the derivatives of the state observation function for every element j in the state vector X. |

### Modifiers

| Modifier | Definition |
| --- | --- |
| `operator()` | Runs a step of the filter. Predicts and updates the estimates per prediction arguments, control input, and measurement output. |
| `update` | Updates the estimates with the outcome of a measurement. |
| `predict` | Produces estimates of the state variables and uncertainties. |

# Resources

Awesome resources to learn about Kalman filters:

- [KalmanFilter.NET](https://www.kalmanfilter.net) by Alex Becker.
- [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) by Roger Labbe.
- [How Kalman Filters Work](https://www.anuncommonlab.com/articles/how-kalman-filters-work) by Tucker McClure of An Uncommon Lab.
- [Wikipedia Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) by Wikipedia, the free encyclopedia.

# License

<img align="right" src="http://opensource.org/trademarks/opensource/OSI-Approved-License-100x137.png">

Kalman for C++ is public domain:

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