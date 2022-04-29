# Kalman Filter for C++

A generic Kalman filter.

# Continuous Integration & Deployment Actions

[![Code Repository](https://img.shields.io/badge/Repository-GitHub%20%F0%9F%94%97-brightgreen)](https://github.com/FrancoisCarouge/Kalman)
<br>
[![Test: Ubuntu 20.04 GCC Trunk](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_ubuntu-20-04_gcc-trunk.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_ubuntu-20-04_gcc-trunk.yml)
<br>
[![Test: Windows 2019 MSVC](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_windows-2019_msvc.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_windows-2019_msvc.yml)
<br>
[![Test: Ubuntu 20.04 Clang Trunk](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_ubuntu-20-04_clang-trunk.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_ubuntu-20-04_clang-trunk.yml)
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

# Examples

## One-Dimensional

```cpp
using kalman = fcarouge::kalman<double>;

kalman k;

k.state_x = 60;
k.estimate_uncertainty_p = 225;
k.transition_observation_h = [] { return kalman::observation{ 1 }; };
k.noise_observation_r = [] {
  return kalman::observation_noise_uncertainty{ 25 };
};

k.observe(48.54);
```

### Multi-Dimensional

Two states, one control, using Eigen3 support.

```cpp
using kalman =
    fcarouge::eigen::kalman<double, 2, 1, 1, std::chrono::milliseconds>;

const double gravitational_acceleration{ -9.8 }; // m.s^-2
const std::chrono::milliseconds delta_time{ 250 };
kalman k;

k.state_x = { 0, 0 };
k.estimate_uncertainty_p =
    kalman::estimate_uncertainty{ { 500, 0 }, { 0, 500 } };
k.transition_state_f = [](const std::chrono::milliseconds &delta_time) {
  const auto dt{ std::chrono::duration<double>(delta_time).count() };
  return kalman::state_transition{ { 1, dt }, { 0, 1 } };
};
k.noise_process_q = [](const std::chrono::milliseconds &delta_time) {
  const auto dt{ std::chrono::duration<double>(delta_time).count() };
  return kalman::process_noise_uncertainty{
    { 0.1 * 0.1 * dt * dt * dt * dt / 4, 0.1 * 0.1 * dt * dt * dt / 2 },
    { 0.1 * 0.1 * dt * dt * dt / 2, 0.1 * 0.1 * dt * dt }
  };
};
k.transition_control_g = [](const std::chrono::milliseconds &delta_time) {
  const auto dt{ std::chrono::duration<double>(delta_time).count() };
  return kalman::control{ 0.0313, dt };
};
k.transition_observation_h = [] { return kalman::observation{ { 1, 0 } }; };
k.noise_observation_r = [] {
  return kalman::observation_noise_uncertainty{ 400 };
};

k.predict(delta_time, gravitational_acceleration);
k.observe(-32.40 );
k.predict(delta_time, 39.72);
k.observe(-11.1);
```

# Motivation

Kalman filters can be difficult to learn, use, and implement. Users often need fair algebra, domain, and software knowledge. Inadequacy leads to incorrectness, underperformance, and a big ball of mud.

This package explores what could be a Kalman filter implementation a la standard library. The following concerns and tradeoffs are considered:
- Separation of the application domain.
- Separation of the algebra implementation.
- Generalization of the support.

# Resources

Awesome resources to learn about Kalman filters:

- [KalmanFilter.NET](https://www.kalmanfilter.net) by Alex Becker.
- [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) by Roger Labbe.

# License

<img align="right" src="http://opensource.org/trademarks/opensource/OSI-Approved-License-100x137.png">

Project for C++ is public domain:

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