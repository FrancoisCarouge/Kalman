# Kalman Filter for C++

A generic Kalman filter.

# Continuous Integration & Deployment Actions

[![Test: Ubuntu 20.04 GCC Trunk](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_ubuntu-20-04_gcc-trunk.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_test_ubuntu-20-04_gcc-trunk.yml)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FFrancoisCarouge%2FKalman.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2FFrancoisCarouge%2FKalman?ref=badge_shield)
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
<br>
[![Test Documentation: Doxygen](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_documentation_doxygen.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/verify_documentation_doxygen.yml)
<br>
<br>
[![Deploy Documentation: Doxygen GitHub Pages](https://github.com/FrancoisCarouge/Kalman/actions/workflows/deploy_documentation_doxygen.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/deploy_documentation_doxygen.yml)
<br>
[![Deploy Code Coverage: Coveralls](https://github.com/FrancoisCarouge/Kalman/actions/workflows/deploy_test_coverage_coveralls.yml/badge.svg)](https://github.com/FrancoisCarouge/Kalman/actions/workflows/deploy_test_coverage_coveralls.yml) Expected due to lack of code.

# Examples

## One-Dimensional

`main.cpp`
```cpp
  using kalman = fcarouge::kalman<float>;

  kalman k;

  k.state = kalman::state_x{ 60 };
  k.estimate_uncertainty = kalman::estimate_uncertainty_p{ 225 };
  k.transition_observation = []() { return kalman::observation_h{ 1 }; };
  k.noise_observation = []() {
    return kalman::measurement_uncertainty_r{ 25 };
  };

  k.update(48.54);
```

### Multi-Dimensional

Two states, one control, using Eigen3 support.

`main.cpp`
```cpp
  using kalman = fcarouge::eigen::kalman<double, 2, 1, 1>;

  kalman k;

  k.state = kalman::state_x{ 0, 0 };
  k.estimate_uncertainty =
      kalman::estimate_uncertainty_p{ { 500, 0 }, { 0, 500 } };
  k.transition_state = []() {
    return kalman::state_transition_f{ { 1, 0.25 }, { 0, 1 } };
  };
  k.noise_process = []() {
    return kalman::process_noise_uncertainty_q{ { 0.000009765625, 0.000078125 },
                                                { 0.000078125, 0.000625 } };
  };
  k.transition_observation = []() { return kalman::observation_h{ { 1, 0 } }; };
  k.noise_observation = []() {
    return kalman::measurement_uncertainty_r{ 400 };
  };
  k.transition_control = []() { return kalman::control_g{ 0.0313, 0.25 }; };

  k.predict(kalman::input_u{ -9.8 });
  k.update(kalman::output_z{ -32.4 });


```

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

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FFrancoisCarouge%2FKalman.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2FFrancoisCarouge%2FKalman?ref=badge_large)