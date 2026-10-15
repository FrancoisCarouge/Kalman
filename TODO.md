# To Do

Findings and follow-up work not yet scheduled.

## Tests

- Every internal filter struct needs both of its equations tested: its prediction in `test/predict/` and its update in `test/update/`. `x_z_u_p_q_r` had an update test and a deduction test but no prediction test, which let a defective prediction equation go unnoticed until `test/predict/u_1x1x1.cpp` was added with its correction. The other structs were checked for the same gap at `f897f39c8` and again at `59c6e0d19`, and one remains: `x_z_u_p_qq_r_ff_gg_ps` has a prediction test, `test/predict/callable_1x1x1.cpp`, and no update test. `x_z_p_r` has no prediction by design.
- The defects recorded in `BUG.md` each name the test missing from the suite under their "Test suite" heading.

## Embedded use

Found while replacing the Kalman filters of ArduPilot (`FrancoisCarouge/ardupilot`, branch `fcarouge`, `docs/FCAROUGE_FINDINGS.md`, finding numbers in parentheses). Rechecked against `59c6e0d19`: all still apply.

- `kalman.hpp` always includes `<print>` and `<format>`, through `print.hpp` and `format.hpp`, and defines `print`. A program that never prints pays about 1.3 s of compile time per translation unit, and needs a hosted `<cstdio>`: firmware with a minimal C library builds only because its headers declare the names. Printing and formatting could be opt-in headers. (F2, F22)
- Compile cost of `kalman.hpp` with g++ 15 at `-O2`: 2.2 s and 154 MB per translation unit. ArduPilot confines it to one source file per filter. (F22)
- No CI job builds the samples for a microcontroller: `arm-none-eabi`, `-fno-exceptions -fno-rtti -fsingle-precision-constant -Wdouble-promotion`, with code size and stack use reports. Nothing guards the embedded properties the items below depend on. (F4)
- A filter stores every intermediate of its last step as a member: gain, innovation and its uncertainty, models, last arguments. A 3-state filter takes 180 bytes where the hand-written one kept 56, a 2-state one 116 where it kept 20. (F35)
- `update()` evaluates each equation as one expression whose temporaries all live at once, and computes `i - k * h` twice. A 3-state, 2-output double precision update has a 1416-byte stack frame on Cortex-M7, above ArduPilot's 1300-byte limit per function, which the caller cannot reduce. (F38)
- The update and prediction are dense, general expressions. Against the generated sparse code of ArduPilot's EKFGSF yaw estimator, five corrections take 1203 ns instead of 271 ns and five predictions 1019 ns instead of 550 ns (x86-64, g++ 15), with 7392 bytes more flash on Cortex-M7. A benchmark against hand-written code for a small filter would track it. (F37)
- The covariance update is the Joseph form only. ArduPilot's filters use `P -= K * H * P` followed by symmetrization; matching them needs a selectable form. (F8)
- ArduPilot's 24-state EKF needs sequential scalar updates, inhibited states, the innovation and its uncertainty readable before the correction (gating, rejection), a replaceable covariance prediction, and post-update steps (symmetrization, variance limits). A two-stage update, innovation then correction, would cover most. (F7)
- The `one<>` and `zero<>` variables cost code and RAM besides defect 10 of `BUG.md`: 4072 bytes of code for a 4-state filter on Cortex-M4, against 1040 bytes for the hand-written one it replaced. (F21)

## Missing functionality: ArduPilot filters as samples

Filters of ArduPilot that cannot be written as a library filter, only beside it, each a candidate sample: a target for the features it lacks and a regression test once they exist. ArduPilot is GPL 3: a sample is to be written from the filter's published model, not from ArduPilot's source. (F40)

- **Terrain offset estimator**, the smallest. One state, the terrain height in m. A rangefinder observation in m and two optical flow observations in rad/s update it in turn, each gated on its innovation test ratio before the correction; after each correction the state is clamped below the vehicle and the variance floored. Lacks: several output types on one filter, a step between the innovation and the correction, a hook after the correction.
- **Gimbal attitude EKF.** Nine error states: angle error in rad, velocity in m/s, delta angle bias in rad. The quaternion is kept outside the filter and the angle error is reset into it after each fusion. Velocity components in m/s and a heading in rad are fused one scalar at a time. Lacks: error-state reset, scalar updates of different output types; a 9-state update takes 4152 bytes of stack in single precision on Cortex-M7.
- **24-state navigation EKF.** Scalar fusions of eleven kinds of observations: airspeed, sideslip, drag, magnetic field, declination, yaw, velocity, position, body velocity, optical flow, range beacon. Gains are masked per inhibited state and the active state count changes. Innovation gating, a variance-only mode for diagnostics, covariance clamping and symmetrization after each fusion, and a generated sparse covariance prediction. Lacks: all of the above, besides the stack and time of a 24-state dense update.

The filters the library does host would make samples too, as embedded examples with units, besides the thermal EKF of `sample/ekf_4x1x0_thermal_soaring`: a 2-state position and velocity filter per axis, a 3-state airspeed scale calibration, a bank of five 3-state yaw EKFs, a 2-state height above ground filter.

## Usability

- A configuration without initial `P`, `Q`, `R`, set later, fails with "This requested filter configuration is not yet supported" without listing the accepted configurations. (F23)
- Spelling the matrix types of a filter, covariance, output model, output uncertainty, requires internal names: `evaluate`, `quotient`, `product`, and `ᴀʙᵀ`, a non-ASCII identifier many code bases reject. Public ASCII aliases would do. (F24)
- Typed matrices require specializing internal customization points, `evaluates`, `transposes`, `one`, `zero`, which exist only in the `support/eigen_typed` and `support/quantity` facades. A public, installed integration header would do. (F16)
- The backend contract includes divisions used only to form types, `state / state` and `scalar / state`, which a backend must implement in full because the division's return type is deduced. Undocumented. (F18)
- The mp-units `element_caster` specializations are defined in `support/quantity/fcarouge/linalg.hpp` and in TypedLinearAlgebra's `support/mp_units/fcarouge/mp_units.hpp`: including both redefines them. (F17)

## Releases

- `support/typed/CMakeLists.txt` fetches TypedLinearAlgebra without `GIT_TAG`: every build takes its default branch at that moment, and no Kalman version names a compatible TypedLinearAlgebra version. (F12)
- The headers declare 0.5.4 and the latest tag is `0.5.3`, from 2026-03-16. A consumer pinning a tag gets seven months old code. (F9, F11)
