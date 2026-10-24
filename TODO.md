# To Do

Findings and follow-up work not yet scheduled.

## Embedded use

Found while replacing the Kalman filters of ArduPilot (`FrancoisCarouge/ardupilot`, branch `fcarouge`), and since 2026-10-22 of PX4, MRPT and ROS 2; finding numbers in parentheses refer to the integration's findings table, kept in the owner's workspace. Rechecked against `6ad1d91f1` (2026-10-23): F44 and F46 resolved there and removed, the others still apply.

- Compile cost of `kalman.hpp` with g++ 15 at `-O2`: 2.2 s and 154 MB per translation unit, 1.3 s of it `<print>` and `<format>`, included even when nothing is formatted. ArduPilot confines it to one source file per filter. (F22)
- The freestanding CI job (`freestanding.yml`, `arm-none-eabi -ffreestanding`) compiles but reports no costs: a filter that grows its object (F35), its frames (F38) or its code (F37) passes. Reporting the object size, the largest stack frame (`-fstack-usage`) and the code size of the freestanding samples, failing on an increase, would guard them. (F4)
- A filter stores every intermediate of its last step as a member: gain, innovation and its uncertainty, models, last arguments. A 3-state filter takes 180 bytes where the hand-written one kept 56, a 2-state one 116 where it kept 20, and on PX4 a 1-state filter 100 bytes where it kept 56, a 2-state one 128 where it kept 36, each 3-state EKFGSF yaw model 232 where it kept 48. (F35)
- `update()` evaluates each equation as one expression whose temporaries all live at once, and computes `i - k * h` twice. A 3-state, 2-output double precision update has a 1416-byte stack frame on Cortex-M7, above ArduPilot's 1300-byte limit per function, which the caller cannot reduce. On PX4, a 2-state, 1-output single precision filter's largest frame grows from 120 to 448 bytes and its code from 678 to 2396 bytes at `-Os`; the EKFGSF yaw model update from 224 to 808 bytes. (F38)
- The update and prediction are dense, general expressions. Against the generated sparse code of ArduPilot's EKFGSF yaw estimator, five corrections take 1203 ns instead of 271 ns and five predictions 1019 ns instead of 550 ns (x86-64, g++ 15), with 7392 bytes more flash on Cortex-M7. A benchmark against hand-written code for a small filter would track it. (F37)
- The covariance update is the Joseph form only. ArduPilot's filters use `P -= K * H * P` followed by symmetrization; matching them needs a selectable form. (F8)
- ArduPilot's 24-state EKF needs sequential scalar updates, inhibited states, the innovation and its uncertainty readable before the correction (gating, rejection), a replaceable covariance prediction, and post-update steps (symmetrization, variance limits). A two-stage update, innovation then correction, would cover most. (F7)

## Missing functionality: ArduPilot filters as samples

Filters of ArduPilot that cannot be written as a library filter, only beside it, each a candidate sample: a target for the features it lacks and a regression test once they exist. ArduPilot is GPL 3: a sample is to be written from the filter's published model, not from ArduPilot's source. (F40)

- **Terrain offset estimator**, the smallest. Written as `sample/ekf_1x1x0_terrain_estimation` on branch `sample-terrain`: untyped, with R set before each update, the consistency test computed outside of the filter from `p()`, and the constraint applied through `x()` after the update, each workaround marked `@todo`. One state, the terrain height in m. A rangefinder observation in m and two optical flow observations in rad/s update it in turn, each gated on its innovation test ratio before the correction; after each correction the state is clamped below the vehicle and the variance floored. Lacks: several output types on one filter, a step between the innovation and the correction, a hook after the correction.
- **Beacon position alignment.** Written as `sample/ekf_3x1x0_beacon_position` on branch `sample-beacon`, with the consistency test outside of the filter after an alignment period; the limited covariance correction is a `@todo`. Originally: three states, a position in m, from scalar range observations to beacons at known positions. Each correction is gated on its innovation, except until alignment completes; the diagonal of the covariance correction is limited to the variances; the covariance is symmetrized; the vertical state is replaced between updates by a separate pair of 1-state filters of the beacons' vertical offset, one per height hypothesis, gated except while aligning. Lacks: gating with a bypass, a limited covariance correction, a post-update hook, editing the state between updates.
- **Gimbal attitude EKF.** Written as `sample/ekf_9x1x0_gimbal_attitude` on branch `sample-gimbal`, an error-state filter: untyped outputs with R set per update, the error-state reset applied through `x()` after each update, and the state transition F set with `f()` before each prediction, because no configuration combines an F computed from prediction arguments with an H computed from update arguments (an `hh_ff_us_ps` filter). Originally: Nine error states: angle error in rad, velocity in m/s, delta angle bias in rad. The quaternion is kept outside the filter and the angle error is reset into it after each fusion. Velocity components in m/s and a heading in rad are fused one scalar at a time. Lacks: error-state reset, scalar updates of different output types; a 9-state update takes 4152 bytes of stack in single precision on Cortex-M7.
- **Wind and airspeed scale estimator** (PX4). Three states, the wind north and east in m/s and the true airspeed scale, unitless; scalar fusions of a true airspeed in m/s and a sideslip in rad, each gated before its correction, the covariance diagonal clamped and symmetrized after it, and the covariance predicted at 1 Hz with a faster scale learning during the first five minutes. Needs one filter with two output types.
- **Vision target estimator** (PX4). Position (with a sensor bias and the target's acceleration) and yaw filters fusing delayed vision measurements at their timestamps: the filter rewinds a history of states and covariances to the measurement's time, fuses it, and replays the later accepted measurements (out-of-sequence measurements). Needs a history and a replay of the filter's steps, or the filter's steps as functions of a given state and covariance.
- **24-state navigation EKF.** Scalar fusions of eleven kinds of observations: airspeed, sideslip, drag, magnetic field, declination, yaw, velocity, position, body velocity, optical flow, range beacon. Gains are masked per inhibited state and the active state count changes. Innovation gating, a variance-only mode for diagnostics, covariance clamping and symmetrization after each fusion, and a generated sparse covariance prediction. Lacks: all of the above, besides the stack and time of a 24-state dense update.

The filters the library does host would make samples too, as embedded examples with units, besides the thermal EKF of `sample/ekf_4x1x0_thermal_soaring`: a 2-state position and velocity filter per axis, a 3-state airspeed scale calibration, a bank of five 3-state yaw EKFs, a 2-state height above ground filter.

## Configurations and integration

- No angle arithmetic: the innovation is `z - h(x)` and the correction `x + K y`, so an angle crossing ±π gives a 2π innovation. MRPT's range-bearing tracking filter diverges on the library (22.5 m off after 2000 steps of a target circling the sensor) unless the measured bearing is wrapped around the predicted one before `update()`; PX4's yaw models wrap their state through `x()` after each update. A customization point for the innovation difference and the correction sum would host them, as MRPT's `OnSubtractObservationVectors` does. (F45)

- A configuration without initial `P`, `Q`, `R`, set later, fails with "This requested filter configuration is not yet supported" without listing the accepted configurations. (F23)
- The mp-units `element_caster` specializations are defined in `support/mp_units/fcarouge/mp_units.hpp` and in TypedLinearAlgebra's file of the same path and name: including both redefines them, and putting both directories on an include path makes one shadow the other. (F17)

## Build reproducibility

- `support/typed/CMakeLists.txt` fetches TypedLinearAlgebra without `GIT_TAG`: every build takes its default branch at that moment, and no Kalman version names a compatible TypedLinearAlgebra version. (F12)

## Proposals from the ArduPilot integration

Options for the owner to decide, not implemented: each touches the named-argument vocabulary or the filter configurations. The samples on branches `sample-terrain`, `sample-beacon` and `sample-gimbal` show the workarounds each would remove.

**Consistency test before the correction (F7).** All three samples recompute the innovation and its uncertainty outside the filter to reject an observation.

- A. A two-stage update: `filter.innovate(arguments...)` computes and stores `y` and `s`, readable through the existing accessors, then `filter.correct()` applies them, or the next `innovate` discards them.
- B. A named `consistency{callable(y, s) -> bool}` declaration argument: `update()` skips the correction when it returns false, and records the outcome.
- C. `update()` returning the innovation and its uncertainty with a flag, the correction applied only by a separate call. Closest to ArduPilot's code, most verbose for simple filters.

**Constraint after the correction.** The terrain sample keeps the terrain below the vehicle, the gimbal sample moves its errors into the nominal values and zeroes them.

- A. A named `constraint{callable(x, p)}` declaration argument applied at the end of `update()`: one hook for clamps, symmetrization and error-state resets.
- B. Leave it to the caller through `x()` and `p()`, and document the pattern in a sample, as now.

**Covariance correction form (F8, F37, F38).** The Joseph form is the only one; ArduPilot uses `P - K H P` with symmetrization and, in the beacon alignment, a correction limited to the variances.

- A. A named `covariance_correction<joseph | simple | simple_symmetric>` declaration tag.
- B. A callable `covariance_correction{callable(p, k, h, r) -> p}` for any form, the limited one included.

**Configurations (F23).** No configuration combines a state transition F computed from prediction arguments with an output model H computed from update arguments (an `hh_ff_us_ps` filter); the gimbal sample sets F before each prediction instead. Supporting the combination, or deducing configurations from the present characteristics rather than an enumerated list, would cover it.

**Several output types (F40).** The terrain and gimbal samples observe one state with sensors of different units and are therefore untyped.

- A. `output<Z1, Z2, ...>` with `update<Zi>(arguments...)` selecting the observation, each with its own R, H and h.
- B. One filter per observation type sharing a state and covariance by reference: no new vocabulary, but shared mutable state.

## Proposals from the PX4, MRPT and ROS 2 integrations

Options for the owner to decide, not implemented. The getter return types (F46) and the freestanding build (F44) were decided and implemented on 2026-10-23 (`f8d89c95c`, `6ad1d91f1`).

**Angle arithmetic (F45).** The innovation is `z - h(x)` and the correction `x + K y`: a bearing or yaw crossing ±π gives a 2π innovation (MRPT's range-bearing tracker ends 22.5 m off on the library without a workaround, 1.24 m with it).

- A. Named `innovation{callable(z, z_predicted) -> y}` and `correction{callable(x, dx) -> x}` declaration arguments, defaulting to the subtraction and the sum: MRPT's `OnSubtractObservationVectors` and `OnNormalizeStateVector`.
- B. A customization point on the element types, picked up for an angle quantity type: no vocabulary change, but implicit.
- C. Leave it to the caller, wrapping the measurement around the prediction before `update()` and the state through `x()` after it, and document the pattern in a sample.
