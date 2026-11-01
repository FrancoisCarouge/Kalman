# To Do

Findings and follow-up work not yet scheduled.

## Embedded use

Found while replacing the Kalman filters of ArduPilot (`FrancoisCarouge/ardupilot`, branch `fcarouge`), and since 2026-10-22 of PX4, MRPT and ROS 2; finding numbers in parentheses refer to the integration's findings table, kept in the owner's workspace. Rechecked against `9facffd80` (2026-10-24): F44 and F46 resolved and removed, F23 narrowed, the others still apply.

- Compile cost of `kalman.hpp` with g++ 15 at `-O2`: 2.2 s and 154 MB per translation unit, 1.3 s of it `<print>` and `<format>`, included even when nothing is formatted. ArduPilot confines it to one source file per filter. (F22)
- A filter stores every intermediate of its last step as a member: gain, innovation and its uncertainty, models, last arguments. A 3-state filter takes 180 bytes where the hand-written one kept 56, a 2-state one 116 where it kept 20, and on PX4 a 1-state filter 100 bytes where it kept 56, a 2-state one 128 where it kept 36, each 3-state EKFGSF yaw model 232 where it kept 48. Storing them costs time too: ArduPilot's 2-state precision landing prediction takes 3.3 times its original's time, 2.2 times when F, Q, G, the input and the arguments stay local. (F35)
- `update()` evaluates each equation as one expression whose temporaries all live at once, and computes `i - k * h` twice. A 3-state, 2-output double precision update has a 1416-byte stack frame on Cortex-M7, above ArduPilot's 1300-byte limit per function, which the caller cannot reduce. On PX4, a 2-state, 1-output single precision filter's largest frame grows from 120 to 448 bytes and its code from 678 to 2396 bytes at `-Os`; the EKFGSF yaw model update from 224 to 808 bytes. (F38)
- The update and prediction are dense, general expressions. Against the generated sparse code of ArduPilot's EKFGSF yaw estimator, five corrections take 1203 ns instead of 271 ns and five predictions 1019 ns instead of 550 ns (x86-64, g++ 15), with 7392 bytes more flash on Cortex-M7. A benchmark against hand-written code for a small filter would track it: on the host, PX4's 2-state landing target filter takes 2.2 times its original's time, its EKFGSF models 1.5 times, its 1-state hover thrust filter 1.10; ArduPilot's precision landing filter 3.5 times, its airspeed calibration 1.5, its thermal EKF 1.36. With the simple covariance form and local prediction intermediates, the precision landing filter is still 2.4 times its original. (F37)
- ArduPilot's 24-state EKF needs sequential scalar updates, inhibited states, the innovation and its uncertainty readable before the correction (gating, rejection), a replaceable covariance prediction, and post-update steps (symmetrization, variance limits). A two-stage update, innovation then correction, would cover most. (F7)

## Missing functionality: ArduPilot filters as samples

Filters of ArduPilot that cannot be written as a library filter, only beside it, each a candidate sample: a target for the features it lacks and a regression test once they exist. ArduPilot is GPL 3: a sample is to be written from the filter's published model, not from ArduPilot's source. (F40)

- On `master`: the terrain estimation, beacon position, gimbal attitude, range-bearing tracking and delayed target tracking (the vision target estimator's delayed measurements) samples.
- **Wind and airspeed scale estimator** (PX4). Three states, the wind north and east in m/s and the true airspeed scale, unitless; scalar fusions of a true airspeed in m/s and a sideslip in rad, each gated before its correction, the covariance diagonal clamped and symmetrized after it, and the covariance predicted at 1 Hz with a faster scale learning during the first five minutes. Needs one filter with two output types. Written as `ekf_3x1x0_wind_airspeed_scale` (branch `sample-wind`, a patch for the owner): untyped output, R set per sensor, the gate outside the filter and the scale bound after each update.
- **24-state navigation EKF.** Scalar fusions of eleven kinds of observations: airspeed, sideslip, drag, magnetic field, declination, yaw, velocity, position, body velocity, optical flow, range beacon. Gains are masked per inhibited state and the active state count changes. Innovation gating, a variance-only mode for diagnostics, covariance clamping and symmetrization after each fusion, and a generated sparse covariance prediction. Lacks: all of the above, besides the stack and time of a 24-state dense update. A subset written as `kf_6x1x3_gnss_ins_sequential_fusion` (branch `sample-sequential`, a patch for the owner): six scalar fusions per GNSS measurement with H and R set before each, a per-component gate; the gain mask a `@todo`.

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

**Conditioning policies (F48).** The ports declare their originals' update forms (since `5777b70f6`), and the simple form of a scalar output is associated at a quadratic cost (`e331e1691`); two gaps remain.

- `floors{σ...}` needs assignable elements, which a typed matrix of mp-units quantities cannot give: a fix writing the floors through `at<I, J>(value)` is on branch `floors-setter` (a patch for the owner). ArduPilot's airspeed calibration and both EKFGSF ports keep their floors by hand.
- `symmetrize` and the floors run after each prediction too, where the originals condition after their updates only: an update-only variant would keep ArduPilot's airspeed calibration at its original's cost with the library floors (1.60 times its original's time with them, 1.48 with the hand clamp after the update).

**Configurations (F23).** Since `4ad94d918` and `9facffd80`, the forms combine a callable state transition Jacobian F of the prediction arguments, a callable Q, and a callable output model H of the update arguments (`x_z_p_qq_r_hh_ff_us_ps`), and a filter declares no empty argument lists; PX4's and ArduPilot's EKFGSF ports use them. Still enumerated: a declaration without P, Q and R is rejected, and the rejection does not list the accepted configurations. `AGENTS.md` states that an empty `update_types<>` or `prediction_types<>` is rejected by a `static_assert` in `type.hpp`, which has none: they are still accepted.

**Several output types (F40).** The terrain and gimbal samples observe one state with sensors of different units and are therefore untyped.

- A. `output<Z1, Z2, ...>` with `update<Zi>(arguments...)` selecting the observation, each with its own R, H and h.
- B. One filter per observation type sharing a state and covariance by reference: no new vocabulary, but shared mutable state.

## Proposals from the PX4, MRPT and ROS 2 integrations

Options for the owner to decide, not implemented. The getter return types (F46) and the freestanding build (F44) were decided and implemented on 2026-10-23 (`f8d89c95c`, `6ad1d91f1`).

**Stored intermediates (F35, F37).** Every step stores its models, gain, innovation, innovation uncertainty, input and arguments as members, read back through the getters. The stores cost memory and time: ArduPilot's 2-state precision landing prediction runs in 15 ns, 10 ns with them kept local, against the original's 4.6 ns (host, `-O3`); PX4's EKFGSF takes 1.5 times its original's time, 1.25 with them local, and 534 bytes less code; each of its models keeps 232 bytes where the original kept 48.

- A. A declaration tag, `intermediates<stored | discarded>`: discarded removes the members and their getters, and the steps use locals; the default keeps today's behaviour.
- B. Store only what a later step reads (x, p, and the models or values the user set), and compute the getters of the others on demand from the stored arguments: no new vocabulary, but a getter of a callable characteristic re-evaluates it.
- C. Keep them, and document the cost.

**Angle arithmetic (F45).** The innovation is `z - h(x)` and the correction `x + K y`: a bearing or yaw crossing ±π gives a 2π innovation (MRPT's range-bearing tracker ends 22.5 m off on the library without a workaround, 1.24 m with it).

- A. Named `innovation{callable(z, z_predicted) -> y}` and `correction{callable(x, dx) -> x}` declaration arguments, defaulting to the subtraction and the sum: MRPT's `OnSubtractObservationVectors` and `OnNormalizeStateVector`.
- B. A customization point on the element types, picked up for an angle quantity type: no vocabulary change, but implicit.
- C. Leave it to the caller, wrapping the measurement around the prediction before `update()` and the state through `x()` after it, and document the pattern in a sample.
