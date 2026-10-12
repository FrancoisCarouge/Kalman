# To Do

Findings and follow-up work not yet scheduled.

## Tests

- Every internal filter struct needs both of its equations tested: its prediction in `test/predict/` and its update in `test/update/`. `x_z_u_p_q_r` had an update test and a deduction test but no prediction test, which let a defective prediction equation go unnoticed until `test/predict/u_1x1x1.cpp` was added with its correction. The other structs were checked for the same gap at `f897f39c8`, and one remains: `x_z_u_p_qq_r_ff_gg_ps` has a prediction test, `test/predict/callable_1x1x1.cpp`, and no update test. `x_z_p_r` has no prediction by design.
- The defects recorded in `BUG.md` each name the test missing from the suite under their "Test suite" heading.
