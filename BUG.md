# Defects

Defects found while reviewing and expanding the unit tests. This file records analysis only: what is wrong, where, how it shows, and what it affects. It does not propose changes.

- Reviewed revision: `0c6e4ad64`, library version 0.5.4 (headers unchanged since `59c6e0d19`).
- Each defect was reproduced with `clang++-21 -std=c++26` and `g++-14 -std=c++26` against the Eigen backend or built-in arithmetic types.
- "Filter" names the internal struct deduced by the factory, as listed in `AGENTS.md`.
- Line numbers refer to the reviewed revision.
- Defect numbers are stable across reviews: a resolved or withdrawn defect leaves a gap, and a new defect takes the next number.

| # | Defect | Kind | Shows as |
| --- | --- | --- | --- |
| 9 | Element values narrow into vector characteristics | Wrong behavior | Silent loss of precision |
| 10 | Named values have unordered dynamic initialization | Wrong behavior | Silent wrong estimate, in programs constructing a filter during static initialization |

Resolved and removed: 1 (`89cb86053`), 2 (`f3989173b`), 3 (`6b54da0bc`), 4 (`4fded89ea`, a declaration mixing arithmetic types is now rejected), 5 (`f897f39c8`), 7 (`fde518be7`, an lvalue operand is now rejected at the call site), 8 (`59c6e0d19`). Withdrawn: 6, the rejection of narrowing values is intended. Of 9, the scalar and declaration paths are resolved; the vector path remains.

## 9. Element values narrow into vector characteristics

**Location:** `include/fcarouge/kalman_filter/internal/kalman.tpp:65`, `:79`, `:92`, `:105`, `:123`, `:144` and the equivalent statements of the other setters; `x_z_p_r.hpp:68` and the output statement of every internal struct.

**Reproduction:**

```cpp
kalman filter{state{column_vector<float, 2>{0.F, 0.F}},
              output<column_vector<float, 2>>, input<float>};
filter.x(1.5, 2.5);      // Accepted: narrows double to float.
filter.update(1.5, 2.5); // Accepted: narrows.
filter.g(1.5, 2.5);      // Accepted: narrows.
filter.update(column_vector<double, 2>{1.5, 2.5}); // Rejected.

kalman double_filter{state{column_vector<double, 2>{0., 0.}}, output<double>};
double_filter.x(1, 2);   // Accepted: converts int to double.
```

None is diagnosed, `-Wconversion` included.

**Analysis:** The library rejects narrowing by intent, and now does so for scalar characteristics and for declarations on both compilers. The element values of a vector characteristic are passed to the backend's variadic constructor, `typename Filter::state{values...}`: list-initialization of a class detects no narrowing, and Eigen's constructor template converts each element itself, inside a system header. A whole vector of another scalar type is rejected, its elements are not.

**Impact:** The same value is rejected for a scalar filter and accepted, with loss of precision, for a vector one.

**Test suite:** No `fail` test sets a vector characteristic from narrowing element values.

## 10. Named values have unordered dynamic initialization

**Location:** `include/fcarouge/kalman_filter/internal/utility.hpp:692`, `:696`, `:711`, `:715`; the `static inline const auto i` identity member of the ten internal structs, such as `x_z_u_p_q_r_h_f_g_us_ps.hpp:70`; the default member initializers that copy `one<>` and `zero<>`, such as `x_z_u_p_q_r_h_f_g_us_ps.hpp:72-82`.

**Reproduction:** three translation units and a matrix type whose `identity()` returns a value, as any backend that is not an expression-template library does.

```cpp
// matrix.hpp
struct matrix2 {
  double data[2][2];
  static matrix2 identity() { return {{{1., 0.}, {0., 1.}}}; }
};
struct holder { // A filter, or an object owning one.
  holder();
  matrix2 model;
};

// filter.cpp
holder::holder() : model{fcarouge::kalman_filter::internal::one<matrix2>} {}

// global.cpp
holder global_holder; // Constructed during static initialization.

// main.cpp
int main() {
  holder local_holder;
  // global_holder.model.data[0][0] is 0 when global.o precedes filter.o on
  // the link line, 1 otherwise. local_holder.model.data[0][0] is 1.
}
```

Both compilers behave the same.

**Analysis:** The four `one<Type>` and `zero<Type>` partial specializations for types with a static `Identity()`, `identity()`, `Zero()`, or `zero()` are `inline` variable templates initialized by a call that is not a constant expression. Each instantiation is a global object with dynamic initialization, guarded, and emitted in every translation unit that instantiates it. The language leaves the dynamic initialization of a templated variable unordered with respect to every other dynamic initialization, of other translation units and of its own. A filter constructed during the dynamic initialization of another object copies whatever the variable holds at that moment: when the translation unit defining that object is initialized first, the copy is of the zero-initialized storage.

A filter copies them in its default member initializers, `h{one<output_model>}`, `f{one<state_transition>}`, `p{one<estimate_uncertainty>}`, and the others, and its identity `i` of the Joseph update is itself a `static inline const` member initialized from `one<>`: one more unordered dynamic initialization, which a filter constructed later still reads in each update. A zero output model `h` makes the gain zero, and the filter never corrects its estimate.

The Eigen backend hides the defect. `Type::Identity()` and `Type::Zero()` return expression objects, `CwiseNullaryOp`, which `auto` deduces and stores unevaluated. For fixed sizes the object has no data: zero-initialized storage evaluates to the same identity or zero matrix. The guard variables and the initialization functions are emitted all the same.

**Impact:** A program that constructs a filter as, or inside, an object of static storage duration may get a filter whose defaults are zero, depending on the link order, without a diagnostic. Observed in ArduPilot, whose vehicles hold their libraries in global objects: its precision landing filter, over a backend whose `identity()` returns a matrix, had a zero output model and landed 2.9 m and 52 m off target in simulation, while its unit tests with local filters passed. The same variables and guards took 360 bytes of RAM for a 4-state filter there, on Cortex-M4. Programs with the Eigen backend are not affected.

**Test suite:** Every test constructs its filters at block scope with the Eigen backend or arithmetic types, which cannot show the defect.
