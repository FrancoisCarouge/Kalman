# Defects

Defects found while reviewing and expanding the unit tests. This file records analysis only: what is wrong, where, how it shows, and what it affects. It does not propose changes.

- Reviewed revision: `d80652ace`, library version 0.5.4.
- Each defect was reproduced with `clang++-21 -std=c++26` and `g++-14 -std=c++26` against the Eigen backend or built-in arithmetic types.
- "Filter" names the internal struct deduced by the factory, as listed in `AGENTS.md`.
- Line numbers refer to the reviewed revision.
- Defect numbers are stable across reviews: a resolved or withdrawn defect leaves a gap, and a new defect takes the next number.

| # | Defect | Kind | Shows as |
| --- | --- | --- | --- |
| 11 | A constant-evaluable `ones` or `zeros` specialization breaks `make_one` and `make_zero` | Wrong behavior | Compilation error |

Resolved and removed: 1 (`89cb86053`), 2 (`f3989173b`), 3 (`6b54da0bc`), 4 (`4fded89ea`, a declaration mixing arithmetic types is now rejected), 5 (`f897f39c8`), 7 (`fde518be7`, an lvalue operand is now rejected at the call site), 8 (`59c6e0d19`), 9 (`2e80e5f49`, narrowing element values of vector characteristics are now rejected), 10 (`24a2f52cc`, the named values are constant-initialized or built at the point of use). Withdrawn: 6, the rejection of narrowing values is intended.

## 11. A constant-evaluable `ones` or `zeros` specialization breaks `make_one` and `make_zero`

**Location:** `include/fcarouge/kalman_filter/internal/utility.hpp:801-807` (`make_one`) and the equivalent `make_zero`; the constant `one<Type>` and `zero<Type>` specializations they return, constrained on `Type::Identity()`, `Type::identity()`, `Type::Zero()`, or `Type::zero()`.

**Reproduction:**

```cpp
// A user type without a static identity(), whose one value is a constant
// expression, given through the documented specialization point.
struct gain {
  double value;
};

template <> struct fcarouge::kalman_filter::internal::ones<gain> {
  [[nodiscard]] static constexpr auto operator()() { return gain{1.}; }
};

constexpr gain one{fcarouge::kalman_filter::internal::make_one<gain>()};
// Does not compile.
```

```
utility.hpp:735:17: error: static assertion failed: This type is not
implemented. See compiler message.
```

**Analysis:** `make_one<Type>()` returns the variable `one<Type>` when `ones<Type>` is constant-evaluable, and calls `ones<Type>{}()` otherwise. The constant `one<Type>` specialization exists only for a type with a static `Identity()` or `identity()`. A type whose `ones` specialization is constant-evaluable but which has neither member selects the primary `one<Type>`, the `not_implemented` placeholder. `make_zero` has the same shape.

The specializations of the library's own facades are such types: `ones<typed_matrix<Matrix, ...>>` and `zeros<typed_matrix<Matrix, ...>>` in `support/typed/fcarouge/kalman_typed_linear_algebra.hpp` build the typed matrix from `make_one<Matrix>()`, a constant expression whenever the backend's identity is. The Eigen backends hide the defect, their `Identity()` and `Zero()` not being constant expressions, so the factories take the run-time path.

**Impact:** A filter over typed matrices of a backend with a `constexpr` identity, such as ArduPilot's fixed-size matrix, does not compile; nor does a user type given through `ones` or `zeros` with a constant value, the path the `one` and `zero` documentation recommends.

**Test suite:** `test/constructor/constant_1x1x0.cpp` covers types with a constant `Identity()`; no test specializes `ones` or `zeros` with a constant-evaluable value for a type without one, or uses a backend with a `constexpr` identity under the typed facade.
