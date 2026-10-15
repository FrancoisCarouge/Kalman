# Defects

Defects found while reviewing and expanding the unit tests. This file records analysis only: what is wrong, where, how it shows, and what it affects. It does not propose changes.

- Reviewed revision: `f897f39c8`, library version 0.5.4.
- Each defect was reproduced with `clang++-21 -std=c++26` and `g++-14 -std=c++26` against the Eigen backend or built-in arithmetic types; the diagnostics quoted are Clang's. Where the two compilers disagree, the defect says so.
- "Filter" names the internal struct deduced by the factory, as listed in `AGENTS.md`.
- Line numbers refer to the reviewed revision.
- Defect numbers are stable across reviews: a resolved or withdrawn defect leaves a gap, and a new defect takes the next number.

| # | Defect | Kind | Shows as |
| --- | --- | --- | --- |
| 2 | `printer::update<N>()` has no definition | Missing functionality | Compilation error |
| 3 | Multi-value setters of `Q`, `R`, `F`, `H`, `G` are ill-formed | Missing functionality | Compilation error |
| 4 | Mixed arithmetic types narrow inside the filter equations | Wrong behavior | Compilation error, or silent loss of precision |
| 7 | The `print` decorator rejects lvalue filters | Usability | Compilation error |
| 8 | Constraints stop at the presence of the member | Incorrect interface | Wrong or ill-formed concept answers, misplaced errors |
| 9 | Narrowing values are not rejected uniformly | Wrong behavior | Silent loss of precision |
| 10 | Named values have unordered dynamic initialization | Wrong behavior | Silent wrong estimate, in programs constructing a filter during static initialization |

Resolved since the previous review, and removed from this list:

- Defect 1, the declared output type ignored with `P`, `R`, `F`: `89cb86053` (`[filter] honor declared output type and uncertainty`). The state-transition overload deducing `x_z_p_r_f<X>` is now constrained by `std::same_as<X, Z>`, and a sibling overload deduces `x_z_p_q_r_h_f<X, Z>` otherwise. Covered by `test/member_type/output_type_1x1x0.cpp`, `test/constructor/output_type_1x1x0.cpp`, `test/deduction/x_z_p_q_r_h_f.cpp`, `test/predict/f_2x1x0.cpp`, and `test/update/f_2x1x0.cpp`. The scalar declaration of its reproduction now has the declared types and runs into defect 4.
- Defect 5, the unconstrained constructor, `predict`, and `update`: `f897f39c8`. The three members now carry a `requires` clause, every trait and concept answer of its reproduction is corrected, and the three misuses are rejected at the call site. Covered by `test/constructor/arguments_1x1x0.cpp`, `test/predict/absent_1x1x0.cpp`, and `test/update/absent_1x1x0.cpp`. What the constraints do not reach is recorded as defect 8.

Withdrawn since the previous review:

- Defect 6, the characteristic setters rejecting convertible values. The library rejects narrowing conversions by intent, for safety: passing an `int` to a `double` characteristic, or a `double` to a `float` one, is a usage defect, and the rejection is the correct behavior. The places where a narrowing value is not rejected are recorded as defect 9, the library's own narrowing as defect 4, and the location of the diagnostic as defect 8.

Resolved in earlier reviews:

- The prediction equation depending on the declared types: `828ae39dd` (`[filter] fix optimization filter`), covered by `test/predict/declarations_1x1x1.cpp` and `test/predict/u_1x1x1.cpp`.
- The stored list-of-rows named arguments dangling: `871c5280b` (`[internal] avoid dangling data`), covered by `test/constructor/stored_2x1x0.cpp` and `test/constructor/stored_2x2x1.cpp`.

## 2. `printer::update<N>()` has no definition

**Location:** `include/fcarouge/kalman_filter/internal/print.hpp:80`.

**Reproduction:**

```cpp
auto filter{kalman{state{0.}, output<double>, input<double>,
                   update_types<double>, prediction_types<double>} |
            print};

filter.update(1., 2.);          // Compiles.
(void)filter.predict<0>();      // Compiles.
(void)filter.update<0>();       // Does not compile.
```

```
error: function 'update<0>' with deduced return type cannot be used before it
is defined
```

**Analysis:** The decorator class `kalman_filter::internal::printer<Filter>` declares all four filtering members: `predict(arguments...)`, `predict<Position>()`, `update(arguments...)`, and `update<Position>()`. The header defines the first three. `update<Position>()` is declared with a deduced return type and has no definition anywhere in the library.

The declaration is not inert. `printer` publicly derives from the filter, and its member template hides the base class `kalman::update<Position>()`. Without the declaration the call would resolve to the base member and work, undecorated. With it, name lookup stops at the declaration in `printer` and the call cannot be compiled.

The undecorated filter is unaffected: `kalman::update<0>()` compiles and returns the update argument. Formatting a decorated filter is unaffected as well, although the formatter prints the update arguments, because `std::formatter<printer<Filter>>` inherits the formatter of the decorated filter and reaches `kalman::update<position>()` through a reference to the base.

**Impact:** The last used update arguments cannot be read from a decorated filter. The asymmetry with `predict<Position>()`, which is defined and prints its own event, indicates an omission.

**Test suite:** `test/print/arguments.cpp` covers `predict<N>()` on a decorated filter only.

## 3. Multi-value setters of `Q`, `R`, `F`, `H`, `G` are ill-formed

**Location:** `include/fcarouge/kalman_filter/internal/kalman.tpp:107`, `:123`, `:139`, `:155`, `:171`.

**Reproduction:**

```cpp
kalman filter{state{column_vector<double, 2>{0., 0.}},
              output<column_vector<double, 2>>, input<double>};

filter.x(1., 2.); // Compiles: sets the state from its elements.
filter.g(1., 0.); // Does not compile.
```

```
kalman.tpp:171:24: error: too many template arguments for variable template
'is_convertible_v'
```

**Analysis:** Every characteristic accessor has the signature `(this auto &&self, const auto &...values)`, documented as "the optional copied initializers to set the ... characteristic". The implementations fall in two groups.

- `x`, `z`, `u`, `p`, `k`, `y`, `s` assign `typename Filter::<type>{values...}`. Any number of values is forwarded to the characteristic's constructor.
- `q`, `r`, `f`, `h`, `g` first test `std::is_convertible_v<decltype(values)..., typename Filter::<type>>` to decide between assigning a value and replacing a callable. `std::is_convertible_v` takes exactly two template arguments, so the pack expansion is well-formed only when `values` holds exactly one value.

For two or more values the expression is ill-formed. It is the condition of an `if constexpr` in a function body, not a constraint, so the outcome is a hard error in `kalman.tpp` rather than a discarded branch or a rejected call.

How far this reaches depends on the characteristic's type. For a square matrix the Eigen backend has no element-wise constructor, so `filter.p(1., 0., 0., 1.)` is rejected too, by Eigen (`THIS_METHOD_IS_ONLY_FOR_VECTORS_OF_A_SPECIFIC_SIZE`), and the two groups end up equivalent in practice. For a characteristic that is a vector, the groups differ: `G` of a filter with several states and a single input is a column vector, `H` of a filter with several states and a single output is a row vector, and both would be constructible from their elements exactly as `x` is. Those calls fail in the library before the backend is consulted.

**Impact:** Vector-shaped `G` and `H`, and any backend type of `Q`, `R`, `F`, `H`, `G` constructible from several values, cannot be set from their elements. The two groups of accessors share one documented contract and behave differently.

**Test suite:** `test/x/values_5x4x0.cpp` covers the multi-value form for the state. The tests of `f`, `g`, `h`, `q`, `r` set a single value.

## 4. Mixed arithmetic types narrow inside the filter equations

**Location:** `include/fcarouge/kalman_filter/internal/x_z_p_q_r_h_f.hpp:72`, `:75`, `:76`; `include/fcarouge/kalman_filter/internal/x_z_u_p_q_r_h_f_g_us_ps.hpp:91`, `:93`, `:94`, `:95`, `:103`.

**Reproduction:**

```cpp
kalman filter{state{0.}, output<float>}; // Compiles.

filter.predict();   // Compiles.
filter.update(1.F); // Does not compile.

kalman general_filter{state{0.}, output<float>, input<float>}; // Compiles.

general_filter.predict(1.F); // Compiles.
general_filter.update(1.F);  // Compiles, and narrows double to float.
```

```
x_z_p_q_r_h_f.hpp:72:32: error: non-constant-expression cannot be narrowed from
type 'double' to 'innovation_uncertainty' (aka 'float') in initializer list
```

With `state{0.F}, output<double>` the errors move to lines 75 and 76: `double` to `state`, and `double` to `estimate_uncertainty`.

**Analysis:** The library rejects narrowing conversions by intent, for safety. Here the narrowing is not in the user's call, which passes values of exactly the declared types: it is in the library's own equations, between member types the library computed.

The member types are computed from the declared types by the usual arithmetic conversions: with a `double` state and a `float` output, `output_uncertainty` and `innovation_uncertainty` are `float` (`float * float`), while `output_model` and `estimate_uncertainty` are `double`. The update then evaluates `h * p * t(h) + r`, a `double`, and stores it in the `float` innovation uncertainty. With a `float` state and a `double` output, the same happens to the state and the estimate uncertainty. No declaration of mixed precision avoids it.

The declaration is accepted in every case: the factory deduces the filter, the object is constructed, and its member types exist. What happens next depends on the internal struct.

- **`x_z_p_q_r_h_f`, the filter without input.** The results are stored by list-initialization, `s = innovation_uncertainty{...}`, which forbids the narrowing. The prediction compiles; instantiating `update` fails, in one direction through `s` and in the other through `x` and `p`. A filter that can be declared can never be updated.
- **`x_z_u_p_q_r_h_f_g_us_ps`, the general filter.** The same expressions are stored by plain assignment, `s = h * p * t(h) + r;`, which converts implicitly. The update and the prediction compile and silently narrow: the innovation uncertainty and the innovation at lines 91 and 93 for a `float` output, the state and the estimate uncertainty at lines 94, 95, and 103 for a `float` state. Neither compiler reports it without `-Wconversion`, which the project's own flags do not enable. This is the narrowing the library intends to reject, performed by the library.

The resolution of defect 1 widened the reach of the first case. Declarations naming `P` and `R`, with or without `F`, and an output type different from the state type now deduce `x_z_p_q_r_h_f`, where they previously deduced a single-type struct with the wrong types:

```cpp
kalman filter{state{0.}, output<float>, estimate_uncertainty{1.},
              output_uncertainty{1.F}, state_transition{1.}};

filter.update(1.F); // Does not compile: x_z_p_q_r_h_f.hpp:72.
```

The callable variants taking separate state and output types carry the same list-initialized statements and were not reproduced individually: `x_z_p_qq_rr_f.hpp:82`, `:85`, `:86`; `x_z_p_q_r_hh_f_us_ps.hpp:97`, `:100`, `:101`; `x_z_p_q_r_hh_ff_ps.hpp:92`, `:95`, `:96`; and `x_z_u_p_qq_r_ff_gg_ps.hpp:102`, `:103`, which assigns `s` plainly and list-initializes `x` and `p` only. The remaining structs, `x_z_p_r`, `x_z_p_r_f`, `x_z_p_q_r`, and `x_z_u_p_q_r`, have a single type parameter and cannot hold mixed types.

The first case also depends on the compiler, as described in defect 9: GCC 14 reports the list-initialization as a `-Wnarrowing` warning and narrows.

**Impact:** A filter whose state and output are arithmetic types of different precision either cannot be updated or loses precision silently in its own equations, depending on which internal struct the declaration happens to deduce.

**Test suite:** `test/member_type/1x1x1.cpp` asserts the member types of a mixed-type general filter, and `test/deduction/x_z_p_q_r_h_f.cpp`, `test/member_type/output_type_1x1x0.cpp`, and `test/constructor/output_type_1x1x0.cpp` the deduction, member types, and construction of mixed-type filters without input; no test updates or predicts one.

## 7. The `print` decorator rejects lvalue filters

**Location:** `include/fcarouge/kalman_filter/internal/print.hpp:281-285`.

**Reproduction:**

```cpp
kalman filter;
auto decorated{filter | print}; // Does not compile.
```

```
print.hpp:49:51: error: base specifier must name a class
```

**Analysis:** `operator|` is declared with a forwarding reference, `(Filter &&filter, const printer &)`, and returns `kalman_filter::internal::printer<Filter>`. For an lvalue operand `Filter` is deduced as an lvalue reference type, and `printer<Filter>` attempts to derive from that reference type, which is ill-formed. A `const` filter fails the same way. The decorator only works on a prvalue or an explicitly moved filter, as in `kalman{...} | print` and `std::move(filter) | print`.

The operator is also an unconstrained template in namespace `fcarouge` on its left operand, so the failure is reported at the class template's base specifier rather than at the pipe expression. It matches any left operand, not only a filter: a `requires`-expression probing `value | print` for an lvalue filter, or for an `int`, is a hard error at the same base specifier rather than an unsatisfied requirement.

**Impact:** Minor. A filter built in one statement cannot be decorated in a later one without an explicit move, and the diagnostic does not say so. The documentation of `print` (`kalman.hpp:525`) describes a "pipe decorator to filter declaration", which matches the working form, without stating the restriction.

**Test suite:** `test/print/` and `test/format/printer.cpp` decorate a prvalue filter only.

## 8. Constraints stop at the presence of the member

**Location:** `include/fcarouge/kalman.hpp:396`, `:433` (`predict`, `update`), `:412`, `:449` (`predict<Position>`, `update<Position>`), `:232-364` (accessors); `include/fcarouge/kalman_filter/internal/print.hpp:77`, `:79`.

**Reproduction:**

```cpp
using filter = decltype(kalman{}); // The minimal filter.
using decorated = decltype(kalman{} | print);

template <typename Type, typename... Arguments>
concept can_update = requires(Type value, Arguments... arguments) {
  value.update(arguments...);
};
template <typename Type, typename... Arguments>
concept can_set_state = requires(Type value, Arguments... arguments) {
  value.x(arguments...);
};
template <typename Type>
concept can_read_update = requires(Type value) { value.template update<0>(); };

struct nothing {};

static_assert(can_update<filter, const char *>);    // Holds.
static_assert(can_update<filter, double, double>);  // Holds.
static_assert(can_update<decorated>);               // Holds.
static_assert(not can_set_state<filter, nothing>);  // Error in kalman.tpp:62.
static_assert(not can_read_update<filter>);         // Error in kalman.tpp:245.

filter instance;
instance.update("x");        // Error in x_z_p_r.hpp:65.
instance.update(1., 2.);     // Error in x_z_p_r.hpp:65.
(void)instance.update<0>();  // Error in kalman.tpp:245.
```

**Analysis:** The resolution of defect 5 constrained the constructor, `predict`, and `update` of `kalman`, and the accessors were already constrained on the presence of their characteristic. Each of those constraints answers whether the member exists for the deduced filter and, for `predict` and `update`, whether the argument count fits. None answers whether the arguments are usable. Four cases remain.

- **Argument types of `predict` and `update`.** The constraint is `requires(Filter &implementation) { implementation.update(arguments...); }`. The internal structs declare `update(const auto &output_z, const auto &...outputs_z)` and the equivalent `predict` without constraints, so the expression is valid for any types, and for any number of trailing values when the filter declares no update or prediction types. A string, an unrelated class, or two outputs for a scalar filter satisfy the constraint and fail in the body of the internal struct. A `requires`-expression answers that the call is valid.
- **Values of the accessors.** The accessors return `decltype(auto)`. Naming a call in an unevaluated operand, a `requires`-expression included, instantiates the body to deduce that type, so an invalid value is not an unsatisfied requirement: it is a hard error at `kalman.tpp:62`. A concept probing whether a characteristic can be set from a given type cannot be written.
- **`predict<Position>()` and `update<Position>()`.** Both are unconstrained and return a deduced type. On a filter without prediction or update arguments, the call and any probe of it are a hard error naming an internal member: `no member named 'update_arguments' in '...x_z_p_r<double>'` at `kalman.tpp:245`, and `prediction_arguments` at `:228`. An out-of-range position on a filter that has the arguments is reported inside `<tuple>`.
- **The `print` decorator.** `printer<Filter>` redeclares `predict(const auto &...arguments)` and `update(const auto &...arguments)` without constraints, and those declarations hide the constrained members of the decorated filter. Every trait and concept answer corrected by the resolution of defect 5 is wrong again on a decorated filter: the minimal filter decorated can be predicted and updated without an output, according to a `requires`-expression, and the calls fail at `print.hpp:253` and `:272`.

**Impact:** Generic code cannot ask whether a filter accepts given values: the answer is either an unconditional yes or a compilation error. Misuses of argument and value types are still reported inside the library, as are all misuses of a decorated filter.

**Test suite:** `test/predict/absent_1x1x0.cpp` and `test/update/absent_1x1x0.cpp` assert the presence of the members by argument count, on undecorated filters, with arguments of the declared types.

## 9. Narrowing values are not rejected uniformly

**Location:** `include/fcarouge/kalman_filter/internal/factory.hpp:62-64`; `include/fcarouge/kalman_filter/internal/kalman.tpp:62` and the equivalent statement of each accessor; the output and input statements of every internal struct, such as `x_z_p_r.hpp:65`.

**Reproduction:**

```cpp
kalman filter{state{0.F}};
filter.x(1.);      // Rejected by Clang. Accepted by GCC, with a warning.
filter.update(1.); // Rejected by Clang. Accepted by GCC, with a warning.

kalman vector_filter{state{column_vector<float, 2>{0.F, 0.F}}, output<float>};
vector_filter.x(1.5, 2.5); // Accepted: narrows double to float.

kalman declared_filter{state{0.F}, output<float>, estimate_uncertainty{1.},
                       output_uncertainty{1.}}; // Accepted: narrows.
```

**Analysis:** The library rejects narrowing conversions by intent, for safety: passing a `double` to a `float` characteristic, or an `int` to a `double` one, is a usage defect, and rejecting it is the correct behavior. The setters, `update`, and `predict` implement the rejection by list-initialization, `typename Filter::state{values...}` and `z = output{output_z, outputs_z...}`. Three paths let a narrowing value through.

- **GCC.** A narrowing conversion in list-initialization is ill-formed, and Clang rejects it. GCC 14 reports it from a non-constant expression as a `-Wnarrowing` warning and performs the conversion, so every setter, update, and prediction accepts a narrowing value there unless warnings are errors. The rejection the library relies on is a compilation error on one of the two supported compiler families only.
- **Backend types.** List-initialization only detects a narrowing between arithmetic types. With an Eigen vector state the values reach the backend's variadic constructor, a function template that converts each element itself: `x(1.5, 2.5)` on a `float` vector, or `x(1, 2)` on a `double` one, compiles on both compilers without a diagnostic, `-Wconversion` included, because the conversion happens in a system header. The same call that is rejected for a scalar filter is accepted for a vector one.
- **Declarations.** The factory converts each named argument to the member type through `internal::convert`, which is a functional cast, `To(value)`. An explicit cast is never narrowing in the language's sense and is never diagnosed. An `estimate_uncertainty{1.}`, `output_uncertainty{1.}`, or `state_transition{1.}` declared for a `float` filter is silently converted to `float`, and an integer initializer to a floating-point member, on both compilers.

The first path is a property of the compiler rather than of the library's source, and the library cannot be read as rejecting narrowing on GCC without `-Werror=narrowing` or equivalent. The other two are in the library and its backend: the same value is rejected or accepted depending on the dimension of the filter and on whether it is given at declaration or afterwards.

Where the rejection does happen, the diagnostic names the library's statement rather than the call, for the reason described in defect 8.

**Impact:** The safety property holds for arithmetic characteristics set after construction, with Clang. It does not hold with GCC, for linear algebra characteristics, or for the values of a declaration, where precision is lost without a diagnostic.

**Test suite:** No `fail` test asserts that a narrowing value is rejected by a setter, an update, a prediction, or a declaration.

## 10. Named values have unordered dynamic initialization

**Location:** `include/fcarouge/kalman_filter/internal/utility.hpp:630`, `:634`, `:649`, `:653`; the `static inline const auto i` identity member of the ten internal structs, such as `x_z_u_p_q_r_h_f_g_us_ps.hpp:70`; the default member initializers that copy `one<>` and `zero<>`, such as `x_z_u_p_q_r_h_f_g_us_ps.hpp:72-82`. Unchanged at `59c6e0d19`, where the four definitions are at `utility.hpp:692`, `:696`, `:711`, `:715`.

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
