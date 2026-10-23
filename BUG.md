# Defects

Defects found while reviewing and expanding the unit tests. This file records analysis only: what is wrong, where, how it shows, and what it affects. It does not propose changes.

- Reviewed revision: `a241cc447`, library version 0.5.4.
- Each defect was reproduced with `clang++-21 -std=c++26` and `g++-14 -std=c++26` against the Eigen backend or built-in arithmetic types.
- "Filter" names the internal struct deduced by the factory, as listed in `AGENTS.md`.
- Line numbers refer to the reviewed revision.
- Defect numbers are stable across reviews: a resolved or withdrawn defect leaves a gap, and a new defect takes the next number.

## 12. The characteristic getters return copies

- Where: `include/fcarouge/kalman_filter/internal/kalman.tpp`, the twelve getters `x()`, `z()`, `u()`, `p()`, `q()`, `r()`, `f()`, `g()`, `h()`, `k()`, `s()` and `y()`, each ending in `return std::forward<decltype(self)>(self).filter.<member>;` under a `decltype(auto)` return type (`x()` at line 68 of `2244cec23`).
- What: `decltype(auto)` of an unparenthesized member access deduces the member's declared type, not a reference. Every getter of an lvalue or `const` lvalue filter returns its characteristic by value: `decltype(filter.x())` is the state type, `decltype(filter.p())` the covariance type. The forwarding of `self` has no effect on the result.
- How it shows: each read copies the whole characteristic. `filter.p().at<2, 2>()` copies the covariance to read one element: 4608 bytes of stack for a 24-state filter in double precision, per call. With TypedLinearAlgebra, whose `data()` of an rvalue typed matrix returns an rvalue reference into it, `const auto & storage = filter.x().data();` binds to the storage of a temporary that dies at the end of the statement: later reads see freed stack, without a diagnostic from g++ 15 `-Wall -Wextra`. Found in MRPT's range-bearing tracking comparison, which read a zero state.
- Affects: the cost of every getter call (F37, F38 of the integration findings) and the lifetime of references taken from a getter's result.
- Reproduced: `static_assert(std::is_same_v<decltype(f.x()), Eigen::Vector4d>)` and the same for a `const` filter and `p()` hold with `clang++-21` and `g++-14` on `2244cec23`.

Resolved and removed: 1 (`89cb86053`), 2 (`f3989173b`), 3 (`6b54da0bc`), 4 (`4fded89ea`, a declaration mixing arithmetic types is now rejected), 5 (`f897f39c8`), 7 (`fde518be7`, an lvalue operand is now rejected at the call site), 8 (`59c6e0d19`), 9 (`2e80e5f49`, narrowing element values of vector characteristics are now rejected), 10 (`24a2f52cc`, the named values are constant-initialized or built at the point of use), 11 (`6f3031910`, constant-evaluable factories). Withdrawn: 6, the rejection of narrowing values is intended.
