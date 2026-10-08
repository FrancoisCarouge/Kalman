# AGENTS.md

Guidance for coding agents working in this repository (the Kalman filter library, `git@github.com:FrancoisCarouge/Kalman.git`). This file is the repository root; all paths below are relative to it.

Human contributors: see [CONTRIBUTING.md](CONTRIBUTING.md). AI coding agents: this file.

## AI agent conduct

- **You are the author.** Understand and be able to defend every line, however much a model wrote. A change nobody can explain is not ready.
- **This is a bleeding-edge C++26, concepts-and-templates, class-template-argument-deduction codebase** (the `.clang-tidy` `Checks: '*'` baseline described under Build & test assumes it). Models are weakest exactly here — watch for confident-but-wrong template metaprogramming and overload resolution in `filter_deducer`, and build and test the change yourself rather than trusting CI or the reviewer to catch it.
- **Propose before you build** for anything touching the named-argument vocabulary (`state`, `output`, `estimate_uncertainty`, ...), the `filter_deducer` overload set, the internal filter structs' characteristics, `conditional_member_types`, or the shape of the public `kalman` API: lay out the options and their consequences and get agreement first, rather than implementing one interpretation and letting the reviewer correct it. README's "Selected Tradeoffs" and "Lessons Learned" record decisions already made; read them before re-litigating one.
- **Keep diffs minimal.** No unrequested reformatting, renaming, or opportunistic refactors riding along with an unrelated change. Run `git status --short` before touching anything, and stage named paths rather than `git add -A`/`git add .`.
- **Never commit in your own name.** Every commit must be attributable to a human: its author and committer are the human contributor's own git identity (`user.name`/`user.email`), never an agent, model, or bot identity. Don't pass `--author`, set `GIT_AUTHOR_*`/`GIT_COMMITTER_*`, or edit the git config to name an AI, and don't add AI `Co-Authored-By` trailers; the human whose name is on the commit answers for it.
- **No publicity.** Don't advertise the agent, model, or vendor anywhere in the project's history or contents: no "Generated with …" or "Written by …" footers, tool links, badges, emoji signatures, or other self-promotion in commit messages, PR/issue bodies, review comments, code comments, or documentation. Contributions speak for themselves, under the human contributor's name.

## Build & test

The repository root is the CMake source directory. The canonical loop — the same one CI runs (`.github/workflows/pipeline.yml`) — is:

```shell
cmake -S . -B build -G Ninja
cmake --build build --parallel
ctest --test-dir build --parallel --verbose
```

- Requires CMake ≥ 4.3 (`cmake_minimum_required(VERSION "4.3")`) and a C++26 compiler. Local builds use whatever `cc`/`c++` resolve to on `$PATH` unless you pass `-DCMAKE_CXX_COMPILER=`; a distribution default such as GCC 13 is too old for C++26, so pass e.g. `-DCMAKE_CXX_COMPILER=clang++-21`. CI exercises `clang++-20`, `clang++-21` (Ninja) and `g++-14`, `g++-15` (Unix Makefiles) on `ubuntu-26.04`, MSVC `cl` (generator `Visual Studio 18 2026`) on `windows-2025` in both Debug and Release, and Apple Clang on `macos-26`.
- MSVC can't take `CMAKE_CXX_STANDARD 26` (CMake doesn't record C++26 support for it, so requesting it hard-fails feature checks, including inside fetched dependencies). `support/support.cmake` `unset()`s `CMAKE_CXX_STANDARD` under `if(MSVC)`, and `support/CMakeLists.txt`'s MSVC branch restores the dialect via `/std:c++latest` directly — the root `CMakeLists.txt` stays compiler-agnostic. Replicate that split rather than adding an `if(compiler)` branch at the root.
- Interprocedural optimization is on for every test and sample driver when supported, unless configured with `-D CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF`, as the coverage workflows do: gcov produces no data from link-time optimized objects.
- The `eigen_typed` and `quantity` backends `FetchContent` the [TypedLinearAlgebra](https://github.com/FrancoisCarouge/TypedLinearAlgebra) library at its default branch head (`support/typed/CMakeLists.txt`); an upstream change there can break this build without any change here. Every third-party `FetchContent_Declare` carries `SYSTEM`, so the project's `-Werror` flags never apply to dependency headers.
- To run a single test, use CTest's `-R`/`--tests-regex` against the generated test name (see the naming convention below), e.g. `ctest --test-dir build -R kalman_test_eigen_linalg_addition --output-on-failure`.
- Formatting (enforced by `.github/workflows/format.yml`): `clang-format-22 --Werror -i -style=file` on `.hpp`/`.tpp`/`.cpp`, and [cmakefmt](https://cmakefmt.dev) (`cmakefmt --check .`; `pipx install cmakefmt`, upgrade with `pipx upgrade cmakefmt`) on every `CMakeLists.txt`/`*.cmake`/`*.cmake.in`. There is **no `.clang-format` file** in the repo, so `-style=file` falls back to clang-format's built-in LLVM style (80 columns, 2-space indent). Hand-written layout will not match it — always finish an edit by running `clang-format-22 -i -style=file` (that exact version; other versions disagree on wrapping) and `cmakefmt --in-place <file>` on every file you touched.
- `cmakefmt` reads the repository-root `.cmakefmt.yaml`: 80 columns and 2-space indent (its defaults), `command_case: unchanged` so mixed-case module commands such as `FetchContent_Declare` keep their spelling, and a `commands:` section describing the project's own `pass`/`sample` (`BACKENDS` keyword) and `amalgamate` functions so their calls wrap keyword-aware. Declare any new project-level CMake function there too; `cmakefmt --list-unknown-commands .` must report none. Recursive discovery honours `.gitignore`, so `build/` is skipped.
- `cmakefmt` is deliberately unpinned: CI installs the latest release on every run and the pre-commit hook is a `language: system` hook calling whatever `cmakefmt` is on `$PATH`. Keep your local install current, or a newer release's layout changes will fail CI while passing locally.
- `.clang-tidy` runs `Checks: '*'` with `WarningsAsErrors: '*'` (minus a short denylist) — a very strict baseline; don't casually suppress warnings. `.github/workflows/clang_tidy.yml` configures the build with `clang++-20` (`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`), then `run-clang-tidy -p build` over every `sample|support|test/*.cpp`, sharded 6 ways. The codebase has **no** `NOLINT`, and the `no-nolint` pre-commit hook rejects any — rework the code instead. Specialize standard library templates such as `std::formatter` over a program-defined template (`fcarouge::kalman<Filter>`), never over a concept-constrained parameter, which `cert-dcl58-cpp` rejects. Gotchas:
- `misc-include-cleaner`: every standard library symbol needs a direct include of its own header, even if transitively available. `.clang-tidy`'s `IgnoreHeaders` whitelists only the Eigen/mp-units/Matplot++/`fcarouge/*` facade headers, never the standard library.
- Run it before finishing: `run-clang-tidy-21 -p build path/to/file.cpp` from a `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` build. With the Ninja generator, first build the targets of the files to lint: their compile commands reference C++ module map (`.modmap`) files that only exist after a build.
- No `HeaderFilterRegex` is set, so clang-tidy reports nothing in `include/` headers directly; header defects surface through the test and sample translation units that instantiate them.
- Pre-commit hooks (`.pre-commit-config.yaml`, run in CI by `.github/workflows/pre_commit.yml`): `pre-commit install` once per clone installs both the `pre-commit` and `commit-msg` hook types; `pre-commit run -a` runs everything. Covers repository hygiene (LF line endings, whitespace, case/Windows-name conflicts, large files, merge markers), gitleaks, `clang-format` pinned to v22 and `cmakefmt` (the same checks as `format.yml`), codespell, workflow/Dependabot/`CITATION.cff` schema validation, actionlint (`.github/actionlint.yaml` declares the `ubuntu-26.04` runner label), and local checks for the SPDX Unlicense header, `NOLINT`, and the `[tag] description` commit-message form. zizmor (GitHub Actions security audit) is not configured.
- actionlint lints `run:` scripts with shellcheck only when `shellcheck` is on `$PATH`, and silently skips that check otherwise — CI has it, so install it locally (`apt install shellcheck` or `pip install shellcheck-py`) or a passing local run can still fail CI.
- Sanitizers (`.github/workflows/sanitizer.yml`, `sanitizer_memory.yml`), Valgrind (`valgrind_memory.yml`), coverage, cppcheck, CodeQL, and Doxygen (`WARN_AS_ERROR = FAIL_ON_WARNINGS`) also gate pull requests. To reproduce a sanitizer locally, export the job's `CXXFLAGS`/`LDFLAGS` and run-time `*SAN_OPTIONS` from the workflow matrix before configuring; `debug.sh` in the parent workspace, when present, has the same blocks commented out.
- `cmake --build build --target amalgamate` generates the single-header `build/amalgamate/fcarouge/kalman.h` from `include/fcarouge/kalman.hpp` (`support/amalgamate/`); `.github/workflows/deploy_amalgamate.yml` publishes it.
- Install: `sudo cmake --install build`.

## Architecture

### The named-parameter, deduced-filter pattern

`fcarouge::kalman<Filter>` (`include/fcarouge/kalman.hpp`) is a thin public wrapper around an internal, compile-time-deduced `Filter` implementation type. Users never name `Filter` themselves — they construct `kalman{...}` with a set of tagged, named arguments (`state{}`, `output<T>`, `input<T>`, `estimate_uncertainty{}`, `process_uncertainty{}`, `output_uncertainty{}`, `output_model{}`, `state_transition{}`, `input_control{}`, `transition{}`, `observation{}`, `update_types<...>`, `prediction_types<...>`, defined in `kalman_filter/internal/type.hpp`), and class template argument deduction resolves the concrete type. `include/fcarouge/kalman_forward.hpp` is the authoritative forward declaration header.

The resolution machinery lives in `kalman_filter/internal/factory.hpp`: `filter_deducer<void>::operator()` is overloaded — one overload per supported combination of named arguments — and each overload picks one of the internal filter implementation structs and forwards the converted arguments to its constructor. `deduce_filter<Arguments...>` is the alias that performs this resolution for the `kalman` class template's deduction guide.

The internal implementation structs are named after the characteristics they carry, one header each under `include/fcarouge/kalman_filter/internal/`:

- `x_z_p_r` — minimal filter: state, output, estimate uncertainty, output uncertainty.
- `x_z_p_r_f` — adds a state transition `F`.
- `x_z_p_q_r` — adds process uncertainty `Q`.
- `x_z_u_p_q_r` — adds a control input `U`.
- `x_z_p_q_r_h_f` — adds observation/output model `H` and state transition `F`.
- `x_z_u_p_q_r_h_f_g_us_ps` — full filter with control input `U`, input control `G`, and extra update/prediction argument packs (`Us...`, `Ps...`).
- `x_z_p_qq_rr_f`, `x_z_p_q_r_hh_f_us_ps`, `x_z_p_q_r_hh_ff_ps`, `x_z_u_p_qq_r_ff_gg_ps` — variants where a doubled letter (`qq`, `rr`, `hh`, `ff`, `gg`) marks a characteristic that is a callable (a function of the state and the extra arguments) rather than a fixed matrix, for gain-scheduling, linear parameter varying (LPV), and extended-filter use cases. Callables are stored by value, their types trailing template parameters of the struct deduced by the factory: no type erasure, no allocation, and a member initializer must never capture a sibling member, which would dangle once the filter is copied or moved.

Each struct implements its own `update`/`predict` equations and stores its characteristics; `kalman_filter/internal/kalman.tpp` defines the public `kalman<Filter>` members, which forward to the deduced struct. `kalman_filter::internal::conditional_member_types<Filter>` (the `kalman` base class) exposes member types conditionally, based on what the deduced `Filter` actually supports — so the public API surface of a given `kalman` instantiation varies with how it was configured. Characteristic presence is probed through the `has_*` concepts in `kalman_filter/internal/utility.hpp`.

### Linear algebra backends

The library core is backend-independent: `include/` never includes a backend. 1x1x1 and 1x1x0 (scalar) filters work with vanilla C++ built-ins; anything higher-dimensional needs a linear algebra backend. `support/<backend>/fcarouge/linalg.hpp` is what test/sample code includes (`#include "fcarouge/linalg.hpp"`), and *which* backend's header is picked up is determined purely by which CMake target you link against. Backends wired in by `support/CMakeLists.txt` (gated on `BUILD_TESTING`):

- `eigen` — Eigen3-backed `kalman_linalg_eigen`; also defines `fcarouge/eigen.hpp`.
- `eigen_typed` — Eigen plus TypedLinearAlgebra for compile-time element type safety (`kalman_linalg_eigen_typed`).
- `quantity` — mp-units physical quantities on top of typed Eigen (`kalman_linalg_quantity`).
- `mp_units` — the mp-units dependency and `fcarouge/mp_units.hpp`, linked by every backend-less test and sample (`kalman_unit_mp_units`).
- `typed` — the TypedLinearAlgebra dependency.
- `matplot` — Matplot++ plotting (`kalman_plot`).
- `main` — shared `main()` driver linked into every test and sample executable (`kalman_main`), carrying the `kalman_realtime` object: the real-time verification entry (`realtime.cpp`) and its `not_realtime` opt-out token (`fcarouge/realtime.hpp`).
- `amalgamate` — the single-header generator (not gated on `BUILD_TESTING`).

### Test/sample generation (`support/support.cmake`)

Tests and samples are not hand-declared per backend; two CMake functions generate them from a file name and an optional `BACKENDS` list:

- `pass(NAME [BACKENDS ...])` — compiles `NAME.cpp` once per backend into `kalman_test_<backend>_<name>`, or once into `kalman_test_<name>` without `BACKENDS` (scalar-only, linked against mp-units).
- `sample(NAME [BACKENDS ...])` — the same shape, producing `kalman_sample_<backend>_<name>` / `kalman_sample_<name>`.

Each executable links `kalman`, `kalman_main`, `kalman_support_options` (the warnings-as-errors and hardening flags in `support/CMakeLists.txt`), and the backend target, gets interprocedural optimization when `check_ipo_supported` allows it, and is registered with CTest, optionally wrapped by the `COMMAND` environment variable (how the Valgrind workflows inject `valgrind`). MSVC skips the `quantity` backend and backend-less samples (mp-units incompatibility). Follow that pattern when adding a new test or sample rather than writing a bespoke `add_test`.

Every test and sample is real-time verified by default: built with `-fsanitize=realtime` (the `Real-Time` job of `sanitizer.yml`), `kalman_main` runs the whole program in a real-time context, as if its entry point were `[[clang::nonblocking]]`, so a memory allocation, lock, or blocking input-output call aborts it. An entry point that genuinely cannot be real-time (it formats, prints, or divides by a matrix with the Eigen backend, whose solver queries the processor cache sizes on first use) opts out explicitly: `#include "fcarouge/realtime.hpp"` and declare `const not_realtime opt_out;` as the first statement of its `test`/`sample` lambda. Never opt out to silence a finding in code meant to be real-time. Without the sanitizer the token does nothing.

Test files are minimal: `#include "fcarouge/kalman.hpp"` (and `fcarouge/linalg.hpp` for a backend), then a single `[[maybe_unused]] const auto test{[] { ...; assert(...); return 0; }()};` block inside `namespace fcarouge::test { namespace { ... } }` with a `//! @test` one-liner on top — no test framework, just `<cassert>` run at static-init time. Samples use the same shape in `namespace fcarouge::sample`, with a `//! @example` Doxygen block, and are the primary documentation of how to configure a filter for a use case — check them before designing a new configuration.

File naming: `<subject>_<feature>[_<state>x<output>x<input>].cpp`, where the dimension triple is the filter's state, output, and input sizes (`5x4x3`, `1x1x0` for no input) and a `_unit` suffix marks the mp-units flavor of a sample.

### Decorators

Filters compose with pipe-style decorators, e.g. `kalman{...} | print`, to attach cross-cutting behavior (printing filter activity) without modifying the filter type itself — see `kalman_filter/internal/print.hpp`. A `std::formatter` specialization (`kalman_filter/internal/format.hpp`) prints whichever characteristics a filter has.

### Other directories

- `benchmark/` — benchmark notes and result images; not wired into the CMake build.
- `documentation/` — Doxygen config/theme.
- `cmake/`, `pkgconfig/` — install/`find_package` export files (`fcarouge-kalman-config.cmake.in`, `.pc.in`).
- The root `CMakeLists.txt` only `add_subdirectory`s `pkgconfig`, `sample`, `support`, `test` when `PROJECT_IS_TOP_LEVEL` — so consumers using `FetchContent`/`find_package` only pull in `cmake/` + `include/`.

## Recipe: supporting a new filter configuration

1. **Proposal.** Describe the named-argument combination, the characteristics it implies, and which internal struct it maps to; agree on it first (see AI agent conduct).
2. **Internal struct.** Reuse an existing `x_...hpp` struct when its characteristics fit. Otherwise create `include/fcarouge/kalman_filter/internal/x_<characteristics>.hpp`: verbatim Unlicense SPDX block, include guard `FCAROUGE_KALMAN_FILTER_INTERNAL_X_<CHARACTERISTICS>_HPP`, `namespace fcarouge::kalman_filter::internal`, named per the letter convention above, modeled on the closest existing struct: its member types, characteristics, and `update`/`predict` equations.
3. **Deduction.** Add the `filter_deducer` overload in `factory.hpp` that converts the named arguments and constructs the struct. Keep overloads unambiguous: a new overload must not tie with an existing one for any argument combination already supported.
4. **Install.** List any new header in the `FILE_SET` of `include/CMakeLists.txt`, or it is not installed; nothing checks this list.
5. **Public surface.** If the configuration exposes a new characteristic, extend `conditional_member_types`, the `has_*` concepts, the `std::formatter`, and the README "Member Types"/"Characteristics" tables together.
6. **Tests.** Add `test/<name>.cpp` files and `pass(...)` lines (alphabetical) in `test/CMakeLists.txt`, with `BACKENDS "eigen" "eigen_typed"` for anything above 1x1. Assert the deduced types (`static_assert(std::same_as<...>)`) as well as the values.
7. **Sample** (optional): a `sample/<name>.cpp` wired with a `sample(...)` line, if the configuration is user-facing and illustrative.
8. **Verify.** `cmake --build build --parallel && ctest --test-dir build --parallel`; single test: `ctest --test-dir build -R kalman_test_<backend>_<name> --output-on-failure`. Keep `clang-format-22 --Werror` / `clang-tidy '*'` clean and Doxygen warning-free.

## Completion checklist

Before declaring any change complete:

- The diff contains only intentional changes; pre-existing or untracked work is untouched.
- Touched `.hpp`/`.tpp`/`.cpp` pass `clang-format-22 --Werror -i -style=file`; touched `CMakeLists.txt`/`*.cmake`/`*.cmake.in` pass `cmakefmt --check`.
- `run-clang-tidy-21 -p build path/to/file.cpp` is clean, with no new `NOLINT`.
- New behavior has a test; a bug fix has a regression test that fails without the fix when practical.
- `README.md` and the relevant `@brief`/`@details`/`@see`/`@todo` Doxygen are updated alongside the code they describe; `CHANGELOG.md`'s `[Unreleased]` section records user-visible changes.
- `cmake --build build --parallel && ctest --test-dir build --parallel` passes, with the specific new/changed test names re-run via `-R` first.
- `pre-commit run -a` passes.

## Conventions

- Header-only library: keep the public API under `include/fcarouge/` (`kalman.hpp`, `kalman_forward.hpp`); implementation details belong in `kalman_filter/internal/` and are not part of the public surface.
- Internal implementation details live in `namespace fcarouge::kalman_filter::internal` — the single internal namespace, mirroring the `include/fcarouge/kalman_filter/internal/` directory, and excluded from Doxygen (`EXCLUDE_SYMBOLS`). Never introduce `fcarouge::internal`, `fcarouge::kalman_internal`, or any other parallel internal namespace. The outer name cannot be `kalman`: a namespace may not share its name with the `fcarouge::kalman` class template in the same scope. Refer to internal entities through the `kf` alias (`kf::has_state<Filter>`, `kf::one<matrix>`) — an unqualified `internal::` does not resolve from `fcarouge` scope. Spell out `kalman_filter::internal` only where the alias cannot be used: reopening the namespace, headers of `kalman_filter/internal/` and backend headers of `support/` that do not include `kalman.hpp`, and the alias declaration itself, declared once in `kalman.hpp` and never re-aliased.
- Every source/CMake file carries the Unlicense SPDX header block — copy it verbatim (version/URL match the root `CMakeLists.txt`) into any new file.
- Consumers `find_package` the `fcarouge-kalman` package and link against its namespaced `kalman` target (see INSTALL.md).
- The author writes precise, terminology-careful `@note`/`@todo`/`@warning` Doxygen comments explaining design rationale directly in headers — match that register when editing docs/comments rather than simplifying.
- Public documentation states the contract and its rationale, not the implementation: what the function means, what it accepts or rejects and why, never which internal struct or helper computes it. Implementation notes belong in `kalman_filter::internal` or in a plain `//` comment at the code they explain.
- Blank lines separate groups of statements, not individual statements: includes of the same origin; using-declarations and type aliases; constant declarations; the construction of related objects; a run of updates or predictions; a run of checks.
- Commit messages follow `[tag] short imperative description` — a lowercase bracketed tag naming the area touched (`[filter]`, `[test]`, `[sample]`, `[cicd]`, `[cmake]`, `[documentation]`, `[support]`, `[linalg]`, ...; `git log --oneline` has the established vocabulary). Match it rather than inventing `Category: ...` or `type(scope):` styles; the `commit-message-tag` hook enforces it.
- PR/issue bodies and other GitHub-rendered Markdown use hard line breaks — a newline in the middle of a paragraph renders as `<br>`, so wrapped prose arrives as a ragged column. Write one paragraph per line there.
- No one-letter lowercase locals or parameters in test/sample code. mp-units exports single-letter unit symbols (`m`, `s`, ...) that files pull in with `using`; a same-named local shadows it, which `-Wshadow` and MSVC (C4456–C4459) report as errors. The filter's own characteristic accessors (`x()`, `p()`, `k()`, ...) are the established mathematical notation and are the exception.
