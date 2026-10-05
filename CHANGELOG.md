# Changelog

All notable changes to Kalman are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [0.5.3] - 2026-03-16

### Added

- Copy construction and assignment of filters.
- Matplot++ plotting support.
- Apollo lunar module abort guidance system extended filter sample.
- Eigen scalar-by-matrix division solution.

### Changed

- Systematic optional member types and API; generalized and simplified characteristics accesses.
- Simplified factory deducer and gain type; state transition is state independent.
- Filter nomenclature fixes; samples renamed to include their dimensions.
- C++23 static call operators; improved printer decorator.
- TypedLinearAlgebra backend updated; naive implementation made compatible with the typed matrix.
- Legacy benchmarks removed.

### Compiler & Build

- Clang 19 and 20 regression pipelines, upgraded Windows runners, latest CMake, exhaustive CppCheck level.
- Coverage comments supported and secured for pull requests from forked repositories.

### Fixed

- Filter name without update types pack, one/identity type, and a Clang `forward_like` defect workaround.

## [0.5.2] - 2025-05-19

### Added

- An authoritative forward declaration header, `fcarouge/kalman_forward.hpp`.
- TypedLinearAlgebra as an external project backend.

### Changed

- Internal namespace renamed `kalman_internal` for global uniqueness; format, printer, and utility support internalized.
- CMake: globally unique project name, lenient minimum version, standard `BUILD_TESTING` variable, subproject fence.

## [0.5.1] - 2025-05-02

### Fixed

- CMake installation.

## [0.5.0] - 2025-05-02

### Changed

- Code coverage produced by the project's own workflow; asserts forced on in support code.
- Eigen back at the head of its repository.

## [0.4.0] - 2025-02-28

### Added

- `{x, z, p, r}` and `{x, z, p, q, r}` filter configurations, conditional parameter packs, and pack outputs.
- Declarative filter construction, a `kalman_filter` concept, and deducing-`this` support.
- Indexed and quantity linear algebra backends; mp-units compatibility tests.
- Standard formatting of linear algebra types.

### Changed

- Type-safe innovation deduction, generalized gain types, and `one` replacing `identity` for generic correctness.
- Native `std::print` support.

### Compiler & Build

- Ubuntu 24.04, Clang 18, and GCC 14 builds; Windows environment preparation; CMake 3.30 minimum.
- Citation file verification.

### Fixed

- MSVC C2968 recursive alias declaration.

## [0.3.0] - 2024-09-16

### Added

- Modular, optional filter API: input, input control, and output model.
- Move construction.
- `println` utility support.

### Changed

- `std::format` replaces fmtlib.
- Some member types now conditionally present.

### Compiler & Build

- CodeQL, dependency review, and OpenSSF Scorecard workflows; action hashes pinned.

## [0.2.0] - 2023-08-07

### Added

- Linear algebra backend alternatives, including the Eigen backend isolation and a lazy evaluation experiment.
- Benchmarks across linear algebra backends.
- Namespaced CMake target alias, shared configuration, and pkg-config support.

### Compiler & Build

- Matrix builds with GCC and Clang, sanitizers, format verification, and CppCheck.

## [0.1.0] - 2022-11-25

### Added

- Initial release: a generalized, customizable Kalman filter supporting state, output, and input dimensions, extended filters, and additional prediction and update arguments.

[0.5.3]: https://github.com/FrancoisCarouge/Kalman/compare/0.5.2...0.5.3
[0.5.2]: https://github.com/FrancoisCarouge/Kalman/compare/0.5.1...0.5.2
[0.5.1]: https://github.com/FrancoisCarouge/Kalman/compare/0.5.0...0.5.1
[0.5.0]: https://github.com/FrancoisCarouge/Kalman/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/FrancoisCarouge/Kalman/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/FrancoisCarouge/Kalman/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/FrancoisCarouge/Kalman/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/FrancoisCarouge/Kalman/releases/tag/0.1.0
