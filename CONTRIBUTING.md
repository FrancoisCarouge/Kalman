# Contributing to Kalman

Human contributors: this file. AI coding agents: see [AGENTS.md](https://github.com/FrancoisCarouge/Kalman/blob/master/AGENTS.md).

Thank you for taking the time to contribute!

By submitting a pull request or a patch, you represent that you have the right to license your contribution to the project owners and the community, agree that your contributions are licensed under the project license, and agree to future changes to the licensing.

## Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](https://github.com/FrancoisCarouge/Kalman/blob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to francois.carouge@gmail.com.

## Reporting Bugs

Please fill out [the bug template](https://github.com/FrancoisCarouge/Kalman/issues/new/choose), the information it asks for helps us resolve issues faster. Found the fix along with the bug? Feel free to skip the report and open a pull request instead.

## Requesting Features

Please fill out [the feature template](https://github.com/FrancoisCarouge/Kalman/issues/new/choose), the information it asks for helps us provide features faster. Willing to implement it yourself? Pull requests for new features are welcome too.

## Security Policy

Please review [the security policy](https://github.com/FrancoisCarouge/Kalman/security/policy), the process helps us better serve the community.

## Questions & Ideas

Have a question or an idea? Start a [Discussion](https://github.com/FrancoisCarouge/Kalman/discussions) instead.

## Pre-commit Hooks

This repository uses [pre-commit](https://pre-commit.com) to catch formatting and linting issues before they reach a commit. The same hooks run in CI on every pull request, via the `Pre-commit` workflow, but installing them locally catches issues before you push. Secret scanning is enforced separately: the `gitleaks` pre-commit hook only checks staged changes at commit time, so the `Gitleaks` workflow additionally runs [gitleaks/gitleaks-action](https://github.com/gitleaks/gitleaks-action) to scan the full push or pull request diff in CI.

```shell
pip install pre-commit
pipx install cmakefmt
pre-commit install
```

Once installed, the hooks run automatically on `git commit`. To run them against the whole repository at any time:

```shell
pre-commit run --all-files
```

See [.pre-commit-config.yaml](https://github.com/FrancoisCarouge/Kalman/blob/master/.pre-commit-config.yaml) for the exact hook set and versions ([cmakefmt](https://cmakefmt.dev) for CMake formatting, always the latest release from your `$PATH`, so keep it current with `pipx upgrade cmakefmt`, [clang-format](https://clang.llvm.org/docs/ClangFormat.html) v22 for C++ formatting, [gitleaks](https://github.com/gitleaks/gitleaks) for secrets, [codespell](https://github.com/codespell-project/codespell) for spelling, [actionlint](https://github.com/rhysd/actionlint) and [check-jsonschema](https://github.com/python-jsonschema/check-jsonschema) for workflow, Dependabot, and citation validation, local SPDX header, zero `NOLINT`, and `[tag] description` commit message checks, plus line ending, whitespace, and end-of-file fixers).

## Pull Request Merge Checklist

Before merging a pull request, the maintainer:

- Adds any first-time contributor to [CONTRIBUTORS.md](https://github.com/FrancoisCarouge/Kalman/blob/master/CONTRIBUTORS.md), linking their GitHub profile and noting what they contributed.
- Credits the contributor by `@handle` in the [CHANGELOG.md](https://github.com/FrancoisCarouge/Kalman/blob/master/CHANGELOG.md) entry for the specific change, in addition to the standing entry in `CONTRIBUTORS.md`.
