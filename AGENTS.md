# Repository Guidelines

## Project Structure & Module Organization
The C++ library lives in `src/` with matching headers in `include/svm_classifier/`. Add new public APIs to the header folder and back them with implementations under `src/`. Tests reside in `tests/` and follow the `test_*.cpp` naming; shared fixtures go in `test_main.cpp`. Examples demonstrating common workflows are under `examples/`, while CMake helpers stay in `cmake/`. Generated artifacts such as `build/`, and coverage outputs should be ignored in commits.

## Build, Test, and Development Commands
Run `./install.sh --build-type Debug` to provision dependencies and create a debug-ready build tree; add `--verbose` when troubleshooting. For manual builds, use `cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_BUILD_TYPE=Release`, then `cmake --build build -j$(nproc)`. Execute `./validate_build.sh --performance` before opening a pull request; it wraps configure, compile, tests, and optional checks. When iterating quickly, drop into `build/` and invoke `ctest --output-on-failure` or run `./svm_classifier_tests "[unit]"` to target a tag.

## Coding Style & Naming Conventions
Stick to the project’s C++17 baseline, four-space indentation, and no tabs. Classes use `PascalCase`, functions and variables `snake_case`, and constants `UPPER_SNAKE_CASE`. Match the existing file naming (`snake_case.cpp/.hpp`). Format changes with `clang-format -i` using the repository configuration; run the `--dry-run --Werror` variant in CI-sensitive branches. Prefer concise Doxygen comments on public interfaces in headers.

## Testing Guidelines
All new behavior needs unit coverage and, when touching training workflows, an integration assertion in `test_svm_classifier.cpp`. Tag Catch2 tests with `[unit]`, `[integration]`, or `[performance]` so targeted runs remain fast. For coverage-sensitive work, rebuild with `cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="--coverage"` and finish with `make -C build-debug coverage`. Capture memory regressions by running `make -C build test_memcheck` if your change adds allocations.

## Commit & Pull Request Guidelines
Use short, imperative commit subjects (e.g., “Improve kernel fallback”) and group related edits logically; the existing history favors single-purpose commits. Branches should read `feature/<topic>` or `fix/<ticket>`. Every PR must describe motivation, summarize tests (`./validate_build.sh --performance`), and link issues when applicable. Include screenshots or logs when altering build scripts or performance metrics, and update docs (`README.md`, `QUICK_START.md`, or `docs/`) as needed.
