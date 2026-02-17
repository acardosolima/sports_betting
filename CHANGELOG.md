# CHANGELOG

<!-- version list -->

## v0.1.0 (2026-02-17)

### Bug Fixes

- Added token to checkout step ([#18](https://github.com/acardosolima/sports_betting/pull/18),
  [`d6f97c1`](https://github.com/acardosolima/sports_betting/commit/d6f97c1427377ba427d1b7a71d477bf6915bd5e6))

- Changed the order of poetry installation in doc workflow
  ([#17](https://github.com/acardosolima/sports_betting/pull/17),
  [`40e7b99`](https://github.com/acardosolima/sports_betting/commit/40e7b9904b16b8825f8b5b8a708463dbd91226a9))

- Updated python version on workflow files
  ([`90617f4`](https://github.com/acardosolima/sports_betting/commit/90617f481d2831eddc4a804444143475fa0289fd))

### Build System

- Added zero version config ([#17](https://github.com/acardosolima/sports_betting/pull/17),
  [`40e7b99`](https://github.com/acardosolima/sports_betting/commit/40e7b9904b16b8825f8b5b8a708463dbd91226a9))

- Changed release workflow file to use RELEASE_TOKEN
  ([#17](https://github.com/acardosolima/sports_betting/pull/17),
  [`40e7b99`](https://github.com/acardosolima/sports_betting/commit/40e7b9904b16b8825f8b5b8a708463dbd91226a9))

- Configure src/ssa as the main package in pyproject.toml
  ([`77a1dcb`](https://github.com/acardosolima/sports_betting/commit/77a1dcb36746ab7410d21e1c996959bcfc768ff2))

- Created workflow for semantic release
  ([#16](https://github.com/acardosolima/sports_betting/pull/16),
  [`a54ea8c`](https://github.com/acardosolima/sports_betting/commit/a54ea8cab090c7928dcaebecb9935d411bd6b520))

- Divided poetry dependencies in groups
  ([`1c86a05`](https://github.com/acardosolima/sports_betting/commit/1c86a05a70439e396fdf744808e70d40cae4d0dc))

- Divided poetry dependencies in groups
  ([`9bbaef4`](https://github.com/acardosolima/sports_betting/commit/9bbaef4755c907f087e9f86b73370b70e1f991fb))

- Fixed config files to reflect the new poetry setup
  ([`a4686c9`](https://github.com/acardosolima/sports_betting/commit/a4686c92389e71ae36e89cf946104eea0185c8e0))

- Migrate dependency management to poetry
  ([`47f96fa`](https://github.com/acardosolima/sports_betting/commit/47f96fa9e6eca6ca20fecf2f9d996130879aa662))

- Removed old references to pip
  ([`a017967`](https://github.com/acardosolima/sports_betting/commit/a0179678ba6a9ddc8f9802f1a39db1632c5e7b02))

### Chores

- Add mlflow to gitignore
  ([`c13759c`](https://github.com/acardosolima/sports_betting/commit/c13759c55bca61b80021cd3392f8bab9114f3a4a))

- Cleaned execution code on mlflow model manager class that were being used for testing
  ([`0b6c8f6`](https://github.com/acardosolima/sports_betting/commit/0b6c8f64746fb85dc52630703d274a85f3fff412))

- Fixed tensorflow version on requirements file
  ([`1ea3dc1`](https://github.com/acardosolima/sports_betting/commit/1ea3dc194366ab1d1ccc7bf2e45a9618ea8314e1))

- Updated requirements to add tensorflow version
  ([`f137aab`](https://github.com/acardosolima/sports_betting/commit/f137aabf8648d7f7c7c737db8128a70dad1d451b))

### Code Style

- Applied autoformatting
  ([`5d71a90`](https://github.com/acardosolima/sports_betting/commit/5d71a9006f533b0a579b4a28509e5d76dd19861b))

### Continuous Integration

- Changed sphinx build command
  ([`449a4c1`](https://github.com/acardosolima/sports_betting/commit/449a4c13942cc36c262d4c46583671c59ff95aec))

- Fixed documentation workflow
  ([`5cc2c0d`](https://github.com/acardosolima/sports_betting/commit/5cc2c0d265d04d6b6fdece76e797a8f8d70860ff))

- Removed flag -W from sphinx documentation workflow
  ([`7ba97d4`](https://github.com/acardosolima/sports_betting/commit/7ba97d43b6790b455ea6c351fac6726fdbd45c38))

- Setup another way to generate documentation to github pages
  ([`e990424`](https://github.com/acardosolima/sports_betting/commit/e9904242346d8750de771d20818e36ca99b00da6))

### Documentation

- Changed utils.rst text disposition
  ([`2d6fbe1`](https://github.com/acardosolima/sports_betting/commit/2d6fbe1f28c330a7ffcba0bb673f8f2f21041d81))

- Updated rst files to show mlflow model manager info
  ([`9360637`](https://github.com/acardosolima/sports_betting/commit/936063746bff54dd4cc8cee8d9eb6bd400681a4e))

### Features

- Changed log_model param usage from deprecated artifact_path to name
  ([`2c7a116`](https://github.com/acardosolima/sports_betting/commit/2c7a1168cc337777aebb5bd32cbbd08bad76a759))

- Implemented class to manage models with MLFlow
  ([`e8dff40`](https://github.com/acardosolima/sports_betting/commit/e8dff4011871ac8c9560bcf8490199f5154cbcb0))

### Testing

- Added neq test for __str__
  ([`c2e9949`](https://github.com/acardosolima/sports_betting/commit/c2e9949ee564f4ed826684d9592587c3e501fe7b))

- Added testing for mlflow_model_manager designed by AI
  ([`dded5c4`](https://github.com/acardosolima/sports_betting/commit/dded5c40af130a96af0f24c41e415141ec8d04d0))

- Fixed typo
  ([`a994af9`](https://github.com/acardosolima/sports_betting/commit/a994af910c3729b129c98f4e4798a482b6fc17bc))

- Fixed unit testing for mlflow model manager class
  ([`9265fc4`](https://github.com/acardosolima/sports_betting/commit/9265fc47361133ede97129349d46e53e7db77b94))

- Updated imports to use new directory structure
  ([`98337ca`](https://github.com/acardosolima/sports_betting/commit/98337ca2ae6f8a9dcb5348eb260b08cec6c1c8c2))


## v0.0.0 (2025-06-06)

- Initial Release
