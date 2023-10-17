# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
Types of changes
* "Added" for new features.
* "Changed" for changes in existing functionality.
* "Deprecated" for soon-to-be removed features.
* "Removed" for now removed features.
* "Fixed" for any bug fixes.
* "Security" in case of vulnerabilities.
-->

## [Unreleased]

### Added

- Added support for Qwen models. ([#15](https://github.com/tomaarsen/attention_sinks/pull/15))

### Changed

- Changed how Attention Sinks are injected into models, allows `attention_sinks` to be integrated with architectures that aren't in `transformers` ([#16](https://github.com/tomaarsen/attention_sinks/pull/16))

## [0.2.3] - 2023-10-10

### Added

- Added support for GPT-J models. ([#13](https://github.com/tomaarsen/attention_sinks/pull/13))

## [0.2.2] - 2023-10-04

### Fixed

- Fix `model.generate` for all model architectures. ([#6](https://github.com/tomaarsen/attention_sinks/pull/6))

## [0.2.1] - 2023-10-03

### Fixed

- Implemented parity between `attention_sinks` and `transformers==4.34.0` for Falcon and Llama.

## [0.2.0] - 2023-10-03

### Added

- Added support for Mistral models. ([#5](https://github.com/tomaarsen/attention_sinks/pull/5))
- Added support for GPT-NeoX/Pythia models. ([#4](https://github.com/tomaarsen/attention_sinks/pull/4))
- Added support for MPT models. ([#3](https://github.com/tomaarsen/attention_sinks/pull/3))

## [0.1.0] - 2023-10-03

### Added

- Added support for Falcon models. ([#2](https://github.com/tomaarsen/attention_sinks/pull/2))

## [0.0.1] - 2023-10-02

### Added

- Implement initial working version.