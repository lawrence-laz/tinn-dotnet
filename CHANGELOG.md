# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [2.1.0] - 2023-10-08

### Added

- `TinyNeuralNetwork` now has a constructor to provide pre-trained weights and biases.
- `TinyNeuralNetwork` weights and biases can now be accessed via read-only properties `Weights` and `Biases`.

## [2.0.0] - 2023-07-28

### Changed

- `TinyNeuralNetwork.Train` no longer calculates or returns error. To get current error values call `TinyNeuralNetwork.GetTotalError` instead.
- Improved `TinyNeuralNetwork.Load` and `TinyNeuralNetwork.Save` performance.

### Fixed

- Swapped parameters in `LossFunctionPartialDerivative`, this was a bug.
- Saving and loading is now independent of the current culture.
- Reserved 10% of training data for verification in the hand written number recognition example.

## [1.0.0] - 2021-01-18

### Added

- Initial `TinyNeuralNetwork` implementation based on [C implementation].
- Example of a hand written number recognition (MNIST database).

[unreleased]: https://github.com/lawrence-laz/tinn-dotnet/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/lawrence-laz/tinn-dotnet/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/lawrence-laz/tinn-dotnet/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/lawrence-laz/tinn-dotnet/compare/v0.3.0...v1.0.0
[C implementation]: https://github.com/glouw/tinn
