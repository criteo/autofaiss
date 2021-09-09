# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org).

## [1.5.0] - 2021-09-09

### Added
### Changed
Make the local numpy iterator memory efficient
### Deprecated
### Removed
### Fixed
### Security

## [1.4.0] - 2021-09-08

### Added
### Changed
speed improvements

* when estimating shape of numpy files, use memory mapping ; reduce estimation from hours to seconds
* do not keep the training vector in memory after training, reduce memory requirements by a factor 2
### Deprecated
### Removed
### Fixed
### Security

## [1.3.2] - 2021-09-03

### Added
### Changed
### Deprecated
### Removed
### Fixed
* fix score function for embeddings to load from files
### Security

## [1.3.1] - 2021-09-03

### Added
### Changed
* convert embeddings to float32 if needed at loading
### Deprecated
### Removed
### Fixed
### Security

## [1.3.0] - 2021-08-09

### Added
* Indices descriptions function
* Indices size estimator
### Changed
* Enhance indices selection function (switch to get_optimal_index_keys_v2 + improvements)
* Update slider notebook
### Deprecated
### Removed
### Fixed
### Security

## [1.2.0] - 2021-08-03

### Added
* add doc notebooks
### Changed
* Create the output folder if missing to avoid error
* use 128 instead of 101 (mostly equivalent) in index param selection for ivf
### Deprecated
### Removed
### Fixed
### Security

## [1.1.0] - 2021-07-06

### Added
* add embedding_column_name to download
### Changed
### Deprecated
### Removed
### Fixed
### Security

## [1.0.1] - 2021-07-06

### Added
### Changed
### Deprecated
### Removed
### Fixed
* import in download.py
### Security

## [1.0.0] - 2021-05-04

### Added
- First release
### Changed
### Deprecated
### Removed
### Fixed
### Security

## [0.1.0] - 2021-04-20

### Added
- Initial commit
### Changed
### Deprecated
### Removed
### Fixed
### Security
