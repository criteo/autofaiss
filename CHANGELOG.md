# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org).

## [2.3.0] - 2021-12-07

### Added
Add make_direct_map option in build index
### Changed
### Deprecated
### Removed
### Fixed
### Security


## [2.2.1] - 2021-11-30

### Added
### Changed
### Deprecated
### Removed
### Fixed
fix shape reading for numpy format
### Security


## [2.2.0] - 2021-11-25

### Added
Add support for Vector Id columns
### Changed
### Deprecated
### Removed
### Fixed
### Security


## [2.1.0] - 2021-11-24

### Added
use fsspec to write the index + make tune index and score index option
### Changed
Decrease memory usage by using a lazy ndarray reader
### Deprecated
### Removed
### Fixed
### Security

## [2.0.0] - 2021-11-19

### Added
* add in memory autofaiss support

### Changed
* improve API by removing the Quantizer class
* quantize -> build_index
* make index_path explicit

## [1.10.1] - 2021-11-19

### Fixed
Add missing fsspec dep in setup.py

## [1.10.0] - 2021-11-18

### Added
Make index creation agnostic of filesystem using fsspec

## [1.9.1] - 2021-09-21

### Added
### Changed
### Deprecated
### Removed
### Fixed
Check if index needs to be trained
### Security

## [1.9.0] - 2021-09-20

### Added
### Changed
* improve estimation of the memory needed for training
### Deprecated
### Removed
### Fixed
### Security

## [1.8.1] - 2021-09-13

### Added
* add pq 256 in list to create large indices
### Changed
### Deprecated
### Removed
### Fixed
### Security

## [1.8.0] - 2021-09-12

### Added
* add pq 128 in list to create large indices
### Changed
### Deprecated
### Removed
### Fixed
### Security

## [1.7.0] - 2021-09-11

### Added
### Changed
* improve memory usage of score command
### Deprecated
### Removed
### Fixed
### Security

## [1.6.2] - 2021-09-11

### Added
### Changed
### Deprecated
### Removed
### Fixed
reserve more space for training
### Security

## [1.6.1] - 2021-09-11

### Added
### Changed
### Deprecated
### Removed
### Fixed
fix small typo train -> build
### Security

## [1.6.0] - 2021-09-11

### Added
### Changed
Improve memory capping and core estimation

* use multiprocessing cpu count to find correct number of cores
* better estimate memory usage and use that to adapt training and adding batch size
### Deprecated
### Removed
### Fixed
### Security

## [1.5.1] - 2021-09-10

### Added
add explicit example in score index doc and in readme
### Changed
### Deprecated
### Removed
### Fixed
fix the batch size of scoring
### Security

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
