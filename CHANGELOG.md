# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org).

## [2.15.1] - 2022-08-10

 ### Added
 * `autofaiss build_partitioned_indexes` accepts an index key that will be used to create all indexes if defined

## [2.15.0] - 2022-08-01

### Added
* Autofaiss supports the creation of multiple indices from a partitionning column using the command `autofaiss build_partitioned_indexes`.

### Changed
* Changed the training logic to work with smaller functions.

## [2.14.3] - 2022-05-09

### Fixed
Fix the estimated number of batches

## [2.14.2] - 2022-05-06

### Fixed
Fix "Fix the number of output index files

## [2.14.1] - 2022-05-01

### Fixed
* Fix the number of output index files

## [2.14.0] - 2022-03-31

### Added
* Add the possiblity to tune the index to return at least k nearest neighbors

## [2.13.2] - 2022-03-15

### Fixed
* Do not save dataframe index for ids

## [2.13.1] - 2022-03-11

### Fixed
* Fix get_index_size by using NamedTemporaryFile

## [2.13.0] - 2022-03-09

### Changed
* use index.add_with_ids to have consecutive ids in N indices mode (#106)

## [2.12.1] - 2022-03-08

### Fixed
* check if folder exists before removing

## [2.12.0] - 2022-03-08

### Changed
* move optimization in executors in distributed mode

## [2.11.1] - 2022-03-06

### Fixed
* fix the number of batches in merge n indices

## [2.11.0] - 2022-03-06

### Added
* add guide for distributed autofaiss

### Changed
* replace embedding iterator by embedding reader
* produce less indices in distributed mode

### Fixed
* Make max_nb_threads in #94 less than or equal to cpu cores

## [2.10.3] - 2022-03-04

### Fixed
* Read indices by small batch one after the other to avoid out of disk error
* Fix the indices naming for N indices in the README
### Changed
* Save temporary indices directly to the specified temporary files instead of doing a copy first

## [2.10.2] - 2022-02-26

### Fixed
* Fix adding memory estimation and quick fix for training
* Fix docstring of read_total_nb_vectors_and_dim to match return output

## [2.10.1] - 2022-02-25

### Added
Improve the estimation of memory available for adding

## [2.10.0] - 2022-02-25

### Added
Improve training memory estimation
Option to produce N indices in the distributed mode

## [2.9.9] - 2022-02-23

### Fixed
Fix _yield_embeddings_batch to avoid the case where slice_start is equal to slice_end

## [2.9.8] - 2022-02-22

### Fixed
Fix the order of indices when merging

## [2.9.7] - 2022-02-21

### Fixed
fix pex publishing

## [2.9.6] - 2022-02-21

### Changed
Pex building for python 3.6 and 3.8

## [2.9.5] - 2022-02-21

### Added
Add pex building

## [2.9.4] - 2022-02-21

### Fixed
better dependencies ranges

## [2.9.3] - 2022-02-18

### Fixed
Fix/Complete some documents
Disable IVF, Flat index_key for large numbers of vectors on CPU

## [2.9.2] - 2022-02-17

### Fixed
Fix "Filter empty files"

## [2.9.1] - 2022-02-17

### Fixed
* Empty ids path and temporary small indices folder at the beginning

## [2.9.0] - 2022-02-16

### Added
* Use a central logger instead of print functions
* Add a verbosity flag to control the log level

## [2.8.0] - 2022-02-14

### Added
* Filter empty files
* Implement 2 stage merging in the distributed module

## [2.7.1] - 2022-02-11

### Fixed
* Make absolute path so that it is more safer to use fsspec
* Fix memory estimation for inverted list

## [2.7.0] - 2022-02-04

### Added
Add support for multiple embeddings folders

## [2.6.0] - 2022-02-02

### Added
Optional distributed indexing support using pyspark

## [2.5.0] - 2022-01-06

### Added
Add support for memory-mapped indices generation 

## [2.4.1] - 2022-01-04

### Fixed
* Add make direct map to argement of index metadata while estimating index size

## [2.4.0] - 2021-12-16

### Added
* add make_direct_map to memory estimation
### Fixed
* clean ids_batch and ids_batch_df between each batch

## [2.3.0] - 2021-12-07

### Added
Add make_direct_map option in build index

## [2.2.1] - 2021-11-30

### Fixed
fix shape reading for numpy format

## [2.2.0] - 2021-11-25

### Added
Add support for Vector Id columns

## [2.1.0] - 2021-11-24

### Added
use fsspec to write the index + make tune index and score index option
### Changed
Decrease memory usage by using a lazy ndarray reader

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

### Fixed
Check if index needs to be trained

## [1.9.0] - 2021-09-20

### Changed
* improve estimation of the memory needed for training

## [1.8.1] - 2021-09-13

### Added
* add pq 256 in list to create large indices

## [1.8.0] - 2021-09-12

### Added
* add pq 128 in list to create large indices

## [1.7.0] - 2021-09-11

### Changed
* improve memory usage of score command

## [1.6.2] - 2021-09-11

### Fixed
reserve more space for training

## [1.6.1] - 2021-09-11

### Fixed
fix small typo train -> build

## [1.6.0] - 2021-09-11

### Changed
Improve memory capping and core estimation

* use multiprocessing cpu count to find correct number of cores
* better estimate memory usage and use that to adapt training and adding batch size

## [1.5.1] - 2021-09-10

### Added
add explicit example in score index doc and in readme
### Fixed
fix the batch size of scoring

## [1.5.0] - 2021-09-09

### Changed
Make the local numpy iterator memory efficient

## [1.4.0] - 2021-09-08

### Changed
speed improvements

* when estimating shape of numpy files, use memory mapping ; reduce estimation from hours to seconds
* do not keep the training vector in memory after training, reduce memory requirements by a factor 2

## [1.3.2] - 2021-09-03

### Fixed
* fix score function for embeddings to load from files

## [1.3.1] - 2021-09-03

### Changed
* convert embeddings to float32 if needed at loading

## [1.3.0] - 2021-08-09

### Added
* Indices descriptions function
* Indices size estimator
### Changed
* Enhance indices selection function (switch to get_optimal_index_keys_v2 + improvements)
* Update slider notebook

## [1.2.0] - 2021-08-03

### Added
* add doc notebooks
### Changed
* Create the output folder if missing to avoid error
* use 128 instead of 101 (mostly equivalent) in index param selection for ivf

## [1.1.0] - 2021-07-06

### Added
* add embedding_column_name to download

## [1.0.1] - 2021-07-06

### Fixed
* import in download.py

## [1.0.0] - 2021-05-04

### Added
- First release

## [0.1.0] - 2021-04-20

### Added
- Initial commit
