
Creating an index
=================

The use-case
------------

You have limited RAM constraint but need to do similarity search on a lot of vectors?
Great! You are in the right place :) This lib automatically builds an optimal index that maximizes the
recall scores given a memory and query speed constraint.

The build_index command
--------------------

The ``autofaiss build_index`` command takes the following parameters:

.. list-table:: Parameters
    :widths: 50 50 100
    :header-rows: 1

    * - Flag available
      - Default
      - Description
    * - --embeddings
      - required
      - Source path of the directory containing your .npy embedding files. If there are several files, they are read in the lexicographical order. This can be a local path or a path in another Filesystem e.g. `hdfs://root/...` or `s3://...`
    * - --index_path
      - required
      - Destination path of the faiss index on local machine.
    * - --index_infos_path
      - required
      - Destination path of the faiss index infos on local machine.
    * - --save_on_disk
      - required
      - Save the index on the disk.
    * - --file_format
      - "npy"
      - File format of the files in embeddings. Can be either `npy` for numpy matrix files or `parquet` for parquet serialized tables
    * - --embedding_column_name
      - "embeddings"
      - Only necessary when file_format=`parquet` In this case this is the name of the column containing the embeddings (one vector per row)
    * - --id_columns
      - None
      - Can only be used when file_format=`parquet`. In this case these are the names of the columns containing the Ids of the vectors, and separate files will be generated to map these ids to indices in the KNN index
    * - --ids_path
      - None
      - Only useful when id_columns is not None and file_format=`parquet`. This will be the path (in any filesystem) where the mapping files Ids->vector index will be store in parquet format
    * - --metric_type
      - "ip"
      - (Optional) Similarity function used for query: ("ip" for inner product, "l2" for euclidian distance)
    * - --max_index_memory_usage
      - "32GB"
      - (Optional) Maximum size in GB of the created index, this bound is strict.
    * - --current_memory_available
      - "32GB"
      - (Optional) Memory available (in GB) on the machine creating the index, having more memory is a boost because it reduces the swipe between RAM and disk.
    * - --max_index_query_time_ms
      - 10
      - (Optional) Bound on the query time for KNN search, this bound is approximative.
    * - --index_key
      - None
      - (Optional) If present, the Faiss index will be build using this description string in the index_factory, more detail in the [Faiss documentation](https://github.com/facebookresearch/faiss/wiki/The-index-factory)
    * - --index_param
      - None
      - (Optional) If present, the Faiss index will be set using this description string of hyperparameters, more detail in the [Faiss documentation](https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning)
    * - --use_gpu
      - False
      - (Optional) Experimental, gpu training can be faster, but this feature is not tested so far.
    * - --nb_cores
      - None
      - (Optional) The number of cores to use, by default will use all cores
      - --make_direct_map
      - False
      - (Optional) If set to True and that the created index is an IVF, call .make_direct_map() on the index to build a mapping (stored on RAM only) that speeds up greatly the calls to .reconstruct().
      - --should_be_memory_mappable
      - False
      - (Optional) If set to true, the created index will be selected only among the indices that can be memory-mapped on disk. This makes it possible to use 50GB indices on a machine with only 1GB of RAM.
    * - --verbose
      - 20
      - (Optional) Set verbosity of logging output: DEBUG=10, INFO=20, WARN=30, ERROR=40, CRITICAL=50

.. _Faiss documentation: https://github.com/facebookresearch/faiss/wiki/The-index-factory

The same function can be called directly from a python environment (from autofaiss import build_index).

It is possible to force the creation of a specific index with specific hyperparameters if more control is needed.
Here is some documentation <https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index> and
<https://github.com/facebookresearch/faiss/wiki/The-index-factory> to help you to choose which index you need.

Time required
-------------

The time required to run this command is:  

* For 1TB of vectors -> 2 hours  
* For 150GB of vectors -> 1 hour  
* For 50GB of vectors -> 20 minutes 

Tuning an existing index
========================

The use-case
------------

You have already created a Faiss index but you would like to have a better recall/query-time ratio?
This command creates a new index with different hyperparameters to be closer to your requirements.

The tune_index command
------------------

The tune_index command set the hyperparameters for the given index.

If an index_param is given, set this hyperparameters to the index,
otherwise perform a greedy heusistic to make the best out or the max_index_query_time_ms constraint

Parameters
----------
index_path : Union[str, Any]
    Path to .index file on local disk if is_local_index_path is True,
    otherwise path on hdfs.
    Can also be an index
index_key: str
    String to give to the index factory in order to create the index.
index_param: Optional(str)
    Optional string with hyperparameters to set to the index.
    If None, the hyper-parameters are chosen based on an heuristic.
output_index_path: str
    Path to the newly created .index file
save_on_disk: bool
    Whether to save the index on disk, default to True.
max_index_query_time_ms: float
    Query speed constraint for the index to create.
use_gpu: bool
    Experimental, gpu training is faster, not tested so far.

Returns
-------
index
    The faiss index

Time required
-------------

The time required to run this command is around 1 minute.

What it does behind
-------------------

The tuning only works for inverted index with HNSW on top of it (95% of indices created by the lib).
there are 3 parameters to tune for that index:

- nprobe:      The number of cells to visit, directly linked to query time (a cell contains on average nb_total_vectors/nb_clusters vectors)
- efSearch:    Search parameter of the HNSW on top of the clusters centers. It has a small impact on search time.
- ht:          The Hamming threshold, adds a boost in speed but reduces the recall.

efSearch is set to be 2 times higher than nprobe, and the Hamming threshold is desactivated by setting it to a high value.

By doing so, we can optimize on only one dimension by applying a binary search given a query time constraint.


Getting scores on an index
==========================

The use-case
------------

You have a faiss index and you would like to know it's 1-recall, intersection recall, query speed, ...?
There is a command for that too, it's the score command.

The score command
-----------------

You just need the path to your index and the embeddings for this one.
Be careful, computing accurate metrics is slow.

Compute metrics on a given index, use cached ground truth for fast scoring the next times.

``autofaiss score_index --embeddings="folder/embs" --index_path="some.index" --output_index_info_path "infos.json" --current_memory_available="4G"``

Parameters
----------
index_path : Union[str, Any]
    Path to .index file. Or in memory index
embeddings: str
    Local path containing all preprocessed vectors and cached files.
output_index_info_path : str
    Path to index infos .json
save_on_disk : bool
    Whether to save on disk
current_memory_available: str
    Memory available on the current machine, having more memory is a boost
    because it reduces the swipe between RAM and disk.


Time required
-------------

The time required to run this command is around 1 hour for 200M vectors of 1280d (1TB).  
If the whole dataset fits in RAM it can be much faster.
