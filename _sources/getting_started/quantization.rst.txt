
Creating an index
=================

The use-case
------------

You have limited RAM constraint but need to do similarity search on a lot of vectors?
Great! You are in the right place :) This lib automatically builds a quantized index that maximizes the
recall scores given a memory and query speed constraint.

The quantize command
--------------------

The ``autofaiss quantize`` command takes the following parameters:

+----------------------------+----------+----------------------------+
| Flag available             | Default  | Description                |
+============================+==========+============================+
| --embeddings_path          | required | Source path of the         |
|                            |          | directory containing your  |
|                            |          | .npy embedding files. If   |
|                            |          | there are several files,   |
|                            |          | they are read in the       |
|                            |          | lexicographical order.     |
+----------------------------+----------+----------------------------+
| --output_path              | required | Destination path of the    |
|                            |          | faiss index on local       |
|                            |          | machine.                   |
+----------------------------+----------+----------------------------+
| --metric_type              | "ip"     | (Optional) Similarity      |
|                            |          | function used for query:   |
|                            |          | ("ip" for inner product,   |
|                            |          | "l2" for euclidian         |
|                            |          | distance)                  |
+----------------------------+----------+----------------------------+
| --max_index_memory_usage   | "32GB"   | (Optional) Maximum size in |
|                            |          | GB of the created index,   |
|                            |          | this bound is strict.      |
+----------------------------+----------+----------------------------+
| --current_memory_available | "32GB"   | (Optional) Memory          |
|                            |          | available (in GB) on the   |
|                            |          | machine creating the       |
|                            |          | index, having more memory  |
|                            |          | is a boost because it      |
|                            |          | reduces the swipe between  |
|                            |          | RAM and disk.              |
+----------------------------+----------+----------------------------+
| --max_index_query_time_ms  | 10       | (Optional) Bound on the    |
|                            |          | query time for KNN search, |
|                            |          | this bound is              |
|                            |          | approximative.             |
+----------------------------+----------+----------------------------+
| --index_key                | None     | (Optional) If present, the |
|                            |          | Faiss index will be build  |
|                            |          | using this description     |
|                            |          | string in the              |
|                            |          | index_factory, more detail |
|                            |          | in the `Faiss              |
|                            |          | documentation`_            |
+----------------------------+----------+----------------------------+
| --index_param              | None     | (Optional) If present, the |
|                            |          | Faiss index will be set    |
|                            |          | using this description     |
|                            |          | string of hyperparameters, |
|                            |          | more detail in the `Faiss  |
|                            |          | docume                     |
|                            |          | ntation <https://github.co |
|                            |          | m/facebookresearch/faiss/w |
|                            |          | iki/Index-IO,-cloning-and- |
|                            |          | hyper-parameter-tuning>`__ |
+----------------------------+----------+----------------------------+
| --use_gpu                  | False    | (Optional) Experimental,   |
|                            |          | gpu training can be        |
|                            |          | faster, but this feature   |
+----------------------------+----------+----------------------------+

.. _Faiss documentation: https://github.com/facebookresearch/faiss/wiki/The-index-factory

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

The tuning command
------------------

The tuning command set the hyperparameters for the given index.

If an index_param is given, set this hyperparameters to the index,
otherwise perform a greedy heusistic to make the best out or the max_index_query_time_ms constraint

Parameters
----------
index_path : str
    Path to .index file on local disk if is_local_index_path is True,
    otherwise path on hdfs.
index_key: str
    String to give to the index factory in order to create the index.
index_param: Optional(str)
    Optional string with hyperparameters to set to the index.
    If None, the hyper-parameters are chosen based on an heuristic.
dest_path: Optional[str]
    Path to the newly created .index file. On local disk if is_local_index_path is True,
    otherwise on hdfs. If None id given, index_path is the destination path.
is_local_index_path: bool
    True if the dest_path and index_path are local path, False if there are hdfs paths.
max_index_query_time_ms: float
    Query speed constraint for the index to create.
use_gpu: bool
    Experimental, gpu training is faster, not tested so far.

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

Parameters
----------
index_path : str
    Path to .index file on local disk if is_local_index_path is True,
    otherwise path on hdfs.
embeddings_path: str
    Local path containing all preprocessed vectors and cached files.
is_local_index_path: bool
    True if the dest_path and index_path are local path, False if there are hdfs paths.
current_memory_available: str
    Memory available on the current machine, having more memory is a boost
    because it reduces the swipe between RAM and disk.


Time required
-------------

The time required to run this command is around 1 hour for 200M vectors of 1280d (1TB).  
If the whole dataset fits in RAM it can be much faster.
