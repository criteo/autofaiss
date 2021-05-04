
Creating an index
=================

The use-case
------------

You have limited RAM constraint but need to do similarity search on a lot of vectors?
Great! You are in the right place :) This lib automatically builds a quantized index that maximizes the
recall scores given a memory and query speed constraint.

The quantize command
--------------------

There is a magic command that creates your Faiss index, you can find an exemple here `examples_commands/quantize.sh` :

.. literalinclude:: ../../examples_commands/quantize.sh
   :language: bash

The main important parameters are:

* *embeddings_path* :         Path on hdfs where to find your embeddings  
* *local_embeddings_path* :   Path on local disk on which keep a copy of you embeddings  
* *output_path* :             Path on hdfs where you want to save your index  
* *metric_type* :             Similarity distance on your vectors (inner product or euclidian distance)  
* *max_index_query_time_ms* : the maximum query speed you want  
* *max_index_memory_usage* :  the maximum RAM usage of you index  

You can also use the notebook `build_index.ipynb` to do this only with python code (section "Getting started with the python library" of this documentation).
 

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

You can have an example here: `examples_commands/tuning.sh`

.. literalinclude:: ../../examples_commands/tuning.sh
   :language: bash

You just need to put the path of your index, the index_key describing the index and a maximum query-time value in milliseconds/query.

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

You can have an example here: `examples_commands/score.sh`

.. literalinclude:: ../../examples_commands/score.sh
   :language: bash

You just need the path to your index and the embeddings for this one.
Becareful, computing accurate metrics is slow.


Time required
-------------

The time required to run this command is around 1 hour for 200M vectors of 1280d (1TB).  
If the whole dataset fits in RAM it can be much faster.
