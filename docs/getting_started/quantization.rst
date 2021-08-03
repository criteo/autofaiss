
Creating an index
=================

The use-case
------------

You have limited RAM constraint but need to do similarity search on a lot of vectors?
Great! You are in the right place :) This lib automatically builds a quantized index that maximizes the
recall scores given a memory and query speed constraint.

The quantize command
--------------------

Quick description of the `autofaiss quantize` command:

*embeddings_path*           -> Source path of the embeddings in numpy.  
*output_path*               -> Destination path of the created index.
*metric_type*               -> Similarity distance for the queries.  

*index_key*                 -> (optional) Describe the index to build.  
*index_param*               -> (optional) Describe the hyperparameters of the index.  
*current_memory_available*  -> (optional) Describe the amount of memory available on the machine.  
*use_gpu*                   -> (optional) Whether to use GPU or not (not tested).  

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

You just need the path to your index and the embeddings for this one.
Be careful, computing accurate metrics is slow.


Time required
-------------

The time required to run this command is around 1 hour for 200M vectors of 1280d (1TB).  
If the whole dataset fits in RAM it can be much faster.
