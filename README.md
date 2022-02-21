# AutoFaiss

[![pypi](https://img.shields.io/pypi/v/autofaiss.svg)](https://pypi.python.org/pypi/autofaiss)
[![ci](https://github.com/criteo/autofaiss/workflows/Continuous%20integration/badge.svg)](https://github.com/criteo/autofaiss/actions?query=workflow%3A%22Continuous+integration%22)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/criteo/autofaiss/blob/master/docs/notebooks/autofaiss_getting_started.ipynb)

**Automatically create Faiss knn indices with the most optimal similarity search parameters.**

It selects the best indexing parameters to achieve the highest recalls given memory and query speed constraints.

## Doc and posts and notebooks

Using [faiss](https://github.com/facebookresearch/faiss) efficient indices, binary search, and heuristics, Autofaiss makes it possible to *automatically* build in 3 hours a large (200 million vectors, 1TB) KNN index in a low amount of memory (15 GB) with latency in milliseconds (10ms).

Get started by running this [colab notebook](https://colab.research.google.com/github/criteo/autofaiss/blob/master/docs/notebooks/autofaiss_getting_started.ipynb), then check the [full documentation](https://criteo.github.io/autofaiss).  
Get some insights on the automatic index selection function with this [colab notebook](https://colab.research.google.com/github/criteo/autofaiss/blob/master/docs/notebooks/autofaiss_index_selection_demo.ipynb).

Then you can check our [multimodal search example](https://colab.research.google.com/github/criteo/autofaiss/blob/master/docs/notebooks/autofaiss_multimodal_search.ipynb) (using OpenAI Clip model).

Read the [medium post](https://medium.com/criteo-engineering/introducing-autofaiss-an-automatic-k-nearest-neighbor-indexing-library-at-scale-c90842005a11) to learn more about it!

## Installation

To install run `pip install autofaiss`

It's probably best to create a virtual env:
``` bash
python -m venv .venv/autofaiss_env
source .venv/autofaiss_env/bin/activate
pip install -U pip
pip install autofaiss
```

## Using autofaiss in python

If you want to use autofaiss directly from python, check the [API documentation](https://criteo.github.io/autofaiss/API/api.html) and the [examples](examples)

In particular you can use autofaiss with on memory or on disk embeddings collections:

### Using in memory numpy arrays

If you have a few embeddings, you can use autofaiss with in memory numpy arrays:

```python
from autofaiss import build_index
import numpy as np

embeddings = np.float32(np.random.rand(100, 512))
index, index_infos = build_index(embeddings, save_on_disk=False)

query = np.float32(np.random.rand(1, 512))
_, I = index.search(query, 1)
print(I)
```

### Using numpy arrays saved as .npy files

If you have many embeddings file, it is preferred to save them on disk as .npy files then use autofaiss like this:

```python
from autofaiss import build_index

build_index(embeddings="embeddings", index_path="my_index_folder/knn.index",
            index_infos_path="my_index_folder/index_infos.json", max_index_memory_usage="4G",
            current_memory_available="4G")
```

## Memory-mapped indices

Faiss makes it possible to use memory-mapped indices. This is useful when you don't need a fast search time (>50ms)
and still want to reduce the memory footprint to the minimum.

We provide the should_be_memory_mappable boolean in build_index function to generate memory-mapped indices only.
Note: Only IVF indices can be memory-mapped in faiss, so the output index will be a IVF index.

To load an index in memory mapping mode, use the following code:
```python
import faiss
faiss.read_index("my_index_folder/knn.index", faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
```

You can have a look to the [examples](examples/memory_mapped_autofaiss.py) to see how to use it.

Technical note: You can create a direct map on IVF indices with index.make_direct_map() (or directly from the
build_index function by passing the make_direct_map boolean). Doing this speeds up a lot
the .reconstruct() method, function that gives you the value of one of your vector given its rank.
However, this mapping will be stored in RAM... We advise you to create your own direct map in a memory-mapped
numpy array and then call .reconstruct_from_offset() with your custom direct_map.

## Using autofaiss with pyspark
Autofaiss allows users to build indices in Spark, you need to do the following steps:

1. Install pyspark by `pip install pyspark`.
2. Prepare your embeddings files.
3. Create a spark session before using `build_index` (optional), if you don't create it, a default session would
    be created with the least configuration.
### Producing N indices
In the distributed mode, you can generate a set of indices with the total memory larger than your current available
memory by setting `nb_indices_to_keep` different from 1.
For example, if you set `nb_indices_to_keep` to 3 and your `index_path` is `knn.index`, you are expected to produce 3
indices at the end of `build_index` with the followings names:
 - `knn.index01`
 - `knn.index02`
 - `knn.index03`

A [concrete example](examples/distributed_autofaiss_n_indices.py) shows how to produce N indices and how to use them.
## Using the command line

Create embeddings
``` python
import os
import numpy as np
embeddings = np.random.rand(1000, 100)
os.mkdir("embeddings")
np.save("embeddings/part1.npy", embeddings)
os.mkdir("my_index_folder")
```

Generate a Knn index
``` bash
autofaiss build_index --embeddings="embeddings" --index_path="my_index_folder/knn.index" --index_infos_path="my_index_folder/index_infos.json" --metric_type="ip"
```

Try the index
``` python
import faiss
import glob
import numpy as np

my_index = faiss.read_index(glob.glob("my_index_folder/*.index")[0])

query_vector = np.float32(np.random.rand(1, 100))
k = 5
distances, indices = my_index.search(query_vector, k)

print(list(zip(distances[0], indices[0])))
```

## How are indices selected ?

To understand better why indices are selected and what are their characteristics, check the [index selection demo](https://colab.research.google.com/github/criteo/autofaiss/blob/master/docs/notebooks/autofaiss_index_selection_demo.ipynb)

## Command quick overview
Quick description of the `autofaiss build_index` command:

*embeddings*        -> Source path of the embeddings in numpy.  
*index_path*                -> Destination path of the created index.
*index_infos_path*          -> Destination path of the index infos.
*save_on_disk*              -> Save the index on the disk.
*metric_type*               -> Similarity distance for the queries.  

*index_key*                 -> (optional) Describe the index to build.  
*index_param*               -> (optional) Describe the hyperparameters of the index.  
*current_memory_available*  -> (optional) Describe the amount of memory available on the machine.  
*use_gpu*                   -> (optional) Whether to use GPU or not (not tested).  

## Command details

The `autofaiss build_index` command takes the following parameters:

| Flag available               |  Default     | Description                                                                                                                                                                                                                                               |
|------------------------------|:------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --embeddings                 | required     | directory (or list of directories) containing your .npy embedding files. If there are several files, they are read in the lexicographical order. This can be a local path or a path in another Filesystem e.g. `hdfs://root/...` or `s3://...`                                                                                                        |
| --index_path                 | required     | Destination path of the faiss index on local machine.                                                                                                                                                                                                     |
| --index_infos_path           | required     | Destination path of the faiss index infos on local machine.                                                                                                                                                                                                     |
| --save_on_disk               | required     | Save the index on the disk.                                                                                                                                                                                                     |
| --file_format                | "npy"        | File format of the files in embeddings Can be either `npy` for numpy matrix files or `parquet` for parquet serialized tables |
| --embedding_column_name      | "embeddings" | Only necessary when file_format=`parquet` In this case this is the name of the column containing the embeddings (one vector per row) |
| --id_columns                 | None         | Can only be used when file_format=`parquet`. In this case these are the names of the columns containing the Ids of the vectors, and separate files will be generated to map these ids to indices in the KNN index |
| --ids_path                   | None         | Only useful when id_columns is not None and file_format=`parquet`. This will be the path (in any filesystem) where the mapping files Ids->vector index will be store in parquet format|
| --metric_type                |   "ip"       | (Optional) Similarity function used for query: ("ip" for inner product, "l2" for euclidian distance)                                                                                                                                                                                                            |
| --max_index_memory_usage     |  "32GB"      | (Optional) Maximum size in GB of the created index, this bound is strict.                                                                                                                        |
| --current_memory_available   |  "32GB"      | (Optional) Memory available (in GB) on the machine creating the index, having more memory is a boost because it reduces the swipe between RAM and disk.                                                                               |
| --max_index_query_time_ms    |    10        | (Optional) Bound on the query time for KNN search, this bound is approximative.                                                                                                                                   |
| --index_key                  |   None       | (Optional) If present, the Faiss index will be build using this description string in the index_factory, more detail in the [Faiss documentation](https://github.com/facebookresearch/faiss/wiki/The-index-factory)
| --index_param                |   None       | (Optional) If present, the Faiss index will be set using this description string of hyperparameters, more detail in the [Faiss documentation](https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning) |
| --use_gpu                    |   False      | (Optional) Experimental, gpu training can be faster, but this feature is not tested so far.                                                                                                                                         |
| --nb_cores                   |   None       | (Optional) The number of cores to use, by default will use all cores                                                                                                                                         |
| --make_direct_map            |   False      | (Optional) Create a direct map allowing reconstruction of embeddings. This is only needed for IVF indices. Note that might increase the RAM usage (approximately 8GB for 1 billion embeddings).                                                                                                                                         |
| --should_be_memory_mappable  |   False      | (Optional) Boolean used to force the index to be selected among indices having an on-disk memory-mapping implementation.                                                                                                                                             |
| --distributed                |   None       | (Optional) If "pyspark", create the index using pyspark. Otherwise, the index is created on your local machine.|
| --temporary_indices_folder   |   "hdfs://root/tmp/distributed_autofaiss_indices"       | (Optional) Folder to save the temporary small indices, only used when distributed = "pyspark" |
| --verbose                    |   20         | (Optional) Set verbosity of logging output: DEBUG=10, INFO=20, WARN=30, ERROR=40, CRITICAL=50 |
| --nb_indices_to_keep         |   1          | (Optional) Number of indices to keep when distributed is "pyspark". |

## Install from source

First, create a virtual env and install dependencies:
```
python3 -m venv .env
source .env/bin/activate
make install
```


`python -m pytest -x -s -v tests -k "test_get_optimal_hyperparameters"` to run a specific test