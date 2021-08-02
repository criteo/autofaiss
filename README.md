# AutoFaiss

[![pypi](https://img.shields.io/pypi/v/autofaiss.svg)](https://pypi.python.org/pypi/autofaiss)
[![ci](https://github.com/criteo/autofaiss/workflows/Continuous%20integration/badge.svg)](https://github.com/criteo/autofaiss/actions?query=workflow%3A%22Continuous+integration%22)

**Automatically create Faiss knn indices with the most optimal similarity search parameters.**

It selects the best indexing parameters to achieve the highest recalls given memory and query speed constraints.

Check the full documentation at https://criteo.github.io/autofaiss/

## How to use autofaiss?

To install run `pip install autofaiss`

It's probably best to create a virtual env:
``` bash
python -m venv .venv/autofaiss_env
source .venv/autofaiss_env/bin/activate
pip install -U pip
pip install autofaiss
```


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
autofaiss quantize --embeddings_path="embeddings" --output_path="my_index_folder" --metric_type="ip"
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

# Command quick overview
Quick description of the `autofaiss quantize` command:

*embeddings_path*           -> Source path of the embeddings in numpy.  
*output_path*               -> Destination path of the created index.
*metric_type*               -> Similarity distance for the queries.  

*index_key*                 -> (optional) Describe the index to build.  
*index_param*               -> (optional) Describe the hyperparameters of the index.  
*current_memory_available*  -> (optional) Describe the amount of memory available on the machine.  
*use_gpu*                   -> (optional) Whether to use GPU or not (not tested).  

# Command details

The `autofaiss quantize` command takes the following parameters:

| Flag available             |  Default | Description                                                                                                                                                                                                                                               |
|----------------------------|:--------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --embeddings_path          | required | Source path of the directory containing your .npy embedding files. If there are several files, they are read in the lexicographical order.                                                                                                                |
| --output_path              | required | Destination path of the faiss index on local machine.                                                                                                                                                                                                     |
| --metric_type              |   "ip"   | (Optional) Similarity function used for query: ("ip" for inner product, "l2" for euclidian distance)                                                                                                                                                                                                            |
| --max_index_memory_usage   |  "32GB"  | (Optional) Maximum size in GB of the created index, this bound is strict.                                                                                                                        |
| --current_memory_available |  "32GB"  | (Optional) Memory available (in GB) on the machine creating the index, having more memory is a boost because it reduces the swipe between RAM and disk.                                                                               |
| --max_index_query_time_ms  |    10    | (Optional) Bound on the query time for KNN search, this bound is approximative.                                                                                                                                   |
| --index_key                |   None   | (Optional) If present, the Faiss index will be build using this description string in the index_factory, more detail in the [Faiss documentation](https://github.com/facebookresearch/faiss/wiki/The-index-factory)
| --index_param              |   None   | (Optional) If present, the Faiss index will be set using this description string of hyperparameters, more detail in the [Faiss documentation](https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning) |
| --use_gpu                  |   False  | (Optional) Experimental, gpu training can be faster, but this feature is not tested so far.                                                                                                                                         |

## Install from source

First, create a virtual env and install dependencies:
```
python -m venv .venv/autofaiss_env
source .venv/autofaiss_env/bin/activate
make install
```


