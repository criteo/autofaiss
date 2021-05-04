# AutoFaiss

[![pypi](https://img.shields.io/pypi/v/autofaiss.svg)](https://pypi.python.org/pypi/autofaiss)
[![ci](https://github.com/criteo/autofaiss/workflows/Continuous%20integration/badge.svg)](https://github.com/criteo/autofaiss/actions?query=workflow%3A%22Continuous+integration%22)

Creating and tuning quantized KNN indices for efficient similarity search with low memory usage.

## How to use autofaiss?

To install run `pip install autofaiss`

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

autofaiss quantize

*embeddings_path*     -> path on the hdfs of your embeddings in .parquet format.  
*output_path*         -> destination path on the hdfs for the created index.
*metric_type*         -> Similarity distance for the queries.  

*index_key*           -> (optional) describe the index to build.  
*index_param*         -> (optional) describe the hyperparameters of the index.  
*memory_available*    -> (optional) describe the amount of memory available on the machine.  
*use_gpu*             -> (optional) wether to use GPU or not (not tested).  

## Install from source

First, create a virtual env and install dependencies:
```
python -m venv .venv/autofaiss_env
source .venv/autofaiss_env/bin/activate
make install
```


