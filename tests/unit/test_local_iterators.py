import numpy as np
import random
from autofaiss.datasets.readers.local_iterators import read_embeddings_local
import shutil
import os


def build_test_collection(min_size=2, max_size=10000, dim=512, nb_files=5):
    tmp_dir = "/tmp/autofaiss_test"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    sizes = [random.randint(min_size, max_size) for _ in range(nb_files)]
    dim = dim
    for i, size in enumerate(sizes):
        arr = np.random.rand(size, dim).astype("float32")
        np.save(str(tmp_dir) + "/" + str(i) + ".npy", arr)
    return tmp_dir, sizes, dim


def test_read_embeddings_local():
    tmp_dir, sizes, dim = build_test_collection(min_size=2, max_size=12547, dim=512, nb_files=37)
    filenames = sorted([str(tmp_dir) + "/" + str(i) + ".npy" for i in range(len(sizes))])
    ref = np.concatenate([np.load(f) for f in filenames])
    batch_size = 62
    it = read_embeddings_local(tmp_dir, batch_size=batch_size)
    total_expected = sum(sizes)
    last = total_expected // batch_size
    last_expected_batch = total_expected - last * batch_size
    assert ref.shape[0] == total_expected

    total_found = 0
    current = 0
    for i, batch in enumerate(it):
        if last == i:
            expected_batch_size = last_expected_batch
        else:
            expected_batch_size = batch_size
        assert batch.shape[0] == expected_batch_size
        assert batch.shape[1] == dim
        np.testing.assert_array_equal(ref[current : (current + expected_batch_size)], batch)
        current += expected_batch_size
        total_found += batch.shape[0]
    assert total_expected == total_found
