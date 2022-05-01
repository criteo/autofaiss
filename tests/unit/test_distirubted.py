from autofaiss.indices.distributed import _get_merge_batches


def test_get_merge_batches():
    for input_size in range(2, 500):
        for output_size in range(1, input_size):
            batches = list(_get_merge_batches(input_size, output_size))
            # test output size is expected
            assert len(batches) == output_size
            # test no empty batch
            assert all(batch[0] <= input_size - 1 for batch in batches)
