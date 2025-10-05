from autofaiss.indices.distributed import _batch_loader


def test_batch_loader():
    for input_size in range(2, 500):
        for output_size in range(1, input_size):
            batches = list(_batch_loader(nb_batches=output_size, total_size=input_size))
            # test output size is expected
            assert len(batches) == output_size
            # test no empty batch
            assert all(batch[1] <= input_size - 1 for batch in batches)
            # test on continuous between batches
            assert all(prev_end == next_start for (_, _, prev_end), (_, next_start, _) in zip(batches, batches[1:]))
            # test last element is covered
            assert batches[-1][2] >= input_size
            # test range sum
            assert sum(end - start for _, start, end in batches) == input_size
