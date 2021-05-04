import fire
import os


def create_toctrees(folder_input, folder_output):
    d = folder_input
    folders = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
    for dd in folders:
        last_part = dd.split("/")[-1]
        with open(os.path.join(folder_output, f"{last_part}.rst"), "w") as f:
            f.write(
                f"""{last_part}
{'-'*len(last_part)}

.. toctree::
   :glob:
   :maxdepth: 1

   ../_notebooks/{last_part}/*
            """
            )


if __name__ == "__main__":
    fire.Fire(create_toctrees)
