import json
import fire
import os
import re


def make_source_not_title(s):
    return re.sub(r"^#", "##", s)


def remove_title(cell):
    if cell["cell_type"] != "markdown":
        return cell
    cell["source"] = [make_source_not_title(s) for s in cell["source"]]
    return cell


def add_title(file):
    last_part = file.split("/")[-1]
    name = str.join(".", last_part.split(".")[:-1]).replace("_", " ").replace("-", " ").capitalize()
    with open(file, encoding="utf-8") as f:
        i = json.load(f)
    i["cells"] = [remove_title(cell) for cell in i["cells"]]
    i["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": [f"# {name}"]})
    with open(file, "w", encoding="utf-8") as f:
        json.dump(i, f, indent=1)


def add_title_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".ipynb"):
                add_title(os.path.join(root, file))


if __name__ == "__main__":
    fire.Fire(add_title_folder)
