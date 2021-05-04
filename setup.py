#!/usr/bin/env python3
"""Setup script"""

from pathlib import Path
import re

import setuptools

if __name__ == "__main__":

    # Read metadata from version.py
    with Path("autofaiss/version.py").open(encoding="utf-8") as file:
        metadata = dict(re.findall(r'__([a-z]+)__\s*=\s*"([^"]+)"', file.read()))

    # Read description from README
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    _INSTALL_REQUIRES = [
        "dataclasses",
        "fire>=0.4.0",
        "numpy>=1.18.2",
        "pandas>=1.0.5",
        "pyarrow>=0.14",
        "tqdm>=4.46.0",
        "faiss-cpu>=1.7.0",
    ]

    _TEST_REQUIRE = ["pytest"]

    # Run setup
    setuptools.setup(
        name="autofaiss",
        version=metadata["version"],
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Intended Audience :: Developers",
        ],
        long_description=long_description,
        long_description_content_type="text/markdown",
        description=long_description.split("\n")[0],
        author=metadata["author"],
        install_requires=_INSTALL_REQUIRES,
        tests_require=_TEST_REQUIRE,
        dependency_links=[],
        entry_points={"console_scripts": ["autofaiss = autofaiss.external.quantize:main"]},
        data_files=[(".", ["requirements.txt", "README.md"])],
        packages=setuptools.find_packages(),
        url="https://github.com/criteo/autofaiss",
    )
