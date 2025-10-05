"""Check version and git tag script."""

from pathlib import Path
import re
import sys
import subprocess


if __name__ == "__main__":
    # Read package version
    with Path("autofaiss/version.py").open(encoding="utf-8") as file:
        metadata = dict(re.findall(r'__([a-z]+)__\s*=\s*"([^"]+)"', file.read()))
        version = metadata["version"]

    # Read git tag
    with subprocess.Popen(["git", "describe", "--tags"], stdout=subprocess.PIPE) as process:
        tagged_version = process.communicate()[0].strip().decode(encoding="utf-8")

    # Exit depending on version and tagged_version
    if version == tagged_version:
        print(f"Tag and version are the same ({version}) !")
        sys.exit(0)
    else:
        print(f"Tag {tagged_version} and version {version} are not the same !")
        sys.exit(1)
