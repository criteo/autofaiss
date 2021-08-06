""" function to interact with os """

import os
import subprocess
from typing import List


def run_command(cmd: str) -> bool:
    """Function to run a bash command"""
    try:
        subprocess.run(cmd.split(), check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def list_local_files(path: str) -> List[str]:
    """function to list the files in a directory"""

    for infos in os.walk(path):
        return infos[2]
    return []
