""" context manager to make reading/writing on hdfs as simple as writing on a local file """

from typing import Optional

import pyarrow
from pyarrow.filesystem import FileSystem


class HDFSFileSystem:
    """Context aware HDFSFileSystem using pyarrow.hdfs.
    Open and closes connection to HDFS thanks to a context manager
    >>> from deepr.io import HDFSFileSystem
    >>> with HDFSFileSystem() as fs:  # doctest: +SKIP
    ...     fs.open("path/to/file")  # doctest: +SKIP
    """

    def __init__(self):
        self._hdfs = None

    def __enter__(self):
        self._hdfs = pyarrow.hdfs.connect()
        return self._hdfs

    def __exit__(self, type, value, traceback):
        # pylint: disable=redefined-builtin
        self._hdfs.close()

    def __getattr__(self, name):
        # Expose self._hdfs methods and attributes (mimic inheritance)
        return getattr(self._hdfs, name)


class HDFSFile:
    """FileSystemFile, support of "r", "w" modes, readlines and iter.
    Makes it easier to read or write file from any filesystem. For
    example, if you use HDFS you can do
    >>> from deepr.io import HDFSFileSystem
    >>> with HDFSFileSystem() as fs:
    ...     with HDFSFile(fs, "viewfs://root/user/foo.txt", "w") as file:  # doctest: +SKIP
    ...         file.write("Hello world!")  # doctest: +SKIP
    The use of context manager means that the connection to the
    filesystem is automatically opened / closed, as well as the file
    buffer.
    Attributes
    ----------
    filesystem : FileSystem
        FileSystem instance
    path : str
        Path to file
    mode : str, Optional
        Write / read mode. Supported: "r", "rb" (default), "w", "wb".
    """

    def __init__(self, filesystem: FileSystem, path: str, mode: str = "rb", encoding: Optional[str] = "utf-8"):
        self.filesystem = filesystem
        self.path = path
        self.mode = mode
        self.encoding = None if "b" in mode else encoding
        self._file = filesystem.open(self.path, mode={"r": "rb", "w": "wb"}.get(mode, mode))

    def __iter__(self):
        yield from self.readlines()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # pylint: disable=redefined-builtin
        return self._file.__exit__(type, value, traceback)

    def __getattr__(self, name):
        # Expose self._file methods and attributes (mimic inheritance)
        return getattr(self._file, name)

    def write(self, data):
        if self.mode == "w":
            self._file.write(data.encode(encoding=self.encoding))
        elif self.mode == "wb":
            self._file.write(data)
        else:
            raise ValueError(f"Mode {self.mode} unkown (must be 'w' or 'wb').")

    def read(self):
        if self.mode == "r":
            return self._file.read().decode(encoding=self.encoding)
        elif self.mode == "rb":
            return self._file.read()
        else:
            raise ValueError(f"Mode {self.mode} unkown (must be 'r' or 'rb')")

    def readlines(self):
        return self.read().split("\n" if self.mode == "r" else b"\n")
