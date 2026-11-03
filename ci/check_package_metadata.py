"""Reject direct dependency references in built distributions before uploading."""

import argparse
from email.parser import BytesParser
from pathlib import Path
import tarfile
import zipfile

from packaging.requirements import Requirement


def read_metadata(path):
    """Read the single authoritative metadata file without extracting an archive."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA") and name.count("/") == 1
            ]
            if len(names) != 1:
                raise ValueError(f"{path}: expected one wheel METADATA file")
            return archive.read(names[0])
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.name.endswith("/PKG-INFO")
                and member.name.count("/") == 1
                and member.isfile()
            ]
            if len(members) != 1:
                raise ValueError(f"{path}: expected one top-level sdist PKG-INFO")
            return archive.extractfile(members[0]).read()
    raise ValueError(f"{path}: expected a .whl or .tar.gz distribution")


def check_distribution(path):
    """Validate every requirement, including inactive extras and platform markers."""
    metadata = BytesParser().parsebytes(read_metadata(path))
    if not metadata.get("Name") or not metadata.get("Version"):
        raise ValueError(f"{path}: missing package name or version")
    direct = [
        value
        for value in metadata.get_all("Requires-Dist", [])
        if Requirement(value).url is not None
    ]
    if direct:
        raise ValueError(
            f"{path}: PyPI rejects direct dependency references:\n"
            + "\n".join(f"  {value}" for value in direct)
        )


def main():
    """Check explicitly supplied artifacts and fail if any metadata is invalid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.artifacts:
        check_distribution(path)
        print(f"{path}: no direct dependency references")


if __name__ == "__main__":
    main()
