"""Exercise the upload guard against wheel and sdist metadata."""

import io
import tarfile
import zipfile

import pytest

from ci.check_package_metadata import check_distribution


@pytest.fixture(params=["wheel", "sdist"])
def artifact(request, tmp_path):
    """Create either distribution format with supplied core metadata."""

    def write(requirements):
        """Write metadata and return the archive path."""
        metadata = (
            "Metadata-Version: 2.2\nName: example\nVersion: 1.0\n"
            + "".join(f"Requires-Dist: {value}\n" for value in requirements)
            + "\n"
        ).encode()
        if request.param == "wheel":
            path = tmp_path / "example-1.0-py3-none-any.whl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("example-1.0.dist-info/METADATA", metadata)
        else:
            path = tmp_path / "example-1.0.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                for name in [
                    "example-1.0/PKG-INFO",
                    "example-1.0/example.egg-info/PKG-INFO",
                ]:
                    info = tarfile.TarInfo(name)
                    info.size = len(metadata)
                    archive.addfile(info, io.BytesIO(metadata))
        return path

    return write


@pytest.mark.parametrize(
    "requirement",
    [
        "example @ git+https://example.org/repo.git@v1.0",
        'example @ git+https://example.org/repo.git ; extra == "external"',
        'example @ https://example.org/example.whl ; python_version < "3.0"',
        "example @ file:///tmp/example",
    ],
)
def test_rejects_direct_references(artifact, requirement):
    """Pinned URLs, extras and false markers must all fail before upload."""
    with pytest.raises(ValueError, match="PyPI rejects direct dependency"):
        check_distribution(artifact(["numpy", requirement]))


def test_accepts_index_requirements(artifact):
    """Ordinary dependencies and optional version constraints remain valid."""
    check_distribution(artifact(["numpy>=1.24", 'openai-whisper; extra == "external"']))


def test_rejects_invalid_requirement(artifact):
    """Malformed dependency metadata cannot silently pass."""
    with pytest.raises(ValueError):
        check_distribution(artifact(["not a requirement"]))


def test_rejects_missing_metadata(tmp_path):
    """An empty wheel must not produce a false success."""
    path = tmp_path / "empty.whl"
    with zipfile.ZipFile(path, "w"):
        pass
    with pytest.raises(ValueError, match="expected one wheel METADATA"):
        check_distribution(path)
