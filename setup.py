from pathlib import Path

from setuptools import setup


def collect_conf_data_files() -> list[tuple[str, list[str]]]:
    conf_root = Path("conf")
    if not conf_root.exists():
        return []

    data_files: list[tuple[str, list[str]]] = []

    for directory in sorted(
        [
            conf_root,
            *[
                path
                for path in conf_root.rglob("*")
                if path.is_dir() and not any(part.startswith(".") for part in path.parts)
            ],
        ]
    ):
        if any(part.startswith(".") for part in directory.parts):
            continue

        files = sorted(
            str(path)
            for path in directory.iterdir()
            if path.is_file() and not path.name.startswith(".")
        )
        if files:
            data_files.append((directory.as_posix(), files))

    return data_files


setup(data_files=collect_conf_data_files())
