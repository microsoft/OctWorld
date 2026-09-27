#!/usr/bin/env python3
"""Download/check all public auxiliary models without loading Torch or a GPU.

Only the unpublished OctWorld-1.5 model package is deliberately omitted.
The manifest pins upstream revisions, file sizes, and LFS SHA-256/Git blob hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")
BLOCK_SIZE = 8 * 1024 * 1024


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    for model in manifest.values():
        for relative in [model["directory"], *(entry[0] for entry in model["files"])]:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe manifest path: {relative}")
        if "repo_id" in model and len(model["revision"]) != 40:
            raise ValueError("Hugging Face revisions must be pinned commits")
        for name, size, algorithm, digest in model["files"]:
            length = {"sha256": 64, "git-sha1": 40}.get(algorithm)
            if size <= 0 or len(digest) != length:
                raise ValueError(f"Invalid manifest entry: {name}")
    return manifest


def file_error(path: Path, entry: list) -> str | None:
    """Return None only for a byte-for-byte match; never deserialize weights."""
    _, size, algorithm, expected = entry
    if not path.is_file():
        return "missing file"
    if path.stat().st_size != size:
        return f"size mismatch (expected {size}, got {path.stat().st_size})"
    digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
    if algorithm == "git-sha1":
        digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            digest.update(block)
    return None if digest.hexdigest() == expected else f"{algorithm} mismatch"


def publish_stream(source, destination: Path, entry: list) -> None:
    """Verify a temporary file before publishing; preserve any existing file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".partial", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            shutil.copyfileobj(source, stream, BLOCK_SIZE)
            stream.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        error = file_error(temporary, entry)
        if error:
            raise ValueError(f"Downloaded {destination.name}: {error}")
        # Hard-linking publishes atomically and refuses to overwrite a concurrent writer.
        destination.hardlink_to(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def download_file(model: dict, entry: list, destination: Path) -> None:
    if "repo_id" in model:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        cached = try_to_load_from_cache(
            model["repo_id"], entry[0], revision=model["revision"]
        )
        if isinstance(cached, str) and file_error(Path(cached), entry) is None:
            print(f"  Reusing Hugging Face cache: {entry[0]}", flush=True)
            with open(cached, "rb") as stream:
                publish_stream(stream, destination, entry)
        else:
            # Keep resumable transfers separate from runtime files until verified.
            # Staging is on the same filesystem, so publication needs no full copy.
            local_dir = destination
            for _ in Path(entry[0]).parts:
                local_dir = local_dir.parent
            downloaded = Path(hf_hub_download(
                repo_id=model["repo_id"], filename=entry[0], revision=model["revision"],
                local_dir=local_dir / ".download-staging",
            ))
            error = file_error(downloaded, entry)
            if error:
                # Only remove our own staged file, never an existing runtime file.
                downloaded.unlink(missing_ok=True)
                raise ValueError(f"Downloaded {destination.name}: {error}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.hardlink_to(downloaded)
            downloaded.unlink()
    else:
        request = Request(model["url"], headers={"User-Agent": "OctWorld-model-downloader"})
        with urlopen(request, timeout=120) as response:
            publish_stream(response, destination, entry)
    error = file_error(destination, entry)
    if error:
        raise ValueError(f"Downloaded file failed verification: {destination}: {error}")


def main(argv: list[str] | None = None) -> int:
    manifest = load_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "model_weights")
    parser.add_argument("--models", nargs="+", choices=list(manifest), default=list(manifest))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Offline, read-only size and hash verification")
    mode.add_argument("--list", action="store_true", help="Print the pinned download plan without network or writes")
    args = parser.parse_args(argv)
    names = list(dict.fromkeys(args.models))
    root = args.output_dir.expanduser().absolute()
    print(f"Model root: {root}")
    print("SKIP OctWorld-1.5: personal Hugging Face checkpoint is not published yet.\n"
          "This command checks/downloads auxiliaries only, not generation readiness.")
    total = sum(entry[1] for name in names for entry in manifest[name]["files"])
    print(f"Selected auxiliary files: {total / 1e9:.2f} GB ({total / 2**30:.2f} GiB)")
    failures = []
    for name in names:
        model = manifest[name]
        print(f"\n[{name}] {model.get('repo_id', model.get('url'))}", flush=True)
        if "revision" in model:
            print(f"  revision: {model['revision']}")
        for entry in model["files"]:
            path = root / model["directory"] / entry[0]
            if args.list:
                print(f"  {path.relative_to(root)} ({entry[1]} bytes)")
                continue
            error = file_error(path, entry)
            if error is None:
                print(f"  OK {path.relative_to(root)}", flush=True)
                continue
            if args.check or path.exists() or path.is_symlink():
                failures.append(f"{path}: {error}")
                print(f"  FAIL {failures[-1]}", flush=True)
                continue
            print(f"  Download {path.relative_to(root)}", flush=True)
            try:
                download_file(model, entry, path)
            except ValueError:
                failures.append(f"{path}: downloaded content did not pass manifest verification")
                print(f"  FAIL {failures[-1]}", flush=True)
            except Exception as exc:
                # Do not echo HTTP exceptions: signed download URLs can contain credentials.
                failures.append(f"{path}: download failed ({type(exc).__name__})")
                print(f"  FAIL {failures[-1]}", flush=True)
            else:
                print(f"  OK {path.relative_to(root)}", flush=True)
    if failures:
        print(f"\nFAILED: {len(failures)} auxiliary file(s). Existing mismatched files are never overwritten.\n"
              "Move mismatched files aside after inspection, check network/access, then rerun.", file=sys.stderr)
        return 1
    print("\nPlan only; no files changed." if args.list else "\nAll selected auxiliary files verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())