# /// script
# requires-python = ">=3.11,<3.16"
# dependencies = ["modal==1.6.4"]
# ///
"""Build and test an upstream pybind11 commit in parallel Modal sandboxes.

This driver sends modal_ci_worker.py to each VM. Only the driver needs Modal.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import sys
from pathlib import Path
from time import monotonic, time
from typing import NamedTuple

import modal

APP = "pybind11-test-runner"
CACHE = APP + "-baseline"
WORKER = Path(__file__).with_name("modal_ci_worker.py")
WEEK = 7 * 24 * 3600
UBUNTU = (
    "ubuntu@sha256:9a04f13b3f80ef056361e30868593b1cfd2279fe19e4081cb93a90bf56819d6a"
)


class Row(NamedTuple):
    python: str
    options: str
    slices: int = 6


# Interpreter build, CMake options and pytest slice count, from .github/workflows/ci.yml.
MATRIX = (
    # standard-large
    Row("3.9.25", "-DPYBIND11_FINDPYTHON=ON -DCMAKE_CXX_STANDARD=17"),
    Row(
        "3.12.15", "-DPYBIND11_TEST_SMART_HOLDER=ON -DPYBIND11_SIMPLE_GIL_MANAGEMENT=ON"
    ),
    # standard-small
    Row("3.13.16", "-DCMAKE_CXX_STANDARD=23 -DPYBIND11_SIMPLE_GIL_MANAGEMENT=ON"),
    Row("3.14.8t", "-DCMAKE_CXX_STANDARD=17 -DPYBIND11_TEST_SMART_HOLDER=ON"),
    Row("pypy3.11-v7.3.23", "-DCMAKE_CXX_STANDARD=17", 12),
    # type-creation-legacy
    Row(
        "3.14.8", "-DPYBIND11_TYPE_CREATION_VIA_SPEC=OFF -DPYBIND11_TEST_PRECOMPILE=ON"
    ),
)


def toolchain(python: str) -> modal.Image:
    uv = "https://github.com/astral-sh/uv/releases/download/0.13.4/uv-x86_64-unknown-linux-gnu.tar.gz"
    image = modal.Image.from_registry(UBUNTU).run_commands(
        "apt-get update && apt-get install -y ca-certificates",
        "apt-get --snapshot 20261119T000000Z update && "
        "apt-get --snapshot 20261119T000000Z install -y --no-install-recommends "
        "g++ git curl ninja-build make python3 libboost-dev libeigen3-dev catch2 libbz2-1.0 libsqlite3-0",
        f"curl -fsSL --retry 4 {uv} -o /tmp/uv.tgz && "
        "tar xzf /tmp/uv.tgz -C /usr/local/bin --strip-components=1 && rm /tmp/uv.tgz",
        "UV_TOOL_BIN_DIR=/usr/local/bin uv tool install --python /usr/bin/python3 cmake==4.2.3",
    )
    if python.startswith("pypy"):
        return image.run_commands(
            f"curl -fsSL --retry 4 https://downloads.python.org/pypy/{python}-linux64.tar.gz -o /tmp/pypy.tgz",
            "mkdir /opt/pypy && tar xzf /tmp/pypy.tgz -C /opt/pypy --strip-components=1 && rm /tmp/pypy.tgz",
            "uv venv --python /opt/pypy/bin/pypy3 /opt/tests",
        )
    return image.run_commands(f"uv venv --managed-python --python {python} /opt/tests")


class Shard(NamedTuple):
    row: Row
    target: str
    ordinal: int = 0

    @property
    def name(self) -> str:
        part = self.target
        if part == "pytest":
            part += f"-{self.ordinal + 1}of{self.row.slices}"
        return f"{self.row.python}/{part}"


async def run_shard(
    app: modal.App,
    shard: Shard,
    image: modal.Image,
    worker: str,
    args: argparse.Namespace,
    origin: str,
) -> str:
    start = monotonic()
    creating = asyncio.ensure_future(
        modal.Sandbox.create.aio(
            "sleep",
            "infinity",
            app=app,
            image=image,
            runtime="vm",
            cpu=(0.125, 8),
            memory=(1024, 16384),
            timeout=1800,
            experimental_options={"performance_sensitive": True}
            if args.performance_sensitive
            else None,
        )
    )
    # Ctrl-C can arrive after Modal accepted create but before we have its ID.
    try:
        box = await asyncio.shield(creating)
        # Executing the worker also waits for scheduling and boot. Bound that wait
        # separately from the time allowed for the worker's build and tests.
        launch = box.exec.aio(
            "/usr/bin/python3",
            "-c",
            worker,
            args.sha,
            shard.row.python,
            shard.target,
            str(shard.ordinal),
            str(shard.row.slices),
            *shard.row.options.split(),
            timeout=1500,
        )
        try:
            proc = await asyncio.wait_for(launch, timeout=120)
        except (asyncio.TimeoutError, modal.exception.TimeoutError) as exc:
            hint = (
                "Pass --no-performance-sensitive unless Modal has allowlisted this workspace."
                if args.performance_sensitive
                else "Check capacity in the Modal dashboard."
            )
            msg = f"{box.object_id} did not start the worker within 120s. {hint}"
            raise RuntimeError(msg) from exc
        out, err, code = await asyncio.gather(
            proc.stdout.read.aio(), proc.stderr.read.aio(), proc.wait.aio()
        )
        if code:
            msg = f"{box.object_id} exited {code}\n{out}{err}"
            raise RuntimeError(msg)
        snapshot = ""
        if args.seed_cache:
            snapshot = (await box.snapshot_filesystem.aio(ttl=WEEK)).object_id
        print(
            f"{shard.name} {box.object_id} {origin} {monotonic() - start:.1f}s  {out.strip()}",
            flush=True,
        )
        return snapshot
    finally:
        box = await creating
        stopping = asyncio.ensure_future(box.terminate.aio(wait=True))
        try:
            await asyncio.shield(stopping)
        finally:
            await stopping


async def main(args: argparse.Namespace) -> int:
    start = monotonic()
    app = await modal.App.lookup.aio(APP, create_if_missing=True)
    cache = modal.Dict.from_name(CACHE, create_if_missing=True)
    # A different driver or worker could produce an incompatible build tree.
    worker = WORKER.read_text()
    recipe = hashlib.sha256(Path(__file__).read_bytes() + worker.encode()).hexdigest()
    baseline: tuple[float, str, dict[str, str]] | None = (
        None if args.seed_cache else await cache.get.aio(recipe)
    )
    snapshots: dict[str, str] = {}
    baseline_sha = ""
    if baseline and time() < baseline[0]:
        _, baseline_sha, snapshots = baseline

    jobs = {}
    for preset in MATRIX:
        if args.python and preset.python not in args.python:
            continue
        row = preset._replace(slices=1) if args.single_slice else preset
        parts = [Shard(row, "pytest", i) for i in range(row.slices)]
        parts.append(Shard(row, "test_cmake_build"))
        if not row.python.startswith("pypy"):  # cpptest is a no-op on PyPy.
            parts.append(Shard(row, "cpptest"))
        base = toolchain(row.python)
        for shard in parts:
            saved = snapshots.get(shard.name)
            image = modal.Image.from_id(saved) if saved else base
            origin = baseline_sha[:10] if saved else "cold"
            jobs[shard.name] = run_shard(app, shard, image, worker, args, origin)

    print(f"Running {len(jobs)} shards on Modal ({APP})", flush=True)
    expires = time() + WEEK - 1800  # Allow the last reader time to start its sandboxes.
    results = await asyncio.gather(*jobs.values(), return_exceptions=True)
    saved_images = {}
    for name, result in zip(jobs, results):
        if isinstance(result, BaseException):
            print(f"{name} FAILED: {result}", file=sys.stderr)
        else:
            saved_images[name] = result
    failed = len(jobs) - len(saved_images)
    if args.seed_cache and not failed:
        # One atomic replacement. Readers see a complete passing baseline even if
        # another seed is interrupted or two seeds finish concurrently.
        await cache.put.aio(recipe, (expires, args.sha, saved_images))
    print(f"{len(jobs)} shards, {failed} failed, {monotonic() - start:.1f}s total")
    return int(bool(failed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sha", help="full commit SHA fetchable from pybind/pybind11")
    parser.add_argument(
        "--seed-cache",
        action="store_true",
        help="run the full matrix cold and save it as the baseline if it passes",
    )
    parser.add_argument(
        "--python",
        choices=[row.python for row in MATRIX],
        action="append",
        help="run only this interpreter (repeatable)",
    )
    parser.add_argument(
        "--single-slice",
        action="store_true",
        help="run each selected row's pytest suite in one sandbox",
    )
    parser.add_argument(
        "--performance-sensitive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use performance-sensitive scheduling, which needs a workspace allowlisted by Modal",
    )
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-fA-F]{40}", args.sha):
        parser.error("expected a full 40-digit hexadecimal pybind11 commit SHA")
    args.sha = args.sha.lower()
    if args.seed_cache and (args.python or args.single_slice):
        parser.error("--seed-cache requires the full matrix")
    return args


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main(parse_args())))
    except modal.exception.AuthError:
        sys.exit("Modal authentication failed. Run 'uvx modal setup' and retry.")
    except modal.exception.Error as error:
        sys.exit(str(error))
    except KeyboardInterrupt:
        sys.exit(130)
