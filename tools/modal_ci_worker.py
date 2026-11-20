"""Run one build-and-test target inside a Modal sandbox (standard library only)."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from time import sleep
from zlib import crc32

PYTHON = "/opt/tests/bin/python"
UPSTREAM = "https://github.com/pybind/pybind11"


def run(*args: str) -> str:
    result = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
    )
    if result.returncode:
        sys.exit(f"{' '.join(args)} exited {result.returncode}\n{result.stdout}")
    return result.stdout


def checkout(sha: str) -> bool:
    """Check out the commit and report whether its test requirements need installing."""
    restored = Path(".git").is_dir()
    if not restored:
        run("git", "init", "-q")
    elif run("git", "rev-parse", "HEAD").strip() == sha:
        return False

    for attempt in range(3):
        fetch = subprocess.run(
            ["git", "fetch", "--quiet", "--no-tags", "--depth=1", UPSTREAM, sha],
            check=False,
        )
        if fetch.returncode == 0:
            break
        if attempt < 2:
            sleep(2**attempt)
    else:
        sys.exit(f"Could not fetch commit {sha} from {UPSTREAM}")

    changed = not restored or bool(
        run("git", "diff", "--name-only", "HEAD", sha, "--", "tests/requirements.txt")
    )
    # Restoring a snapshot changes inode numbers and ctimes. Refresh the index
    # before checkout so Git preserves unchanged files' mtimes for Ninja.
    run("git", "config", "core.trustctime", "false")
    run("git", "config", "core.checkStat", "minimal")
    run("git", "update-index", "-q", "--refresh")
    run("git", "checkout", "--quiet", "--force", "--detach", sha)
    return changed


def imports(path: Path) -> set[str]:
    """Names of sibling test files and bound submodules used by a Python test."""
    nodes = list(ast.walk(ast.parse(path.read_text(), filename=str(path))))
    used = set()
    aliases = {"pybind11_tests"}
    for node in nodes:
        if isinstance(node, ast.ImportFrom):
            used.add(node.module or "")
            prefix = f"{node.module}." if node.module else ""
            used.update(prefix + alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                used.add(alias.name)
                if alias.name == "pybind11_tests":
                    aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # importorskip() and import_module() also take literal module names.
            used.add(node.value)
    for node in nodes:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
        ):
            used.add("pybind11_tests." + node.attr)
    return {".".join(name.removeprefix("tests.").split(".")[:2]) for name in used}


def test_slices(folder: Path, count: int, python: str) -> list[list[str]]:
    """Partition the CMake test list, keeping each import with the file that provides it."""
    cmake = (folder / "CMakeLists.txt").read_text()
    listed = re.search(
        r"set\(PYBIND11_TEST_FILES\s+([^)]*)\)", re.sub(r"#[^\n]*", "", cmake)
    )
    if not listed or not listed[1].strip():
        sys.exit("Could not find PYBIND11_TEST_FILES in tests/CMakeLists.txt")
    names = {Path(entry).stem for entry in listed[1].split()}
    owners = {name: name for name in names}
    for name in sorted(names):
        cpp = folder / f"{name}.cpp"
        if cpp.exists():
            for module in re.findall(r"\bTEST_SUBMODULE\s*\(\s*(\w+)", cpp.read_text()):
                owners[f"pybind11_tests.{module}"] = name
        elif not (folder / f"{name}.py").is_file():
            sys.exit(f"Unrecognized PYBIND11_TEST_FILES entry: {name}")

    groups = [{name} for name in sorted(names)]
    for name in sorted(names):
        path = folder / f"{name}.py"
        if not path.exists():
            continue
        together = {name} | {owners[ref] for ref in imports(path) if ref in owners}
        if python.endswith("t") and name == "test_multiple_interpreters":
            # Collecting the conduit test enables the GIL, as in the unsliced run.
            together.add(owners["test_cpp_conduit"])
        # A test can import more than one module, including one already grouped.
        neighbors = [group for group in groups if group & together]
        groups = [group for group in groups if not group & together]
        groups.append(set.union(*neighbors))

    slices: list[list[str]] = [[] for _ in range(count)]
    # On PyPy, running test_multiple_registered_instances_for_same_pointee_recursive
    # before test_init_factory_basic gave 21 registered instances instead of 25 + 3.
    # Reversing them passed. #6261 resets statistics but retains live instances.
    # Extra collections also made the pair pass. Which wrappers were freed is unknown.
    isolate = (
        count > 1 and python.startswith("pypy") and "test_factory_constructors" in names
    )
    for group in groups:
        if isolate and "test_factory_constructors" in group:
            index = count - 1
        else:
            index = crc32(min(group).encode()) % (count - int(isolate))
        slices[index].extend(sorted(group))
    return slices


def build_and_test(
    sha: str, python: str, target: str, index: int, count: int, options: list[str]
) -> None:
    # Modal image defaults can override compiler flags, certificate lookup,
    # module search paths, hash randomization and build timestamps in the tests.
    for key in (
        "CFLAGS",
        "SSL_CERT_DIR",
        "PYTHONPATH",
        "PYTHONHASHSEED",
        "SOURCE_DATE_EPOCH",
    ):
        os.environ.pop(key, None)
    os.environ.update(
        PATH="/opt/tests/bin:" + os.environ["PATH"],
        PYTHONDEVMODE="1",
        PYTEST_TIMEOUT="300",
    )
    # A whole unsliced extension can exhaust VM memory with one compiler per visible CPU.
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = "8"
    Path("/src").mkdir(exist_ok=True)
    os.chdir("/src")
    if checkout(sha):
        run(
            "uv",
            "pip",
            "install",
            "--exact",
            "--python",
            PYTHON,
            "-r",
            "tests/requirements.txt",
            "setuptools",
        )

    override = ""
    if target == "pytest" and count > 1:
        files = test_slices(Path("tests"), count, python)[index]
        if not files:
            print("empty slice")
            return
        override = ";".join(files)

    run(
        "cmake",
        "-S.",
        "-Bbuild",
        "-GNinja",
        "-Werror=dev",
        "-DPYBIND11_WERROR=ON",
        "-DDOWNLOAD_CATCH=ON",
        f"-DPython_EXECUTABLE={PYTHON}",
        f"-DPYTHON_EXECUTABLE={PYTHON}",
        f"-DPYBIND11_TEST_OVERRIDE={override}",
        *options,
    )
    log = run("cmake", "--build", "build", "--target", target)
    # Ninja reports these steps for both the top-level and nested CMake builds.
    objects = re.findall(r"Building (?:C|CXX) object (\S+)", log)
    links = re.findall(
        r"Linking (?:C|CXX) (?:executable|shared module|static library) (\S+)", log
    )
    work = f"{len(objects)} objects, {len(links)} links"
    if 0 < len(objects) <= 3:
        work += " (" + ", ".join(Path(item).name for item in [*objects, *links]) + ")"
    summary = next(
        (
            line.strip("= ")
            for line in reversed(log.splitlines())
            if re.search(
                r"(?:passed|skipped|xfailed|xpassed).* in [\d.]+s|All tests passed|^test cases:",
                line,
            )
        ),
        target,
    )
    print(f"{work}, {summary}")


if __name__ == "__main__":
    commit, interpreter, target_name, ordinal, total, *flags = sys.argv[1:]
    build_and_test(commit, interpreter, target_name, int(ordinal), int(total), flags)
