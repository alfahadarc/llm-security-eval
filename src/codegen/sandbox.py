"""Run untrusted Python code in a restricted subprocess: no network, read-only file system, limits.
(The code runs as `python -E -s -B <entry>` inside the work directory.)

Backend 1 (preferred): bubblewrap (`bwrap`): all namespaces unshared (no network, own PID space), the root file
system mounted read-only, /tmp and /home hidden behind empty tmpfs mounts, and a fresh temporary work directory as
the only writable place. Backend 2 (fallback): `unshare --user --net`, which blocks the network but does NOT
isolate the file system; `SandboxResult.backend` says which one ran, and the file-system tests only run for bwrap.
Resource limits (CPU time, address space, file size, open files) and a wall-clock timeout apply to both; the
whole process group is killed on timeout. No Docker, no venv. Read access to the host file system (except /tmp
and /home) remains; generated code is treated as untrusted for WRITES, network and resources, and every task and
fixture in this project is synthetic.
"""
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_BYTES = 1_000_000
MAX_FILE_BYTES = 5_000_000
_BACKEND = None


@dataclass
class SandboxResult:
    returncode: int  # negative = killed by that signal; -9 also on timeout
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool
    backend: str


def _try(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def backend():
    """'bwrap' if bubblewrap works unprivileged here, else 'unshare', else raise."""
    global _BACKEND
    if _BACKEND is None:
        if shutil.which("bwrap") and _try(["bwrap", "--unshare-all", "--ro-bind", "/", "/", "true"]):
            _BACKEND = "bwrap"
        elif shutil.which("unshare") and _try(["unshare", "--user", "--map-root-user", "--net", "true"]):
            _BACKEND = "unshare"
        else:
            raise RuntimeError("no sandbox backend available (neither bwrap nor unshare works)")
    return _BACKEND


def _limits(timeout_s, mem_mb):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout_s) + 2, int(timeout_s) + 2))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def _command(kind, workdir, entry):
    # -E (ignore PYTHON* variables), -s (no user site), -B (no .pyc); not -I, so the work directory stays on sys.path.
    py = [sys.executable, "-E", "-s", "-B", entry]
    if kind == "bwrap":
        cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
               "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
               "--tmpfs", "/tmp", "--tmpfs", "/home"]
        if os.path.isdir("/root"):
            cmd += ["--tmpfs", "/root"]
        return cmd + ["--bind", workdir, workdir, "--chdir", workdir] + py
    return ["unshare", "--user", "--map-root-user", "--net", "--pid", "--fork"] + py


def _read_capped(path):
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        data = fh.read(MAX_OUTPUT_BYTES)
    return data.decode("utf-8", errors="replace"), size > MAX_OUTPUT_BYTES


def run_python(files, entry, timeout_s=5, mem_mb=256):
    """Write `files` ({relative name: text}) into a fresh temp directory and run `python -I <entry>` there.

    Returns a SandboxResult. The work directory is deleted afterwards. Raises ValueError for unsafe file names
    and RuntimeError when no backend works.
    """
    for name in list(files) + [entry]:
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or not name:
            raise ValueError(f"unsafe file name {name!r}")
    if entry not in files:
        raise ValueError(f"entry {entry!r} is not among the files")
    kind = backend()
    root = tempfile.mkdtemp(prefix="sandbox_")
    try:
        workdir = os.path.join(root, "work")
        os.mkdir(workdir)
        for name, text in files.items():
            target = Path(workdir) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        out_path, err_path = os.path.join(root, "stdout.txt"), os.path.join(root, "stderr.txt")
        timed_out = False
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            proc = subprocess.Popen(_command(kind, workdir, entry), cwd=workdir, stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=err, start_new_session=True,
                                    preexec_fn=_limits(timeout_s, mem_mb), env={"PATH": "/usr/bin:/bin"})
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.kill()
                proc.wait()
        stdout, cut_out = _read_capped(out_path)
        stderr, cut_err = _read_capped(err_path)
        return SandboxResult(returncode=proc.returncode, stdout=stdout, stderr=stderr, timed_out=timed_out,
                             truncated=cut_out or cut_err, backend=kind)
    finally:
        shutil.rmtree(root, ignore_errors=True)
