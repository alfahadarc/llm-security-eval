import os
import subprocess
import time
import uuid

import pytest

from codegen import sandbox

pytestmark = pytest.mark.filterwarnings("ignore")


def run(code, **kw):
    return sandbox.run_python({"main.py": code}, "main.py", **kw)


def needs_bwrap():
    if sandbox.backend() != "bwrap":
        pytest.skip("file-system isolation needs the bwrap backend")


def test_a_backend_is_available():
    assert sandbox.backend() in ("bwrap", "unshare")


def test_normal_code_runs_and_output_is_returned():
    r = run("print('hello'); import sys; print('err', file=sys.stderr)")
    assert r.returncode == 0 and r.stdout.strip() == "hello" and "err" in r.stderr
    assert not r.timed_out and not r.truncated


def test_failing_code_reports_its_exit_code_and_error():
    r = run("raise ValueError('boom')")
    assert r.returncode != 0 and "ValueError: boom" in r.stderr


def test_network_is_blocked():
    r = run("import socket\ns = socket.socket(); s.settimeout(2)\n"
            "try:\n    s.connect(('1.1.1.1', 80)); print('REACHABLE')\n"
            "except OSError as e:\n    print('blocked', type(e).__name__)\n")
    assert "blocked" in r.stdout and "REACHABLE" not in r.stdout


def test_dns_and_local_network_are_blocked_too():
    r = run("import socket\ntry:\n    socket.create_connection(('localhost', 22), timeout=2); print('REACHABLE')\n"
            "except OSError:\n    print('blocked')\n")
    assert "blocked" in r.stdout


def test_infinite_loop_is_killed_near_the_timeout():
    start = time.time()
    r = run("while True:\n    pass\n", timeout_s=1)
    assert time.time() - start < 6
    assert r.timed_out or r.returncode != 0  # the CPU limit may fire first; either way it is stopped


def test_sleeping_program_is_killed_by_the_wall_clock_timeout():
    start = time.time()
    r = run("import time\ntime.sleep(60)\n", timeout_s=1)
    assert r.timed_out and time.time() - start < 6


def test_memory_bomb_is_stopped():
    r = run("x = bytearray(2 * 1024**3)\nprint('ALLOCATED')\n", mem_mb=128)
    assert "ALLOCATED" not in r.stdout and r.returncode != 0


def test_writing_outside_the_work_directory_fails_but_inside_works():
    needs_bwrap()
    r = run("import os\nfor p in ('/etc/sandbox_probe', '/usr/sandbox_probe', '/sandbox_probe'):\n"
            "    try:\n        open(p, 'w').write('x'); print('WROTE', p)\n"
            "    except OSError:\n        print('denied', p)\n"
            "open('inside.txt', 'w').write('ok'); print('inside', os.path.exists('inside.txt'))\n")
    assert "WROTE" not in r.stdout and r.stdout.count("denied") == 3 and "inside True" in r.stdout


def test_host_tmp_and_home_are_hidden():
    needs_bwrap()
    marker = f"/tmp/sandbox_marker_{uuid.uuid4().hex}"
    open(marker, "w").write("host")
    try:
        r = run(f"import os\nprint('marker', os.path.exists({marker!r}))\nprint('home', os.listdir('/home'))\n")
    finally:
        os.remove(marker)
    assert "marker False" in r.stdout and "home []" in r.stdout


def test_work_directory_is_removed_after_the_run():
    r = run("import os\nprint(os.getcwd())\n")
    assert r.returncode == 0
    assert not os.path.exists(r.stdout.strip())


def test_child_process_does_not_survive_the_timeout():
    marker = "sandbox-child-" + uuid.uuid4().hex  # unique, so no other command line can contain it
    code = ("import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', {marker!r}])\n"
            "time.sleep(60)\n")
    run(code, timeout_s=1)
    time.sleep(0.5)
    found = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True)
    assert found.stdout.strip() == "", f"leftover process: {found.stdout}"


def test_huge_output_is_truncated():
    r = run("import sys\nsys.stdout.write('a' * 3_000_000)\n")
    assert r.truncated and len(r.stdout) <= sandbox.MAX_OUTPUT_BYTES


def test_several_files_and_imports_work():
    r = sandbox.run_python({"main.py": "from helper import twice\nprint(twice(21))",
                           "helper.py": "def twice(x):\n    return 2 * x\n",
                           "sub/data.txt": "x"}, "main.py")
    assert r.stdout.strip() == "42"


def test_stdin_is_closed():
    r = run("import sys\nprint(repr(sys.stdin.read()))\n")
    assert r.stdout.strip() == "''"


def test_subprocess_and_shell_tools_work_inside_the_sandbox():
    r = run("import subprocess\nprint(subprocess.run('echo hi; touch f; ls', shell=True, capture_output=True, "
            "text=True).stdout)\n")
    assert "hi" in r.stdout and "f" in r.stdout


@pytest.mark.parametrize("bad", ["/etc/x.py", "../x.py", ""])
def test_unsafe_file_names_are_rejected(bad):
    with pytest.raises(ValueError):
        sandbox.run_python({bad: "print(1)", "main.py": "print(1)"}, "main.py")


def test_entry_must_be_one_of_the_files():
    with pytest.raises(ValueError):
        sandbox.run_python({"main.py": "print(1)"}, "other.py")
