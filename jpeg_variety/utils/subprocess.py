"""Subprocess helpers.

We centralize subprocess behavior for consistent error reporting.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RunResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str


class CommandError(RuntimeError):
    def __init__(self, message: str, result: RunResult):
        super().__init__(message)
        self.result = result


def run(
    cmd: Iterable[str],
    *,
    input_bytes: bytes | None = None,
    timeout_s: float | None = None,
) -> RunResult:
    """Run a command, capturing stdout/stderr.

    Raises CommandError on non-zero exit.
    """

    cmd_list = [str(c) for c in cmd]

    proc = subprocess.run(
        cmd_list,
        input=input_bytes,
        capture_output=True,
        check=False,
        timeout=timeout_s,
    )
    res = RunResult(
        cmd=cmd_list,
        returncode=proc.returncode,
        stdout=(proc.stdout or b"").decode("utf-8", errors="replace"),
        stderr=(proc.stderr or b"").decode("utf-8", errors="replace"),
    )

    if res.returncode != 0:
        # Include stderr tail for signal.
        tail = res.stderr.strip().splitlines()[-20:]
        tail_text = "\n".join(tail)
        raise CommandError(
            f"Command failed with exit code {res.returncode}: {' '.join(res.cmd)}\n{tail_text}",
            res,
        )

    return res
