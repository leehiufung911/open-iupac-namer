"""
tests/conftest.py

Test-suite fixtures and Windows-specific py2opsin hardening.

py2opsin (used by several round-trip tests) writes to a fixed temp file
``py2opsin_temp_input.txt`` in the current working directory and removes
it in a ``finally`` block.  On Windows the underlying Java process can
hold the file handle for a few hundred milliseconds after subprocess
completion (often due to antivirus scanning), causing
``PermissionError: [WinError 32]`` on the ``os.remove`` call and making
otherwise-deterministic tests flaky.

This conftest wraps :func:`py2opsin.py2opsin` once at session start with
a guarded variant that:

* Retries the temp-file removal a handful of times (with short sleeps),
  swallowing the error if the file ultimately cannot be removed (it will
  simply be overwritten on the next call).
* Optionally retries the entire OPSIN call once if the first call returns
  empty output, which can happen when a stale ``java.exe`` from a prior
  run is still holding the temp file open.

The wrapper is a no-op on non-Windows platforms or when ``py2opsin`` is
not importable.
"""

from __future__ import annotations

import os
import sys
import time

try:
    import py2opsin as _p2o
    _HAVE_OPSIN = True
except ImportError:
    _HAVE_OPSIN = False


def _force_remove_stale(path: str) -> None:
    """Best-effort removal of the py2opsin temp file."""
    for _ in range(5):
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.5)


def _install_py2opsin_retry_wrapper() -> None:
    """Patch ``py2opsin.py2opsin`` to retry on Windows file-lock races."""
    if not _HAVE_OPSIN or sys.platform != "win32":
        return

    original = _p2o.py2opsin
    tmp_path = "py2opsin_temp_input.txt"

    def _resilient_py2opsin(*args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(4):
            # Pre-clear any stale temp file before the call.
            _force_remove_stale(tmp_path)
            try:
                result = original(*args, **kwargs)
                # py2opsin returns False on internal exception, "" /
                # [""] on parse failure.  Empty results may also be
                # symptomatic of a stale-file read; retry once.
                if attempt == 0 and result is False:
                    time.sleep(0.5)
                    continue
                return result
            except PermissionError as exc:
                last_exc = exc
                time.sleep(0.5 * (attempt + 1))
            except Exception as exc:  # other py2opsin internal errors
                last_exc = exc
                time.sleep(0.3 * (attempt + 1))
        # All retries exhausted — re-raise the last exception so the
        # failure is visible.
        if last_exc is not None:
            raise last_exc
        return False

    _p2o.py2opsin = _resilient_py2opsin


_install_py2opsin_retry_wrapper()
