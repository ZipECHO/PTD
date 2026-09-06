"""Lifecycle management for the PTD cache server process."""

import atexit
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_CACHE_SERVER_HANDLE = None
_CACHE_SERVER_LOCK = threading.RLock()


def _repo_cache_server_script() -> str:
    return str(Path(__file__).resolve().parents[1] / "cache_server.py")


def _ipc_ready() -> bool:
    """Return whether the shared memory and semaphores are available."""
    try:
        from multiprocessing.shared_memory import SharedMemory
        import posix_ipc

        shared = []
        try:
            for name in ("guess_tree", "request"):
                shm = SharedMemory(name=name)
                shared.append(shm)
            for name in ("/guess_res_sem", "/request_sem"):
                sem = posix_ipc.Semaphore(name)
                sem.close()
            return True
        finally:
            for shm in shared:
                shm.close()
    except (FileNotFoundError, ImportError, OSError):
        return False


@dataclass
class CacheServerHandle:
    """A managed cache server process."""

    process: Optional[subprocess.Popen]
    batch_size: int
    auto_cleanup: bool = False

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid if self.process is not None else None

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the process and let its signal handler clean IPC resources."""
        if self.process is None or self.process.poll() is not None:
            return
        try:
            self.process.send_signal(signal.SIGTERM)
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)
        finally:
            self.process = None


def _launch_cache_server(
    batch_size: int,
    rest_corpus_path: Optional[str],
    tokenizer_path: Optional[str],
    timeout: float,
    auto_cleanup: bool,
) -> CacheServerHandle:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if (rest_corpus_path is None) != (tokenizer_path is None):
        raise ValueError("rest_corpus_path and tokenizer_path must be provided together")

    command = [sys.executable, _repo_cache_server_script(), str(batch_size)]
    if rest_corpus_path is not None:
        command.extend([rest_corpus_path, tokenizer_path])

    process = subprocess.Popen(
        command,
        cwd=str(Path(_repo_cache_server_script()).parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    handle = CacheServerHandle(process=process, batch_size=batch_size, auto_cleanup=auto_cleanup)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _ipc_ready():
            return handle
        if process.poll() is not None:
            code = process.returncode
            handle.stop()
            raise RuntimeError(f"Cache server exited during startup (return code {code})")
        time.sleep(0.1)

    handle.stop()
    raise TimeoutError(f"Cache server did not become ready within {timeout:.1f}s")


def ensure_cache_server(
    batch_size: int = 1,
    rest_corpus_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    timeout: float = 30.0,
) -> CacheServerHandle:
    """Return a running server, starting one on demand when necessary."""
    global _CACHE_SERVER_HANDLE
    with _CACHE_SERVER_LOCK:
        if _CACHE_SERVER_HANDLE is not None and _CACHE_SERVER_HANDLE.is_alive():
            if batch_size > _CACHE_SERVER_HANDLE.batch_size:
                raise RuntimeError(
                    "An existing PTD cache server has batch_size="
                    f"{_CACHE_SERVER_HANDLE.batch_size}, requested {batch_size}. "
                    "Stop it before starting a larger batch."
                )
            return _CACHE_SERVER_HANDLE
        _CACHE_SERVER_HANDLE = _launch_cache_server(
            batch_size=batch_size,
            rest_corpus_path=rest_corpus_path,
            tokenizer_path=tokenizer_path,
            timeout=timeout,
            auto_cleanup=True,
        )
        return _CACHE_SERVER_HANDLE


def start_cache_server(
    batch_size: int = 1,
    rest_corpus_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    timeout: float = 30.0,
) -> CacheServerHandle:
    """Explicitly start (or reuse) a server owned by the caller."""
    global _CACHE_SERVER_HANDLE
    with _CACHE_SERVER_LOCK:
        if _CACHE_SERVER_HANDLE is not None and _CACHE_SERVER_HANDLE.is_alive():
            return _CACHE_SERVER_HANDLE
        _CACHE_SERVER_HANDLE = _launch_cache_server(
            batch_size=batch_size,
            rest_corpus_path=rest_corpus_path,
            tokenizer_path=tokenizer_path,
            timeout=timeout,
            auto_cleanup=False,
        )
        return _CACHE_SERVER_HANDLE


def stop_cache_server(handle: Optional[CacheServerHandle] = None) -> None:
    """Stop a managed server and clear the process handle."""
    global _CACHE_SERVER_HANDLE
    with _CACHE_SERVER_LOCK:
        target = handle or _CACHE_SERVER_HANDLE
        if target is not None:
            target.stop()
        if target is _CACHE_SERVER_HANDLE:
            _CACHE_SERVER_HANDLE = None


@atexit.register
def _cleanup_cache_server() -> None:
    stop_cache_server()


__all__ = ["CacheServerHandle", "ensure_cache_server", "start_cache_server", "stop_cache_server"]
