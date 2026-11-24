# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import operator
import os
import stat
import sys
import threading
import warnings
from collections import OrderedDict
from collections.abc import Iterable
from types import ModuleType, TracebackType

from cuda.core import CUDAWarning, Stream
from cuda.core._memory._buffer import Buffer

from ._transfer import _allocate, _CountPool, _new_stream, _resize_output, _Transfer, _validate_stream

# cuObject and other remote storage backends can reuse the GPU buffer and
# transfer lifetime machinery when they have compatible completion semantics.
_Path = str | bytes | os.PathLike[str] | os.PathLike[bytes]
_FileEntry = tuple[Buffer, int] | tuple[Buffer, int, int | None]
_MAX_OFFSET = (1 << 63) - 1
_MODES = {
    "rb": os.O_RDONLY,
    "r+b": os.O_RDWR,
    "wb": os.O_WRONLY | os.O_CREAT,
    "w+b": os.O_RDWR | os.O_CREAT,
    "xb": os.O_WRONLY | os.O_CREAT | os.O_EXCL,
    "x+b": os.O_RDWR | os.O_CREAT | os.O_EXCL,
}


def open_file(path: _Path, mode: str = "rb", *, direct: bool = False) -> _File:
    """Open a regular binary file for transfers between disk and GPU memory.

    Parameters
    ----------
    path : str or bytes or os.PathLike
        Path to a local filesystem file.
    mode : str, optional
        Binary mode. Use ``"rb"`` to read, ``"wb"`` to create or truncate,
        or ``"xb"`` to create exclusively. Add ``+`` for read/write access.
    direct : bool, optional
        Open with Linux ``O_DIRECT``. cuFile chooses the transfer path based
        on the filesystem, hardware, and its configuration. It can use host
        staging internally.

    Returns
    -------
    :class:`._File`
        An open file for positional asynchronous and batch I/O. Use as a
        context manager or call ``close()``.

    Notes
    -----
    Requires Linux and the cuFile runtime.
    """
    return _open_file(path, mode, direct=direct, truncate=True)


def _open_file(path: _Path, mode: str, *, direct: bool, truncate: bool) -> _File:
    if sys.platform != "linux":
        raise RuntimeError("cuFile storage is available only on Linux")
    if mode in ("rb+", "wb+", "xb+"):
        mode = mode[0] + "+b"
    if mode not in _MODES:
        raise ValueError("mode must be a binary mode: rb, r+b, wb, w+b, xb, or x+b")
    from cuda.bindings import cufile

    # Resolve the optional runtime before opening a path for writing.
    cufile.get_version()
    file = _File(cufile, mode)
    fd = os.open(path, _MODES[mode] | (os.O_DIRECT if direct else 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("cuFile storage requires a regular file")
        desc = cufile.Descr()
        desc.type = cufile.FileHandleType.OPAQUE_FD
        desc.handle.fd = fd
        desc.fs_ops = 0
        file._handle = cufile.handle_register(desc.ptr)
        if truncate and mode.startswith("w"):
            os.ftruncate(fd, 0)
    except BaseException as error:
        if file._handle is not None:
            try:
                cufile.handle_deregister(file._handle)
            except Exception as exc:
                error.add_note(f"cuFile deregistration failed: {exc}")
        try:
            os.close(fd)
        except OSError as exc:
            error.add_note(f"closing the file descriptor failed: {exc}")
        raise
    file._fd = fd
    return file


def open(path: _Path, mode: str = "rb", *, stream: Stream | None, direct: bool = False) -> _StreamFile:
    """Open a binary GPU file with a cursor bound to a CUDA stream.

    Pass an explicit CUDA stream, or pass `stream=None` to create one on the
    current context (device 0's primary context if none is current).
    `mode` and `direct` have the same meaning as in :func:`open_file`.
    Closing the returned object closes the underlying file and any stream
    created by this function.
    """
    owns_stream = stream is None
    if stream is None:
        stream = _new_stream()
    else:
        _validate_stream(stream)
    with_file = None
    try:
        with_file = open_file(path, mode, direct=direct)
        return _StreamFile(with_file, stream, owns_file=True, owns_stream=owns_stream)
    except BaseException as error:
        if with_file is not None:
            with_file.__exit__(type(error), error, error.__traceback__)
        if owns_stream:
            try:
                stream.close()
            except Exception as cleanup:
                error.add_note(f"closing the CUDA stream failed: {cleanup}")
        raise


class _Request:
    """Completion of one positional transfer submitted to a CUDA stream."""

    def __init__(self, cufile: ModuleType, writing: bool, *, empty: bool = False):
        self._cufile = cufile
        self._writing = writing
        self._native = None if empty else _Transfer()
        self._count = 0 if empty else None
        self._observed = False
        self._lock = threading.Lock()

    def _complete(self, *, wait: bool) -> int | None:
        # Keep a local reference to release CUDA owners after unlocking.
        if not self._lock.acquire(blocking=wait):
            return None
        try:
            native = self._native
            if native is not None:
                count = native.complete(wait)
                if count is not None:
                    self._count = count
                    self._native = None
            return self._count
        finally:
            self._lock.release()

    @property
    def is_done(self) -> bool:
        """Whether the transfer has finished. Use `result()` for its status."""
        return self._complete(wait=False) is not None

    def result(self) -> int:
        """Wait for this transfer and return the number of bytes transferred.

        A successful read may be short, including zero bytes at EOF. Raises
        on a transfer error. Repeated calls return the same outcome.
        """
        count: int = self._complete(wait=True)  # type: ignore[assignment]
        with self._lock:
            self._observed = True
        if count == -1:
            operation = "write" if self._writing else "read"
            raise OSError(f"cuFile asynchronous {operation} failed")
        if count < 0:
            raise self._cufile.cuFileError(-count)
        return count


class _File:
    """A binary file for positional device I/O, returned by `open_file()`."""

    def __init__(self, cufile: ModuleType, mode: str):
        self._fd: int | None = None
        self._handle: int | None = None
        self._cufile = cufile
        self._readable = mode.startswith("r") or "+" in mode
        self._writable = not mode.startswith("r") or "+" in mode
        self._pending: OrderedDict[_Request, None] = OrderedDict()
        self._pending_lock = threading.Lock()
        self._count_pool: _CountPool | None = _CountPool()
        self._unresolved_batch = False

    @property
    def closed(self) -> bool:
        """Whether this file has been closed."""
        return self._fd is None

    @property
    def size(self) -> int:
        """Current file size in bytes. Wait for pending writes to observe their size."""
        fd = self._fd
        if fd is None:
            raise ValueError("I/O on a closed cuFile file")
        return os.fstat(fd).st_size

    def __enter__(self) -> _File:
        if self.closed:
            raise ValueError("I/O on a closed cuFile file")
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        try:
            self.close()
        except Exception as cleanup:
            if exc is None:
                raise
            exc.add_note(f"closing the cuFile file failed: {cleanup}")

    def __del__(self):
        if self._fd is not None and not sys.is_finalizing():
            try:
                self.close()
            except Exception as exc:
                warnings.warn(f"closing the cuFile file failed: {exc}", CUDAWarning, stacklevel=2)

    def close(self) -> None:
        """Wait for outstanding I/O, deregister the file, and close its descriptor.

        Raises if an outstanding request failed and its error has not already
        been retrieved. When completion cannot be established, keeps the file
        open so the caller can retry.
        """
        fd = self._fd
        if fd is None:
            return
        if self._unresolved_batch:
            raise RuntimeError("cuFile batch completion could not be established; the file is still retained")
        for request in self._pending:
            request._complete(wait=True)
        self._cufile.handle_deregister(self._handle)
        self._fd = None
        self._handle = None
        error = None
        try:
            os.close(fd)
        except OSError as exc:
            error = exc
        for request in self._pending:
            if request._count < 0 and not request._observed:  # type: ignore[operator]
                try:
                    request.result()
                except Exception as exc:
                    if error is None:
                        error = exc
                    else:
                        error.add_note(str(exc))
        self._pending.clear()
        self._count_pool = None
        if error is not None:
            raise error

    def _check_access(self, writing: bool) -> None:
        if self.closed:
            raise ValueError("I/O on a closed cuFile file")
        if self._unresolved_batch:
            raise RuntimeError("cuFile batch completion could not be established")
        if not (self._writable if writing else self._readable):
            raise OSError("cuFile file is not writable" if writing else "cuFile file is not readable")

    def _prepare(self, buffer: Buffer, offset: int, size: int | None, *, writing: bool) -> tuple[int, int]:
        self._check_access(writing)
        offset = operator.index(offset)
        if not isinstance(buffer, Buffer):
            raise TypeError("cuFile I/O requires a cuda.core.Buffer")
        if buffer.is_closed:
            raise RuntimeError("Buffer has been closed")
        size = buffer.size if size is None else operator.index(size)
        if size < 0 or size > buffer.size:
            raise ValueError("size must be between zero and the GPU buffer size")
        if offset < 0 or offset + size > _MAX_OFFSET:
            raise ValueError("the file byte range must fit in a nonnegative signed 64-bit offset")
        return offset, size

    def _transfer(self, buffer: Buffer, offset: int, size: int | None, *, writing: bool, stream: Stream) -> _Request:
        offset, size = self._prepare(buffer, offset, size, writing=writing)
        _validate_stream(stream)
        if (
            size == 0
            and (buffer.size or buffer.handle)
            and (not buffer.is_device_accessible or buffer.is_host_accessible)
        ):
            raise ValueError("cuFile I/O requires device-resident GPU memory")
        request = _Request(self._cufile, writing, empty=size == 0)
        native = request._native
        if native is not None:
            # Check one prior request per submission to bound the polling
            # cost and give every stream a chance to retire completed I/O.
            with self._pending_lock:
                previous = next(iter(self._pending), None)
                if previous is not None:
                    self._pending.move_to_end(previous)
            if previous is not None:
                count = previous._complete(wait=False)
                if count is not None and (count >= 0 or previous._observed):
                    with self._pending_lock:
                        self._pending.pop(previous, None)
            # Publish before submission in case the CUDA event or context
            # restoration fails after cuFile has accepted the operation.
            with request._lock:
                with self._pending_lock:
                    self._pending[request] = None
                native.submit(
                    self._cufile,
                    self._handle,  # type: ignore[arg-type]
                    buffer,
                    size,
                    offset,
                    stream,
                    writing,
                    self._count_pool,  # type: ignore[arg-type]
                )
        return request

    def read_async(self, size: int | None = None, *, offset: int = 0, stream: Stream) -> tuple[Buffer, _Request]:
        """Allocate a GPU buffer and enqueue a positional read on `stream`.

        The buffer has capacity `size` bytes, or the current file size minus
        `offset` (at least zero) when `size` is omitted. Returns the buffer
        and a request whose `result()` gives the valid byte count. The buffer
        retains its full capacity after a short read.
        """
        self._check_access(writing=False)
        offset = operator.index(offset)
        if not 0 <= offset <= _MAX_OFFSET:
            raise ValueError("the file offset must fit in a nonnegative signed 64-bit offset")
        size = max(0, self.size - offset) if size is None else operator.index(size)
        if not 0 <= size <= _MAX_OFFSET - offset:
            raise ValueError("invalid read size for the file offset")
        buffer = _allocate(size, stream)
        return buffer, self.readinto_async(buffer, offset=offset, stream=stream)

    def readinto_async(self, buffer: Buffer, *, offset: int = 0, stream: Stream, size: int | None = None) -> _Request:
        """Queue a positional GPU read on `stream` and return its request.

        `size` defaults to the buffer's byte size. Subsequent CUDA work in
        this stream runs after the transfer. Use the request's `result()`
        to check for errors and get the number of valid bytes.
        """
        return self._transfer(buffer, offset, size, writing=False, stream=stream)

    def write_async(self, buffer: Buffer, *, offset: int = 0, stream: Stream, size: int | None = None) -> _Request:
        """Queue a positional GPU write on `stream` and return its request.

        `size` defaults to the buffer's byte size. Use the request's
        `result()` to check for errors and get the number of bytes written.
        """
        return self._transfer(buffer, offset, size, writing=True, stream=stream)

    def writeinto_async(self, buffer: Buffer, *, offset: int = 0, stream: Stream, size: int | None = None) -> _Request:
        """Write the supplied GPU buffer into this file, as in `write_async()`."""
        return self.write_async(buffer, offset=offset, stream=stream, size=size)

    def read_batch(self, entries: Iterable[_FileEntry]) -> list[int]:
        """Read a batch of `(buffer, offset[, size])` entries from this file.

        Waits for the batch and returns byte counts in input order. See
        :func:`cuda.core.storage.read_batch` for ordering and error semantics.
        """
        from ._batch import read_batch

        self._check_access(writing=False)
        return read_batch((self, *entry) for entry in entries)

    def write_batch(self, entries: Iterable[_FileEntry]) -> list[int]:
        """Write a batch of `(buffer, offset[, size])` entries into this file.

        Waits for the batch and returns byte counts in input order. See
        :func:`cuda.core.storage.write_batch` for ordering and error semantics.
        """
        from ._batch import write_batch

        self._check_access(writing=True)
        return write_batch((self, *entry) for entry in entries)

    def streaming(self, stream: Stream | None) -> _StreamFile:
        """Return a file-like view with an independent cursor at byte zero.

        Pass a CUDA stream or explicitly pass `None` to create one, as in
        :func:`open`. Closing the view closes a stream it created and
        leaves the backing file open. Closing the file invalidates the view.
        """
        if self.closed:
            raise ValueError("I/O on a closed cuFile file")
        owns_stream = stream is None
        if stream is None:
            stream = _new_stream()
        else:
            _validate_stream(stream)
        return _StreamFile(self, stream, owns_file=False, owns_stream=owns_stream)


class _StreamFile:
    """A binary file-like object whose GPU I/O is ordered on one CUDA stream."""

    def __init__(self, file: _File, stream: Stream, *, owns_file: bool, owns_stream: bool):
        self._file = file
        self._stream = stream
        self._owns_file = owns_file
        self._owns_stream = owns_stream
        self._closed = False
        self._position = 0

    @property
    def stream(self) -> Stream:
        """The CUDA stream on which this object's I/O is submitted."""
        return self._stream

    @property
    def closed(self) -> bool:
        """Whether this object or its backing file has been closed."""
        return self._closed or self._file.closed

    def __enter__(self) -> _StreamFile:
        self.tell()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        try:
            self.close()
        except Exception as cleanup:
            if exc is None:
                raise
            exc.add_note(f"closing the streamed cuFile file failed: {cleanup}")

    def close(self) -> None:
        """Close this view and any file or stream it owns."""
        try:
            if self._owns_file:
                self._file.close()
        finally:
            if not self._owns_file or self._file.closed:
                self._closed = True
                if self._owns_stream:
                    self._stream.close()

    def tell(self) -> int:
        """Return the current byte offset."""
        if self.closed:
            raise ValueError("I/O on a closed cuFile file")
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        """Move the cursor and return the new byte offset.

        `whence` can be `os.SEEK_SET`, `os.SEEK_CUR`, or `os.SEEK_END`.
        Seeking beyond EOF is allowed.
        """
        position = self.tell()
        offset = operator.index(offset)
        whence = operator.index(whence)
        if whence == os.SEEK_CUR:
            offset += position
        elif whence == os.SEEK_END:
            offset += self._file.size
        elif whence != os.SEEK_SET:
            raise ValueError("invalid whence")
        if not 0 <= offset <= _MAX_OFFSET:
            raise ValueError("the file position must fit in a nonnegative signed 64-bit offset")
        self._position = offset
        return offset

    def readinto(self, buffer: Buffer, size: int | None = None) -> int:
        """Read into a GPU buffer at the cursor, then advance by bytes read.

        Enqueues the read on this object's stream and waits for it. Returns
        the actual byte count, which can be less than `size` at EOF.
        The default is the buffer's byte size.
        """
        request = self._file.readinto_async(buffer, offset=self.tell(), stream=self._stream, size=size)
        count = request.result()
        self._position += count
        return count

    def read(self, size: int = -1) -> Buffer:
        """Read up to `size` bytes into a new GPU buffer and advance the cursor.

        A size of `-1` reads to the current EOF. The returned buffer's
        byte size is the actual number read, including zero at EOF.
        """
        size = operator.index(size)
        buffer, request = self._file.read_async(None if size == -1 else size, offset=self.tell(), stream=self._stream)
        count = request.result()
        _resize_output(buffer, count)
        self._position += count
        return buffer

    def readuntil(self, end: int | None = None) -> Buffer:
        """Read up to the exclusive byte offset `end` into a new GPU buffer.

        If `end` is omitted, use the current file size. The cursor advances
        by the actual bytes read. When it is already at or beyond `end`,
        return a zero-byte buffer.
        """
        if end is None:
            return self.read()
        end = operator.index(end)
        if not 0 <= end <= _MAX_OFFSET:
            raise ValueError("end must fit in a nonnegative signed 64-bit offset")
        return self.read(max(0, end - self.tell()))

    def write(self, buffer: Buffer, size: int | None = None) -> int:
        """Write from a GPU buffer at the cursor and advance by bytes written.

        Enqueues the write on this object's stream and waits for it. The
        default size is the buffer's byte size. Returns the actual byte count.
        """
        request = self._file.write_async(buffer, offset=self.tell(), stream=self._stream, size=size)
        count = request.result()
        self._position += count
        return count
