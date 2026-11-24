# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
import sys
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from cuda import bindings
from cuda.bindings import driver
from cuda.core import Device, storage
from cuda.core.storage import _file, open_file

pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="cuFile requires Linux"),
    pytest.mark.thread_unsafe(reason="mock storage tests patch module-level CUDA bindings"),
]


@pytest.fixture
def fake_cufile(monkeypatch):
    """Use ordinary file descriptors to exercise the Python file and cursor contract."""

    class FakeBuffer:
        is_closed = False
        is_device_accessible = True
        is_host_accessible = False

        def __init__(self, size):
            self.size = size
            self.handle = bytearray(size)

    def read(fd, dst, size, offset, _):
        data = os.pread(fd, size, offset)
        dst[: len(data)] = data
        return len(data)

    def write(fd, src, size, offset, _):
        return os.pwrite(fd, src[:size], offset)

    class Transfer:
        def submit(self, cufile, fd, buffer, size, offset, stream, writing, count_pool):
            if buffer.is_host_accessible:
                raise ValueError("cuFile I/O requires device-resident GPU memory")
            io = cufile.write if writing else cufile.read
            self.count = io(fd, buffer.handle, size, offset, 0)

        def complete(self, wait):
            return getattr(self, "count", 0) if wait else None

    def descriptor():
        desc = SimpleNamespace(handle=SimpleNamespace())
        desc.ptr = desc
        return desc

    backend = SimpleNamespace(
        get_version=lambda: 1090,
        Descr=descriptor,
        FileHandleType=SimpleNamespace(OPAQUE_FD=1),
        handle_register=lambda desc: desc.handle.fd,
        handle_deregister=lambda _handle: None,
        read=read,
        write=write,
    )
    monkeypatch.setattr(bindings, "cufile", backend, raising=False)
    monkeypatch.setattr(_file, "Buffer", FakeBuffer)
    monkeypatch.setattr(_file, "_Transfer", Transfer)
    monkeypatch.setattr(_file, "_validate_stream", lambda _stream: None)
    monkeypatch.setattr(_file, "_allocate", lambda size, _stream: FakeBuffer(size))
    monkeypatch.setattr(_file, "_resize_output", lambda buffer, size: setattr(buffer, "size", size))
    return backend, FakeBuffer


def test_cursor_reads_and_independent_views(fake_cufile, tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"012345")
    stream = object()
    with storage.open(path, stream=stream) as reader:
        backing_file = reader._file
        assert reader.stream is stream
        assert reader.read(2).handle == b"01"
        assert reader.readuntil(4).handle == b"23"
        tail = reader.read(8)
        assert tail.size == 2
        assert tail.handle[: tail.size] == b"45"
        assert reader.tell() == 6
        assert reader.readuntil().size == 0
        assert reader.seek(-3, os.SEEK_END) == 3
        assert reader.read().handle == b"345"
    assert reader.closed
    assert backing_file.closed

    with open_file(path) as file:
        with file.streaming(stream) as first, file.streaming(stream) as second:
            first.read(3)
            assert second.tell() == 0
        assert first.closed
        assert not file.closed


def test_positional_operations_and_close(fake_cufile, tmp_path):
    _, Buffer = fake_cufile
    path = tmp_path / "data.bin"
    src = Buffer(4)
    src.handle[:] = b"cuda"
    stream = object()
    with open_file(path, "w+b") as file:
        assert file.write_async(src, stream=stream).result() == 4
        request = file.writeinto_async(src, offset=4, stream=stream, size=2)
        assert not request.is_done
        assert file.size == 6
        dst = Buffer(4)
        assert file.readinto_async(dst, offset=3, stream=stream).result() == 3
        assert dst.handle[:3] == b"acu"
        allocated, suffix = file.read_async(offset=3, stream=stream)
        assert allocated.size == suffix.result() == 3
        oversized, short = file.read_async(8, offset=3, stream=stream)
        assert oversized.size == 8
        assert short.result() == 3
        empty, eof = file.read_async(offset=9, stream=stream)
        assert empty.size == eof.result() == 0
        with pytest.raises(ValueError, match="buffer size"):
            file.readinto_async(dst, stream=stream, size=5)
        with pytest.raises(TypeError, match="cuda.core.Buffer"):
            file.readinto_async(bytearray(4), stream=stream)
        with pytest.raises(ValueError, match="device-resident"):
            dst.is_host_accessible = True
            file.readinto_async(dst, stream=stream)
    assert request.is_done
    assert request.result() == 2
    assert path.read_bytes() == b"cudacu"


def test_registration_failure_preserves_existing_file(fake_cufile, monkeypatch, tmp_path):
    backend, _ = fake_cufile
    path = tmp_path / "data.bin"
    path.write_bytes(b"keep")
    seen = []

    def fail_registration(desc):
        seen.append(desc.handle.fd)
        raise OSError("registration failed")

    monkeypatch.setattr(backend, "handle_register", fail_registration)
    with pytest.raises(OSError, match="registration failed"):
        open_file(path, "wb")
    assert path.read_bytes() == b"keep"
    with pytest.raises(OSError):
        os.fstat(seen[0])


def test_binary_modes(fake_cufile, tmp_path):
    with pytest.raises(ValueError, match="binary mode"):
        open_file(tmp_path / "data", "r")


def test_owned_stream_is_explicit(fake_cufile, monkeypatch, tmp_path):
    closed_streams = []
    monkeypatch.setattr(_file, "_new_stream", lambda: SimpleNamespace(close=lambda: closed_streams.append(1)))
    path = tmp_path / "data"
    path.touch()
    with pytest.raises(TypeError, match="stream"):
        storage.open(path)
    with open_file(path) as file:
        with pytest.raises(TypeError, match="stream"):
            file.streaming()
        with file.streaming(None):
            assert not closed_streams
        assert closed_streams == [1]
        assert not file.closed
    with storage.open(path, stream=None):
        assert closed_streams == [1]
    assert closed_streams == [1, 1]


def test_close_waits_after_submission_error(fake_cufile, monkeypatch, tmp_path):
    backend, Buffer = fake_cufile
    steps = []

    class Transfer:
        def submit(self, *args):
            steps.append("submitted")
            raise RuntimeError("event record failed")

        def complete(self, wait):
            assert wait
            steps.append("completed")
            return 4

    monkeypatch.setattr(_file, "_Transfer", Transfer)
    monkeypatch.setattr(backend, "handle_deregister", lambda _handle: steps.append("deregistered"))
    with open_file(tmp_path / "data", "wb") as file, pytest.raises(RuntimeError, match="event record failed"):
        file.write_async(Buffer(4), offset=0, stream=object())
    assert steps == ["submitted", "completed", "deregistered"]


@pytest.fixture
def cufile_device(tmp_path):
    from cuda.pathfinder import DynamicLibNotFoundError

    try:
        if driver.cuInit(0)[0] != driver.CUresult.CUDA_SUCCESS:
            pytest.skip("CUDA driver is unavailable")
    except DynamicLibNotFoundError as exc:
        pytest.skip(str(exc))
    cufile = pytest.importorskip("cuda.bindings.cufile")
    try:
        cufile.get_version()
    except DynamicLibNotFoundError as exc:
        pytest.skip(str(exc))

    device = Device()
    device.set_current()
    try:
        with open_file(tmp_path / "probe.bin", "w+b"):
            pass
    except cufile.cuFileError as exc:
        if exc.status in (
            cufile.OpError.IO_NOT_SUPPORTED,
            cufile.OpError.PLATFORM_NOT_SUPPORTED,
            cufile.OpError.DRIVER_NOT_INITIALIZED,
        ):
            pytest.skip(str(exc))
        raise
    return device


def test_stream_ordered_cufile(tmp_path, cufile_device):
    device = cufile_device
    stream = device.create_stream()
    path = tmp_path / "data.bin"
    other = tmp_path / "other.bin"
    with open_file(path, "w+b") as file, device.allocate(4096, stream=stream) as src:
        src.fill(ord("a"), stream=stream)
        first = file.write_async(src, offset=0, stream=stream)
        src.fill(ord("b"), stream=stream)
        second = file.write_async(src, offset=4096, stream=stream)
        assert first.result() == second.result() == 4096
        with file.streaming(stream) as reader:
            with reader.readuntil() as data, storage.open(other, "wb", stream=stream) as writer:
                assert writer.write(data) == 8192
            with reader.read() as eof:
                assert eof.size == 0
    stream.sync()
    assert other.read_bytes() == b"a" * 4096 + b"b" * 4096

    with storage.open(path, stream=None) as reader, reader.read(4096) as first_block:
        owned_stream = reader.stream
        assert first_block.size == 4096
    assert owned_stream.is_closed
    assert not stream.is_closed


def test_cufile_batch_across_files(tmp_path, cufile_device):
    device = cufile_device
    stream = device.create_stream()
    first_path = tmp_path / "first.bin"
    second_path = tmp_path / "second.bin"
    with ExitStack() as stack:
        first = stack.enter_context(open_file(first_path, "w+b"))
        second = stack.enter_context(open_file(second_path, "w+b"))
        src = stack.enter_context(device.allocate(4096, stream=stream))
        a = stack.enter_context(device.allocate(4096, stream=stream))
        b = stack.enter_context(device.allocate(4096, stream=stream))
        src.fill(ord("x"), stream=stream)
        stream.sync()
        assert storage.write_batch([(first, src, 0), (second, src, 4096)]) == [4096, 4096]
        assert storage.read_batch([(second, b, 4096), (first, a, 0, 0), (first, a, 0)]) == [4096, 0, 4096]
        assert first.write_batch([(b, 4096), (a, 8192)]) == [4096, 4096]
        assert first.read_batch([(a, 8192), (b, 4096)]) == [4096, 4096]
    assert first_path.read_bytes() == b"x" * 12288
    assert second_path.read_bytes() == b"\0" * 4096 + b"x" * 4096


def test_transfer_count_slots_reuse_and_overflow(cufile_device, monkeypatch):
    from cuda.core.storage._transfer import _CountPool, _Transfer

    count_type = ctypes.c_ssize_t
    slot_size = ctypes.sizeof(count_type)
    monkeypatch.setenv("CUDA_PYTHON_CUFILE_PINNED_SLAB_SIZE", str(2 * slot_size))
    pool = _CountPool()
    addresses = []

    def submit_read(_file, _buffer, _size, _offset, _buffer_offset, count, _stream):
        assert count_type.from_address(count).value == 0
        addresses.append(count)
        count_type.from_address(count).value = len(addresses)

    fake = SimpleNamespace(read_async=submit_read, cuFileError=OSError)
    device = cufile_device
    stream = device.create_stream()
    try:
        with device.allocate(4096, stream=stream) as buffer:
            first, second, overflow, reused = (_Transfer() for _ in range(4))
            for transfer in (first, second, overflow):
                transfer.submit(fake, 0, buffer, 4096, 0, stream, False, pool)
            assert addresses[1] == addresses[0] + slot_size
            assert len(set(addresses)) == 3
            assert first.complete(True) == 1
            reused.submit(fake, 0, buffer, 4096, 0, stream, False, pool)
            assert addresses[3] == addresses[0]
            assert second.complete(True) == 2
            assert overflow.complete(True) == 3
            assert reused.complete(True) == 4
    finally:
        stream.close()


@pytest.mark.parametrize("accepted", [False, True])
def test_transfer_count_slot_after_submission_error(cufile_device, monkeypatch, accepted):
    from cuda.core.storage._transfer import _CountPool, _Transfer

    count_type = ctypes.c_ssize_t
    monkeypatch.setenv("CUDA_PYTHON_CUFILE_PINNED_SLAB_SIZE", str(ctypes.sizeof(count_type)))
    pool = _CountPool()
    addresses = []
    error_type = RuntimeError if accepted else OSError

    def submit_read(_file, _buffer, _size, _offset, _buffer_offset, count, _stream):
        addresses.append(count)
        if len(addresses) == 1:
            if accepted:
                count_type.from_address(count).value = 1
            raise error_type("submission failed")
        count_type.from_address(count).value = 2

    fake = SimpleNamespace(read_async=submit_read, cuFileError=OSError)
    device = cufile_device
    stream = device.create_stream()
    try:
        with device.allocate(4096, stream=stream) as buffer:
            first, second = _Transfer(), _Transfer()
            with pytest.raises(error_type, match="submission failed"):
                first.submit(fake, 0, buffer, 4096, 0, stream, False, pool)
            second.submit(fake, 0, buffer, 4096, 0, stream, False, pool)
            assert first.complete(True) == int(accepted)
            assert second.complete(True) == 2
            assert (addresses[0] != addresses[1]) == accepted
    finally:
        stream.close()


def test_batch_path_handles_close_on_validation_error(fake_cufile, monkeypatch, tmp_path):
    backend, Buffer = fake_cufile
    opened, closed = [], []
    first_path = tmp_path / "first.bin"
    first_path.touch()
    path = tmp_path / "owned.bin"
    path.touch()

    def register(desc):
        opened.append(desc.handle.fd)
        return desc.handle.fd

    monkeypatch.setattr(backend, "handle_register", register)
    monkeypatch.setattr(backend, "handle_deregister", closed.append)
    with open_file(first_path) as file:
        with pytest.raises(ValueError, match="file byte range"):
            storage.read_batch([(file, Buffer(0), 0), (path, Buffer(0), 0), (os.fsencode(path), Buffer(0), -1)])
        assert len(opened) == 2
        assert closed == opened[1:]
        assert not file.closed
    assert closed == opened[::-1]


def test_cufile_batch_paths_preserve_existing_data(tmp_path, cufile_device):
    device = cufile_device
    stream = device.create_stream()
    existing = tmp_path / "existing.bin"
    missing = tmp_path / "new.bin"
    existing.write_bytes(b"a" * 12288)
    with open_file(existing, "rb") as file, device.allocate(4096, stream=stream) as data:
        data.fill(ord("b"), stream=stream)
        stream.sync()
        assert storage.write_batch([(existing, data, 4096), (str(missing), data, 0)]) == [4096, 4096]
        assert storage.read_batch([(os.fsencode(missing), data, 0), (file, data, 0, 0)]) == [4096, 0]
        assert not file.closed
    assert existing.read_bytes() == b"a" * 4096 + b"b" * 4096 + b"a" * 4096
    assert missing.read_bytes() == b"b" * 4096
