# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterable
from contextlib import ExitStack
from ctypes import addressof, c_ssize_t, c_uint

from cuda.core._memory._buffer import Buffer

from ._file import _File, _open_file, _Path
from ._transfer import _BatchOwners

_FileOrPath = _File | _Path
_Entry = tuple[_FileOrPath, Buffer, int] | tuple[_FileOrPath, Buffer, int, int | None]


def read_batch(entries: Iterable[_Entry]) -> list[int]:
    """Read several file ranges into GPU buffers in one cuFile batch.

    Each entry is ``(file, buffer, offset)`` or ``(file, buffer, offset, size)``.
    A file can be a path or a handle from :func:`cuda.core.storage.open_file`.
    Handles opened for paths are closed when the call exits. Offsets and sizes
    are in bytes. An omitted or ``None`` size uses the buffer's full size.
    Zero-size entries return zero.

    Waits on the host and returns byte counts in input order. The operations
    may execute in any order. Use independent ranges and buffers where
    accesses would conflict. Have a CUDA context current, and synchronize any
    prior CUDA work that uses these buffers before calling. Keep the files
    and buffers open through the call. A failed entry raises after the batch
    completes, and other entries may already have transferred data.
    """
    return _batch(entries, writing=False)


def write_batch(entries: Iterable[_Entry]) -> list[int]:
    """Write from GPU buffers into several file ranges in one cuFile batch.

    Entries, byte counts, and synchronization follow :func:`read_batch`.
    Each entry supplies the GPU buffer containing its source bytes. Paths
    are opened for positional writes. Missing files are created and existing
    bytes outside the written ranges are preserved.
    """
    return _batch(entries, writing=True)


def _batch(entries: Iterable[_Entry], *, writing: bool) -> list[int]:
    with ExitStack() as files:
        opened: dict[bytes, _File] = {}
        counts: list[int] = []
        active = []
        for entry in entries:
            target, buffer, offset, *sizes = entry
            if len(sizes) > 1:
                raise TypeError("expected (file or path, buffer, offset[, size])")
            if isinstance(target, _File):
                file = target
            else:
                path = os.fsencode(target)
                file = opened.get(path)
                if file is None:
                    mode = "wb" if writing else "rb"
                    file = files.enter_context(_open_file(path, mode, direct=False, truncate=False))
                    opened[path] = file
            offset, size = file._prepare(buffer, offset, sizes[0] if sizes else None, writing=writing)
            if (buffer.size or buffer.handle) and (not buffer.is_device_accessible or buffer.is_host_accessible):
                raise ValueError("cuFile I/O requires device-resident GPU memory")
            if size:
                active.append((file, buffer, offset, size, len(counts)))
            counts.append(0)
        if active:
            _submit_batch(active, counts, writing=writing)
        return counts


def _submit_batch(active: list[tuple[_File, Buffer, int, int, int]], counts: list[int], *, writing: bool) -> None:
    cufile = active[0][0]._cufile
    count = len(active)
    params = cufile.IOParams(count)
    events = cufile.IOEvents(count)
    owners = _BatchOwners(active, params, events)
    opcode = cufile.Opcode.WRITE if writing else cufile.Opcode.READ
    for index, (file, buffer, offset, size, _) in enumerate(active):
        param = params[index]
        param.mode = cufile.BatchMode.BATCH
        param.fh = file._handle
        param.opcode = opcode
        param.cookie = index + 1
        param.u.batch.dev_ptr_base = buffer.handle
        param.u.batch.file_offset = offset
        param.u.batch.dev_ptr_offset = 0
        param.u.batch.size_ = size

    returned = c_uint()
    completed: set[int] = set()
    failures = []
    batch = cufile.batch_io_set_up(count)
    try:
        cufile.batch_io_submit(batch, count, params.ptr, 0)
        while len(completed) < count:
            returned.value = count - len(completed)
            cufile.batch_io_get_status(batch, 1, addressof(returned), events.ptr, 0)
            for i in range(returned.value):
                event = events[i]
                index = event.cookie - 1
                if not 0 <= index < count or index in completed:
                    raise RuntimeError("internal cuda.core error, please report: invalid cuFile batch completion")
                completed.add(index)
                entry_index = active[index][4]
                result = c_ssize_t(event.ret).value
                if event.status == cufile.Status.COMPLETE and result >= 0:
                    counts[entry_index] = result
                else:
                    failures.append(f"entry {entry_index}: {event.status.name} (cuFile result {result})")
    except BaseException as error:
        try:
            cufile.batch_io_destroy(batch)
        except Exception as cleanup:
            if len(completed) != count:
                for file, *_ in active:
                    file._unresolved_batch = True
                owners.abandon()
            error.add_note(f"destroying the cuFile batch failed: {cleanup}")
        raise
    cufile.batch_io_destroy(batch)
    if failures:
        raise OSError("cuFile batch failed: " + "; ".join(failures))
