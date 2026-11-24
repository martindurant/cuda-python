# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os

cimport cython
from cpython.ref cimport Py_DECREF, Py_INCREF
from libc.stdint cimport int64_t, intptr_t
from libcpp.vector cimport vector

from cuda.bindings cimport cydriver
from cuda.core._context cimport Context
from cuda.core._event cimport Event
from cuda.core._memory._buffer cimport Buffer, Buffer_check_open, Buffer_from_deviceptr_handle
from cuda.core._rt cimport (
    ContextHandle,
    DevicePtrHandle,
    EventHandle,
    MemoryPoolHandle,
    StreamHandle,
    as_cu,
    attach_rollback_failure,
    clear_last_error_detail,
    context_get_device,
    create_event_handle,
    deviceptr_alloc_from_pool,
    deviceptr_alloc_host,
    enter_context,
    exit_context,
    get_current_context,
    get_device_mempool,
    get_last_error,
    get_primary_context,
)
from cuda.core._stream cimport Stream, Stream_check_open, Stream_is_default_token
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN


def _new_stream():
    from cuda.core import Device

    Device(0)  # Initialize the driver without changing the current context.
    cdef ContextHandle context
    cdef cydriver.CUdevice device
    with nogil:
        context = get_current_context()
        if not context:
            HANDLE_RETURN(get_last_error())
            context = get_primary_context(0)
            if not context:
                HANDLE_RETURN(get_last_error())
        HANDLE_RETURN(context_get_device(context, &device))
    return Stream._init(ctx=Context._from_handle(Context, context, device), device_id=device)


cdef class _BatchOwners:
    cdef vector[DevicePtrHandle] _buffers
    cdef object _owners

    def __init__(self, entries, params, events):
        cdef Buffer buffer
        self._owners = entries, params, events
        for entry in entries:
            buffer = entry[1]
            self._buffers.push_back(buffer._h_ptr)

    def abandon(self):
        # Completion and destruction both failed. Preserve the live native pointers.
        Py_INCREF(self)


def _validate_stream(Stream stream not None):
    Stream_check_open(stream)
    # Default stream tokens cannot identify the same queue from every thread.
    if Stream_is_default_token(stream):
        raise ValueError("cuFile requires an explicit CUDA stream")


cdef void _check_capture(cydriver.CUstream stream) except *:
    cdef cydriver.CUstreamCaptureStatus status
    with nogil:
        HANDLE_RETURN(cydriver.cuStreamIsCapturing(stream, &status))
    if status != cydriver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
        raise RuntimeError("cuFile I/O is unavailable during CUDA stream capture")


def _allocate(size_t size, Stream stream not None):
    _validate_stream(stream)
    _check_capture(as_cu(stream._h_stream))
    cdef MemoryPoolHandle pool
    cdef DevicePtrHandle allocation
    cdef Context context = stream.context
    with nogil:
        pool = get_device_mempool(context._device_id)
        if not pool:
            HANDLE_RETURN(get_last_error())
    if not pool:
        raise RuntimeError("internal cuda.core error, please report: missing device memory pool")
    with nogil:
        allocation = deviceptr_alloc_from_pool(size, pool, stream._h_stream)
    if not allocation:
        HANDLE_RETURN(get_last_error())
        raise RuntimeError("internal cuda.core error, please report: missing allocation handle")
    return Buffer_from_deviceptr_handle(allocation, size, None)


def _resize_output(Buffer buffer not None, size_t size):
    # This buffer was freshly allocated by the reader and has never been published.
    buffer._size = size


cdef class _CountSlab:
    """One pinned allocation shared by transfers in the same CUDA context."""

    # Keep the native context alive while its handle is used as a cache key.
    cdef ContextHandle _context
    cdef DevicePtrHandle _allocation
    cdef Py_ssize_t* _slots
    cdef Py_ssize_t _capacity, _used, _free
    cdef cython.pymutex _lock

    def __cinit__(self, Context context):
        cdef size_t size
        setting = "CUDA_PYTHON_CUFILE_PINNED_SLAB_SIZE"
        value = int(os.environ.get(setting, 512 * 1024))
        if value < sizeof(Py_ssize_t):
            raise ValueError(f"{setting} must be at least {sizeof(Py_ssize_t)} bytes")
        size = value
        self._context = context._h_context
        with nogil:
            self._allocation = deviceptr_alloc_host(size)
            if not self._allocation:
                HANDLE_RETURN(get_last_error())
        if not self._allocation:
            raise RuntimeError("internal cuda.core error, please report: missing cuFile count slab")
        self._slots = <Py_ssize_t*><intptr_t>as_cu(self._allocation)
        self._capacity = size // sizeof(Py_ssize_t)
        self._free = -1

    cdef Py_ssize_t* acquire(self) noexcept:
        cdef Py_ssize_t index
        with self._lock:
            if self._free >= 0:
                index = self._free
                self._free = self._slots[index]
            elif self._used < self._capacity:
                index = self._used
                self._used += 1
            else:
                return NULL
        return &self._slots[index]

    cdef void release(self, Py_ssize_t* slot) noexcept:
        # Use the idle slots themselves to store the freelist indices.
        with self._lock:
            slot[0] = self._free
            self._free = slot - self._slots


cdef class _CountPool:
    """Cache at most one slab per context for an open file."""

    cdef dict _slabs

    def __cinit__(self):
        self._slabs = {}

    cdef _CountSlab slab(self, Context context):
        key = <intptr_t>as_cu(context._h_context)
        slab = self._slabs.get(key)
        if slab is None:
            slab = self._slabs.setdefault(key, _CountSlab(context))
        return slab


cdef class _Transfer:
    """Native arguments and allocation owners for a single cuFile operation."""

    cdef DevicePtrHandle _buffer, _pinned_count
    cdef StreamHandle _stream
    cdef Event _event
    cdef _CountSlab _count_slab
    cdef size_t _size
    # libcufile runs on 64-bit Linux, where off_t is signed 64-bit.
    cdef int64_t _file_offset, _buffer_offset
    cdef Py_ssize_t* _count
    cdef bint _submitted

    def submit(self, cufile, intptr_t file, Buffer buffer not None,
               size_t size, int64_t offset, Stream stream not None, bint writing,
               _CountPool count_pool not None):
        _validate_stream(stream)
        Buffer_check_open(buffer)
        cdef Context stream_context = stream.context
        self._buffer = buffer._h_ptr
        self._stream = stream._h_stream
        self._size = size
        self._file_offset = offset
        cdef ContextHandle context = stream_context._h_context
        cdef cydriver.CUstream raw_stream = as_cu(self._stream)
        cdef cydriver.CUcontext previous
        cdef int changed
        cdef EventHandle event
        cdef cydriver.CUresult restore
        cdef _CountSlab slab
        _check_capture(raw_stream)

        with nogil:
            HANDLE_RETURN(enter_context(context, &previous, &changed))
        try:
            if not buffer.is_device_accessible or buffer.is_host_accessible:
                raise ValueError("cuFile I/O requires device-resident GPU memory")
            if buffer.device_id != stream_context._device_id:
                raise ValueError("buffer and cuFile stream must be on the same CUDA device")
            with nogil:
                event = create_event_handle(
                    context,
                    cydriver.CUevent_flags.CU_EVENT_DISABLE_TIMING | cydriver.CUevent_flags.CU_EVENT_BLOCKING_SYNC,
                    False, True, False, stream_context._device_id)
                if not event:
                    HANDLE_RETURN(get_last_error())
            if not event:
                raise RuntimeError("internal cuda.core error, please report: missing cuFile transfer event")
            slab = count_pool.slab(stream_context)
            self._count = slab.acquire()
            if self._count == NULL:
                # Keep cached pinned memory bounded at high queue depths.
                with nogil:
                    self._pinned_count = deviceptr_alloc_host(sizeof(Py_ssize_t))
                    if not self._pinned_count:
                        HANDLE_RETURN(get_last_error())
                if not self._pinned_count:
                    raise RuntimeError("internal cuda.core error, please report: missing cuFile transfer count")
                self._count = <Py_ssize_t*><intptr_t>as_cu(self._pinned_count)
            else:
                self._count_slab = slab
            self._count[0] = 0
            completion = Event._from_handle(event)

            # Publish ownership before submission. If recording the event or
            # restoring the context fails, close can still wait for the I/O.
            submit = cufile.write_async if writing else cufile.read_async
            submission_error = cufile.cuFileError
            Py_INCREF(self)
            self._submitted = True
            try:
                submit(file, <intptr_t>as_cu(self._buffer), <intptr_t>&self._size,
                       <intptr_t>&self._file_offset, <intptr_t>&self._buffer_offset,
                       <intptr_t>self._count, <intptr_t>raw_stream)
            except submission_error:
                self._submitted = False
                Py_DECREF(self)
                raise
            with nogil:
                HANDLE_RETURN(cydriver.cuEventRecord(as_cu(event), raw_stream))
            self._event = completion
        except:
            if not self._submitted:
                self._release_count()
            with nogil:
                restore = exit_context(previous, changed, cydriver.CUresult.CUDA_SUCCESS)
                if restore != cydriver.CUresult.CUDA_SUCCESS:
                    clear_last_error_detail()
                    attach_rollback_failure(b"cuCtxSetCurrent", restore, b"restoring the caller's context")
            raise
        with nogil:
            HANDLE_RETURN(exit_context(previous, changed, cydriver.CUresult.CUDA_SUCCESS))

    cdef void _release_count(self) noexcept:
        if self._count_slab is not None:
            self._count_slab.release(self._count)
            self._count = NULL

    def complete(self, bint wait):
        """Return the signed byte count, or None while pending."""
        if not self._submitted:
            return 0
        if self._event is None:
            if not wait:
                return None
            with nogil:
                HANDLE_RETURN(cydriver.cuStreamSynchronize(as_cu(self._stream)))
        elif wait:
            self._event.sync()
        elif not self._event.is_done:
            return None

        cdef object count = self._count[0]
        self._submitted = False
        self._release_count()
        Py_DECREF(self)
        return count
