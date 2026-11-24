.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

GPU file I/O
============

.. currentmodule:: cuda.core.storage

:func:`open_file` and :func:`open` transfer bytes between a regular filesystem
file and device memory through NVIDIA cuFile. They accept binary modes and device-resident
:class:`~cuda.core.Buffer` objects. The cuFile runtime is required on Linux.
cuFile can stage data through host memory internally, depending on its
configuration, the filesystem, and the hardware. Pass ``direct=True`` to
request opening the file with ``O_DIRECT``.

Direct GPU transfers
--------------------

``direct=True`` requests direct filesystem I/O. The path between storage
and GPU memory also depends on whether cuFile can use peer-to-peer DMA for
that GPU, controller, filesystem mount, and operation. A successful transfer
can still have used host staging.

For local NVMe on Linux, NVIDIA documents direct support for ext4 and XFS.
Use ext4 with ordered journaling. The host also needs a supported GDS driver
path, such as NVMe PCI P2PDMA with a compatible Linux kernel and NVIDIA open
driver, or a supported ``nvidia-fs`` setup. For an initial test, use an aligned
GPU buffer, file offset, and transfer size, all on 4 KiB boundaries. Test
writes over an existing allocated range because extending a file or writing
into a hole can change the available I/O path.

Run ``gdscheck -p`` to see the available paths, and use cuFile statistics for
the actual workload to distinguish GDS operations from POSIX fallback. See
NVIDIA's `NVMe P2PDMA setup guide
<https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/nvme-p2pdma-troubleshooting.html>`_
and `cuFile statistics documentation
<https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/gds-library-tracing.html>`_.

Positional transfers
--------------------

Use :func:`open_file` for independent transfers at explicit byte offsets.
The file's ``size`` attribute reports its current size in bytes. Wait for
any pending writes before relying on their effect on the size.

.. code-block:: python

   from cuda.core.storage import open_file

   with open_file("frames.bin", "rb") as file:
       frame_a, first = file.read_async(4096, offset=4096, stream=stream)
       second = file.readinto_async(frame_b, offset=8192, stream=stream)
       count_a = first.result()
       count_b = second.result()

``read_async`` returns a new buffer and its request. The buffer has the requested
capacity, or the file size minus the offset if no size was supplied. After a
short read, only the prefix identified by the request's byte count contains new
data. ``readinto_async`` reads into an existing buffer. ``write_async`` and
``writeinto_async`` both write from a supplied GPU buffer. Operations on existing
buffers default to the full buffer size. Pass ``size=`` to transfer a prefix.

An asynchronous transfer is ordered with CUDA work submitted before and after
it on the same explicit stream. Use CUDA events to establish dependencies
between streams. Calling ``request.result()`` waits for the transfer and
reports its byte count or error. A later CUDA operation on the same stream
can access the buffer once the transfer completes. The caller must account for
short reads and transfer errors before using data that depends on the result.

The open file retains pending transfers, including when the Python request is
discarded. Close the file to wait for outstanding transfers and release its
descriptor. Keep externally owned allocations valid through completion.
Stream I/O requires cuFile stream support and an explicit
:class:`~cuda.core.Stream` created for the buffer's device. Default stream
tokens and stream capture are unsupported.

The first nonempty stream transfer on an open file and CUDA context allocates
a pinned host slab for transfer byte counts. Set
``CUDA_PYTHON_CUFILE_PINNED_SLAB_SIZE`` to change its size in bytes from the
default of 512 KiB. Each in-flight transfer uses one slot. If all slots are
in use, additional transfers allocate their own count storage.

Batch transfers
---------------

Use ``file.read_batch()`` or ``file.write_batch()`` for several ranges of one
file. The top-level :func:`read_batch` and :func:`write_batch` accept paths
and open files in the same native cuFile batch. The functions close the files
they open for paths. Writing to a path creates a missing file and preserves
existing bytes outside the written ranges.

.. code-block:: python

   from cuda.core.storage import open_file, read_batch

   with open_file("a.bin") as a:
       counts = read_batch([(a, header, 0), ("b.bin", frame, 8192, 4096)])
       counts = a.read_batch([(header, 0), (frame, 4096)])

Batch calls wait on the host and return byte counts in input order. Operations
within a batch can run in any order. Synchronize earlier CUDA work on any
affected buffers before submitting a batch. The active cuFile configuration
determines the maximum number of nonempty entries in a batch.

Stream-bound files
------------------

Bind a stream when opening the file to get a file-like object with a byte
cursor. Each read or write submits the transfer to that stream, waits for its
result, and advances the cursor by the actual byte count.

.. code-block:: python

   from cuda.core import storage

   with storage.open("frames.bin", stream=None) as reader:
       header = reader.read(4096)
       first_frame = reader.readuntil(8192)
       remaining = reader.read()

Both ``read()`` and ``readuntil()`` allocate and return a GPU buffer. An
omitted limit reads up to the current file size. The limit for ``read(size)``
is a byte count. The limit for ``readuntil(end)`` is an exclusive file offset.
The returned buffer's size is the actual number of bytes read. Use
``readinto(buffer)`` to reuse a buffer or ``write(buffer)`` to write its bytes.
The cursor supports ``seek()`` and ``tell()``.

Both ``storage.open()`` and ``file.streaming()`` require an explicit stream
argument. Pass ``None`` to create an owned CUDA stream on the current context,
or on device 0's primary context when none is current. Closing a file from
``storage.open()`` also closes its backing file. Each ``file.streaming()``
view starts at byte zero and borrows the already-open file. Closing the
backing file invalidates all its views. The caller must coordinate cursor
mutations and closing a file with other operations on it.

API reference
-------------

.. autofunction:: open_file
.. autofunction:: open
.. autofunction:: read_batch
.. autofunction:: write_batch
