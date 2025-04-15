"""Util functions for reporting device (GPU, CPU) stats."""

import dataclasses

import psutil
import pynvml
import torch


@dataclasses.dataclass
class GpuStats:
  """Holds information about GPU usage stats.

  For memory related, see
  https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
  """

  # device id
  device_id: int
  # memory reserved.
  reserved: float
  # memory occupied.
  occupied: float
  # memory reserved, but not used.
  unused: float
  # nvidia-smi usually reports more memory usages than pytorch (for driver,
  # kernel and etc). `smi_diff` tracks this difference.
  smi_diff: float
  # Gpu utilization.
  util: float

  # Allows unpacking operation like
  # device_id, reserved, occupied, unused, smi_diff, util = GpuStats(...)
  # See https://stackoverflow.com/a/70753113
  def __iter__(self):
    return iter(dataclasses.astuple(self))


def gpu_stats() -> GpuStats:
  """Reports GPU memory usage and utilization."""
  # See https://pytorch.org/docs/stable/notes/cuda.html#memory-management
  bytes_per_gb = 1024.0**3
  device = torch.cuda.current_device()
  occupied = torch.cuda.memory_allocated(device) / bytes_per_gb
  reserved = torch.cuda.memory_reserved(device) / bytes_per_gb
  unused = reserved - occupied

  def smi_mem(device):
    try:
      pynvml.nvmlInit()
      handle = pynvml.nvmlDeviceGetHandleByIndex(device)
      info = pynvml.nvmlDeviceGetMemoryInfo(handle)
      return info.used / bytes_per_gb
    except pynvml.NVMLError:
      return 0.0

  mem_used_smi = smi_mem(device)
  smi_diff = mem_used_smi - reserved

  util = torch.cuda.utilization(device)
  return GpuStats(device, reserved, occupied, unused, smi_diff, util)


def gpu_stats_str(stats: GpuStats | None = None) -> str:
  if stats is None:
    stats = gpu_stats()
  device, reserved, occupied, unused, smi_diff, util = stats
  return (
      f"GPU ({device=}) memory: {reserved:.2f}({occupied=:.2f}, {unused=:.2f}),"
      f" {smi_diff=:.2f} GB. Utilization: {util:.2f}%"
  )


@dataclasses.dataclass
class CpuStats:
  """Holds information about CPU usage stats."""

  # Total CPU virtual memory i.e. virtual memory allocated + unallocated.
  total_virtual_mem: float
  # CPU virtual memory available for use.
  unallocated_virtual_mem: float
  # CPU virtual memory already used.
  allocated_virtual_mem: float
  # Total CPU swap memory i.e. swap memory allocated + unallocated.
  total_swap_mem: float
  # CPU swap memory available for use.
  unallocated_swap_mem: float
  # CPU swap memory already used.
  allocated_swap_mem: float
  # CPU utilization percentage.
  utilization: float


def cpu_stats() -> CpuStats:
  """Reports CPU memory usage and utilization."""

  # https://psutil.readthedocs.io/en/latest/#memory
  gb = 1024.0**3
  vmem = psutil.virtual_memory()
  vmem_total = vmem.total / gb
  vmem_available = vmem.available / gb
  vmem_used = vmem_total - vmem_available
  smem = psutil.swap_memory()
  swap_total = smem.total / gb
  swap_free = smem.free / gb
  swap_used = smem.used / gb
  # https://psutil.readthedocs.io/en/latest/#psutil.cpu_percent
  cpu_util = psutil.cpu_percent(interval=1e-6)
  return CpuStats(
      total_virtual_mem=vmem_total,
      unallocated_virtual_mem=vmem_available,
      allocated_virtual_mem=vmem_used,
      total_swap_mem=swap_total,
      unallocated_swap_mem=swap_free,
      allocated_swap_mem=swap_used,
      utilization=cpu_util,
  )


def cpu_stats_str(stats: CpuStats | None = None) -> str:
  """Returns a string representation of the CPU stats."""

  if stats is None:
    stats = cpu_stats()
  total, occupied, unused = (
      stats.total_virtual_mem,
      stats.allocated_virtual_mem,
      stats.unallocated_virtual_mem,
  )
  virtual_mem = (
      f"CPU virtual memory: {total:.2f}({occupied=:.2f}, {unused=:.2f}) GB"
  )
  total, occupied, unused = (
      stats.total_swap_mem,
      stats.allocated_swap_mem,
      stats.unallocated_swap_mem,
  )
  swap_mem = f"CPU swap memory: {total:.2f}({occupied=:.2f}, {unused=:.2f}) GB"
  percent = stats.utilization
  return f"{virtual_mem} {swap_mem} CPU Utilization: {percent:.2f}%"
