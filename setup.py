"""FusionAttention build: plain-nvcc shared library, no torch headers.

The kernel .so is pure CUDA + extern "C" (csrc/fusionattn.cu) and is loaded via
ctypes at runtime, so it compiles and works regardless of the installed
torch version — an explicit design goal given the sm_61 torch downgrade.

    python setup.py build_py   # -> build_lib/fusionattn/_fusionattn<EXT_SUFFIX>

Env knobs:
    FUSEW4_CUDA_ARCHS reuse? no -> FUSIONATTN_CUDA_ARCHS (';'-separated,
        default "6.1;7.5;8.0;8.6;8.9;9.0")
    FUSIONATTN_NVCC   nvcc binary (default nvcc on PATH)
    FUSIONATTN_CCBIN  host compiler for nvcc -ccbin (default: g++-13 if present)
    FUSIONATTN_SKIP_BUILD=1  skip native build (python-only install)
"""

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "csrc" / "fusionattn.cu"
SHIM = ROOT / "csrc" / "include_shim"
DEFAULT_ARCHS = "6.1;7.5;8.0;8.6;8.9;9.0"



def _env(name, default=""):
    return os.environ.get(name, default)


class NativeBuildPy(build_py):
    def run(self):
        super().run()
        if _env("FUSIONATTN_SKIP_BUILD") == "1":
            self.announce("fusionattn: native build skipped (FUSIONATTN_SKIP_BUILD=1)")
            return
        libdir = Path(self.build_lib) / "fusionattn"
        libdir.mkdir(parents=True, exist_ok=True)
        ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        out = libdir / f"_fusionattn{ext}"
        self._build_kernels(out)

    def _build_kernels(self, out: Path):
        nvcc = _env("FUSIONATTN_NVCC", "nvcc")
        archs = _env("FUSIONATTN_CUDA_ARCHS", DEFAULT_ARCHS)
        ccbin = _env("FUSIONATTN_CCBIN", "")
        if not ccbin and shutil.which("g++-13"):
            ccbin = "g++-13"  # nvcc <= 12.9 rejects gcc-15 hosts
        cmd = [
            nvcc, "-std=c++17", "-O3", "-shared", "-Xcompiler", "-fPIC",
            "-isystem", str(SHIM),
        ]
        for a in archs.split(";"):
            a = a.strip()
            if a:
                num = a.replace("sm_", "").replace(".", "")
                cmd += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
        if ccbin:
            cmd += ["-ccbin", ccbin]
        cmd += ["-Wno-deprecated-gpu-targets", str(SRC), "-o", str(out)]
        self.announce("fusionattn: " + " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(ROOT))


setup(
    name="fusionattn",
    version="0.1.0",
    description="CUDA-core FA2 paged attention backend for vLLM (Pascal sm_61)",
    packages=["fusionattn"],
    license="Apache-2.0",
    cmdclass={"build_py": NativeBuildPy},
)
