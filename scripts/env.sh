#!/usr/bin/env bash
# The Linux/WSL toolchain, worked out once.  Source it, do not run it:
#
#     source scripts/env.sh && make -j4
#
# WHY THIS EXISTS. This machine's WSL side ships no CUDA toolkit, no gcc, and
# not even make -- and sudo wants a password, so apt is out. None of that is a
# reason to build through Windows: conda-forge packages the whole toolchain,
# installs into $HOME, and needs no root at all. CUDA 13.3 is pinned there
# because that is the version the Windows toolkit had, and which toolkit built
# a binary is part of the measurement rather than a detail.
#
# Setting it up from nothing (~7 GB, once):
#
#   curl -Ls -o mm.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest
#   python3 -c "import tarfile,shutil,os;t=tarfile.open('mm.tar.bz2');\
#     f=t.extractfile('bin/micromamba');d=os.path.expanduser('~/.local/bin/micromamba');\
#     os.makedirs(os.path.dirname(d),exist_ok=True);shutil.copyfileobj(f,open(d,'wb'));os.chmod(d,0o755)"
#   export MAMBA_ROOT_PREFIX=$HOME/micromamba
#   ~/.local/bin/micromamba create -y -n cuda -c conda-forge cuda-toolkit=13.3 gxx make
#
# (python3 does the extracting because this image has no bzip2 either.)
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

if [ ! -x "$HOME/.local/bin/micromamba" ]; then
  echo "[env] micromamba not found -- see the setup block in this file" >&2
  return 1 2>/dev/null || exit 1
fi

eval "$("$HOME/.local/bin/micromamba" shell hook -s bash)"
micromamba activate cuda || { echo "[env] no 'cuda' env -- see setup block" >&2; return 1; }

# The CUDA *driver* is not in the env: WSL exposes it at /usr/lib/wsl/lib, which
# WSL also writes into /etc/ld.so.conf.d, so libcuda resolves without help. This
# check is here because the failure mode otherwise is a link error that reads
# like a missing toolkit rather than a missing driver.
if ! ldconfig -p 2>/dev/null | grep -q libcuda; then
  echo "[env] warning: libcuda not on the loader path (is this a GPU-enabled WSL?)" >&2
fi

echo "[env] nvcc $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'), \
$(g++ --version | head -1 | sed 's/ (.*)//'), arch sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
