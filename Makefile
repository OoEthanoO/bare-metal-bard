NVCC      := nvcc

# Detected, not assumed, so this builds on a rented box without editing.
# sm_89 is the 4070 this was developed on; an A10 is sm_86, an A100 sm_80, an
# H100 sm_90. Building for the wrong one either fails to load or silently JITs.
ARCH ?= sm_$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
          | head -1 | tr -d '.' || echo 89)

# CUDA 12.4 rejects gcc >= 14 and Ubuntu 26.04 ships gcc 15, so a pinned host
# compiler is used WHEN ONE IS PRESENT. Most cloud images are older and have a
# perfectly acceptable default, and demanding g++-13 there would just fail.
CCBIN     := $(shell command -v g++-13 2>/dev/null)
CCBIN_FLAG := $(if $(CCBIN),-ccbin $(CCBIN),)

# TF32 tensor cores are Ampere and newer. Detected rather than assumed, so a
# clone builds on a T4 or a V100 -- it just builds without the tensor-core half
# and every GEMM takes the fp32 path.
ARCH_NUM  := $(patsubst sm_%,%,$(ARCH))
TF32      := $(shell [ "$(ARCH_NUM)" -ge 80 ] 2>/dev/null && echo 1 || echo 0)

NVCCFLAGS := $(CCBIN_FLAG) -arch=$(ARCH) -DBMB_TF32=$(TF32) -O3 -lineinfo -std=c++17 -Xcompiler -pthread \
             -Xcompiler -Wall -Xcompiler -Wno-unused-function
LDFLAGS   := -lcublas

SRC     := src/bench.cu src/registry.cu $(wildcard src/kernels/*.cu)
BIN     := bench/sgemm
TOOLS   := bench/device_query bench/test_gemm

.PHONY: all clean run tools test
all: $(BIN)

$(BIN): $(SRC) src/kernels.h src/kernels/lane_major.cuh | bench
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@ $(LDFLAGS)

tools: $(TOOLS)
bench/device_query: tools/device_query.cu | bench
	$(NVCC) -ccbin $(CCBIN) -arch=$(ARCH) -O2 $< -o $@

bench/test_gemm: tools/test_gemm.cu src/gemm.cu src/bgemm.cu src/gemm.h | bench
	$(NVCC) $(NVCCFLAGS) tools/test_gemm.cu src/gemm.cu src/bgemm.cu -o $@ $(LDFLAGS)

# The model. Note: no -lcublas. Nothing here links a vendor BLAS.
GPT_SRC := src/train_gpt.cu src/gpt.cu src/gemm.cu src/bgemm.cu src/nn.cu src/attention.cu src/flash.cu src/ddp.cu
GPT_DEP := src/gpt.h src/gemm.h src/nn.h src/attention.h src/flash.h src/ddp.h src/reduce.cuh src/gelu.cuh

bench/train_gpt: $(GPT_SRC) $(GPT_DEP) | bench
	$(NVCC) $(NVCCFLAGS) $(GPT_SRC) -o $@

bench/test_grad: tools/test_grad.cu src/gpt.cu src/gemm.cu src/bgemm.cu src/nn.cu src/attention.cu src/flash.cu src/ddp.cu $(GPT_DEP) | bench
	$(NVCC) $(NVCCFLAGS) tools/test_grad.cu src/gpt.cu src/gemm.cu src/bgemm.cu src/nn.cu src/attention.cu src/flash.cu src/ddp.cu -o $@

bench/test_ddp: tools/test_ddp.cu src/ddp.cu src/ddp.h | bench
	$(NVCC) $(NVCCFLAGS) tools/test_ddp.cu src/ddp.cu -o $@

bench/test_flash: tools/test_flash.cu src/flash.cu src/attention.cu src/bgemm.cu src/gemm.cu $(GPT_DEP) | bench
	$(NVCC) $(NVCCFLAGS) tools/test_flash.cu src/flash.cu src/attention.cu src/bgemm.cu src/gemm.cu -o $@

gpt: bench/train_gpt

test: bench/test_gemm
	./bench/test_gemm

bench:
	mkdir -p bench

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN) $(TOOLS) bench/train_gpt bench/test_grad bench/test_flash bench/test_ddp
