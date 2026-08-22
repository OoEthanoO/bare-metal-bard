NVCC      := nvcc
# CUDA 12.4 rejects gcc >= 14; Ubuntu 26.04 ships gcc 15 as default.
CCBIN     := g++-13
ARCH      := sm_89
NVCCFLAGS := -ccbin $(CCBIN) -arch=$(ARCH) -O3 -lineinfo -std=c++17 \
             -Xcompiler -Wall -Xcompiler -Wno-unused-function
LDFLAGS   := -lcublas

SRC     := src/bench.cu src/registry.cu $(wildcard src/kernels/*.cu)
BIN     := bench/sgemm
TOOLS   := bench/device_query bench/test_gemm

.PHONY: all clean run tools test
all: $(BIN)

$(BIN): $(SRC) src/kernels.h | bench
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@ $(LDFLAGS)

tools: $(TOOLS)
bench/device_query: tools/device_query.cu | bench
	$(NVCC) -ccbin $(CCBIN) -arch=$(ARCH) -O2 $< -o $@

bench/test_gemm: tools/test_gemm.cu src/gemm.cu src/bgemm.cu src/gemm.h | bench
	$(NVCC) $(NVCCFLAGS) tools/test_gemm.cu src/gemm.cu src/bgemm.cu -o $@ $(LDFLAGS)

# The model. Note: no -lcublas. Nothing here links a vendor BLAS.
GPT_SRC := src/train_gpt.cu src/gpt.cu src/gemm.cu src/bgemm.cu src/nn.cu src/attention.cu src/flash.cu src/ddp.cu
GPT_DEP := src/gpt.h src/gemm.h src/nn.h src/attention.h src/flash.h src/ddp.h src/reduce.cuh

bench/train_gpt: $(GPT_SRC) $(GPT_DEP) | bench
	$(NVCC) $(NVCCFLAGS) $(GPT_SRC) -o $@

bench/test_grad: tools/test_grad.cu src/gpt.cu src/gemm.cu src/bgemm.cu src/nn.cu src/attention.cu src/flash.cu src/ddp.cu $(GPT_DEP) | bench
	$(NVCC) $(NVCCFLAGS) tools/test_grad.cu src/gpt.cu src/gemm.cu src/bgemm.cu src/nn.cu src/attention.cu src/flash.cu src/ddp.cu -o $@

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
	rm -f $(BIN) $(TOOLS) bench/train_gpt bench/test_grad bench/test_flash
