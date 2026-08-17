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

bench/test_gemm: tools/test_gemm.cu src/gemm.cu src/gemm.h src/kernels/k05_tile2d.cu | bench
	$(NVCC) $(NVCCFLAGS) tools/test_gemm.cu src/gemm.cu -o $@ $(LDFLAGS)

test: bench/test_gemm
	./bench/test_gemm

bench:
	mkdir -p bench

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN) $(TOOLS)
