CUDA_HOME ?= /usr/local/cuda-12.8
NVCC = $(CUDA_HOME)/bin/nvcc
CUDA_ARCH = -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120
CFLAGS = -O3 -std=c++17 $(CUDA_ARCH) --expt-relaxed-constexpr --use_fast_math
LDFLAGS = -lcudnn -lcublas -lcudart -lavformat -lavcodec -lswscale -lavutil
INCLUDES = -I$(CUDA_HOME)/include -I/usr/local/include
LIB_DIRS = -L$(CUDA_HOME)/lib64 -L/usr/local/lib

TARGET = cuda_hand

all: $(TARGET)

$(TARGET): main.cu nn.cuh
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $@ main.cu $(LIB_DIRS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

export:
	python3 export_weights.py

.PHONY: all clean export
