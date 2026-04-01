CUDA_HOME ?= /usr/local/cuda-12.8
NVCC = $(CUDA_HOME)/bin/nvcc
CUDA_ARCH = -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120
CFLAGS = -O3 -std=c++17 $(CUDA_ARCH) --expt-relaxed-constexpr --use_fast_math
LDFLAGS = -lcudnn -lcublas -lcusolver -lcudart -lcuda -lavformat -lavcodec -lswscale -lavutil
INCLUDES = -I$(CUDA_HOME)/include -I/usr/local/include
LIB_DIRS = -L$(CUDA_HOME)/lib64 -L/usr/local/lib

TARGET = cuda_droid

all: $(TARGET)

$(TARGET): main.cu nn.cuh se3.cuh ba.cuh
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $@ main.cu $(LIB_DIRS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

# Export weights from PyTorch checkpoint
export:
	python3 export_weights.py --checkpoint /workspace/DROID-SLAM/droid.pth --out data

# Run the benchmark
run: $(TARGET)
	./$(TARGET) --weights data/weights --frames data/frames --calib data/calib.bin

.PHONY: all clean export run
