CUDA_PATH ?= /usr/local/cuda
NVCC      = $(CUDA_PATH)/bin/nvcc

# Try to auto-detect SM (e.g., 86, 75, 90...)
SMS ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '.')

NVCC_DBG  =
NVCCFLAGS = $(NVCC_DBG) -O3 --use_fast_math -lineinfo -m64

ifeq ($(strip $(SMS)),)
$(warning Could not detect GPU compute capability; falling back to PTX-only)
GENCODE_FLAGS = -gencode arch=compute_70,code=compute_70
else
GENCODE_FLAGS = -gencode arch=compute_$(SMS),code=sm_$(SMS) -gencode arch=compute_$(SMS),code=compute_$(SMS)
endif

cudart: cudart.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart cudart.o

cudart.o: main.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart.o -c main.cu

clean:
	rm -f cudart cudart.o
