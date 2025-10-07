# CUDA Compiler
CXX = nvcc

# Directories
INCLUDE_DIR = ./include
BUILD_DIR = build
SRC_DIR = src

# CUTLASS support
CUTLASS_PATH ?= $(HOME)/cutlass

# Compiler flags
CXXFLAGS = -O3 -std=c++17
INCLUDES = -I./include -I/home/oana/cutlass/include
NVCC_FLAGS = $(CXXFLAGS) $(INCLUDES) #--fmad=false

# Add CUTLASS include if directory exists
ifneq ($(wildcard $(CUTLASS_PATH)/include),)
    NVCC_FLAGS += -I$(CUTLASS_PATH)/include
    $(info Building with CUTLASS support from $(CUTLASS_PATH))
else
    $(warning CUTLASS not found at $(CUTLASS_PATH) - building without CUTLASS)
endif

# Libraries
LIBS = -lcublas -lcurand -lcusolver

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cu)

# Object files
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Target executable
TARGET = gemm_test

# Default target
all: $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Create data directory
data:
	mkdir -p data

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Link executable
$(TARGET): $(SOURCES) | data
	$(CXX) $(CXXFLAGS) $(PRECISION_FLAGS) $(INCLUDES) $^ $(LIBS) -o $@

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(TARGET) data/*.csv data/*.dat data/*.bin roofline_plot.png

# Force rebuild
rebuild: clean all

# Test targets
test: $(TARGET)
	./$(TARGET) --test=naive --size=512

test-all: $(TARGET)
	./$(TARGET)

# Debug build
debug: NVCC_FLAGS += -g -G
debug: $(TARGET)

# Profile targets
profile-naive: $(TARGET)
	ncu --set basic ./$(TARGET) --test=naive --size=1024

profile-tiled: $(TARGET)
	ncu --set basic ./$(TARGET) --test=tiled --size=1024

profile-cutlass: $(TARGET)
	ncu --set basic ./$(TARGET) --test=cutlass --size=1024

profile-cublas: $(TARGET)
	ncu --set basic ./$(TARGET) --test=cublas --size=1024

# Convenience targets for different precision combinations
baseline:
	$(MAKE) COMPUTE_TYPE=float ACCUMULATE_TYPE=float

fp16_mixed:
	$(MAKE) COMPUTE_TYPE=__half ACCUMULATE_TYPE=float

bf16_mixed:
	$(MAKE) COMPUTE_TYPE=__nv_bfloat16 ACCUMULATE_TYPE=float

fp64_acc:
	$(MAKE) COMPUTE_TYPE=float ACCUMULATE_TYPE=double

fp16_fp64:
	$(MAKE) COMPUTE_TYPE=__half ACCUMULATE_TYPE=double

bf16_fp64:
	$(MAKE) COMPUTE_TYPE=__nv_bfloat16 ACCUMULATE_TYPE=double

# Help
help:
	@echo "Available targets:"
	@echo "  all               - Build the project (default)"
	@echo "  clean             - Remove build files"
	@echo "  rebuild           - Clean and build"
	@echo "  test              - Run basic test"
	@echo "  test-all          - Run all tests"
	@echo "  debug             - Build with debug symbols"
	@echo "  profile-*         - Profile specific implementations"
	@echo "  baseline           - FP32 compute, FP32 accumulate (default)"
	@echo "  fp16_mixed       - FP16 compute, FP32 accumulate"
	@echo "  bf16_mixed       - BF16 compute, FP32 accumulate"
	@echo "  fp64_acc         - FP32 compute, FP64 accumulate"
	@echo "  fp16_fp64        - FP16 compute, FP64 accumulate"
	@echo "  bf16_fp64        - BF16 compute, FP64 accumulate"
	@echo ""
	@echo "Environment variables:"
	@echo "  CUTLASS_PATH - Path to CUTLASS installation (default: ~/cutlass)"
	@echo ""
	@echo "Custom usage:"
	@echo "  make COMPUTE_TYPE=__half ACCUMULATE_TYPE=double"

.PHONY: all clean rebuild test test-all debug help data error-eval-example