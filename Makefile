# CUDA Compiler
NVCC = nvcc

# Directories
INCLUDE_DIR = ./include
BUILD_DIR = build

# CUTLASS support
CUTLASS_PATH ?= $(HOME)/cutlass

# Compiler flags
NVCC_FLAGS = -O3 -std=c++17 -I$(INCLUDE_DIR)

# Add CUTLASS include if directory exists
ifneq ($(wildcard $(CUTLASS_PATH)/include),)
    NVCC_FLAGS += -I$(CUTLASS_PATH)/include
    $(info Building with CUTLASS support from $(CUTLASS_PATH))
else
    $(warning CUTLASS not found at $(CUTLASS_PATH) - building without CUTLASS)
endif

# Libraries
LIBS = -lcublas

# Source files
SOURCES = main.cu benchmark.cu gemms.cu utils.cu numerical_analysis.cu

# Object files
OBJECTS = $(SOURCES:%.cu=$(BUILD_DIR)/%.o)

# Target executable
TARGET = main

# Default target
all: $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile object files
$(BUILD_DIR)/%.o: %.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Link executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) -o $@ $(LIBS)

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(TARGET) roofline_data.csv roofline_plot.png

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

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build the project (default)"
	@echo "  clean        - Remove build files"
	@echo "  rebuild      - Clean and build"
	@echo "  test         - Run basic test"
	@echo "  test-all     - Run all tests"
	@echo "  debug        - Build with debug symbols"
	@echo "  profile-*    - Profile specific implementations"
	@echo ""
	@echo "Environment variables:"
	@echo "  CUTLASS_PATH - Path to CUTLASS installation (default: ~/cutlass)"

.PHONY: all clean rebuild test test-all debug help