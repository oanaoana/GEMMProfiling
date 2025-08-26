# CUDA Compiler
NVCC = nvcc

# Directories
INCLUDE_DIR = ./include
BUILD_DIR = build
SRC_DIR = src

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
LIBS = -lcublas -lcusolver -lcurand

# Source files
SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/benchmark.cu $(SRC_DIR)/gemms.cu $(SRC_DIR)/utils.cu $(SRC_DIR)/error_analysis.cu $(SRC_DIR)/generate_test_matrix.cu $(SRC_DIR)/config.cu $(SRC_DIR)/matrix_utils.cu

# Object files
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Target executable
TARGET = main

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
$(TARGET): $(OBJECTS) | data
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) -o $@ $(LIBS)

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

# Clean up
clean:
	rm -f $(BUILD_DIR)/*.o $(BIN_DIR)/main test_matrices
	rm -f *.ncu-rep

# Help
help:
	@echo "Available targets:"
	@echo "  all               - Build the project (default)"
	@echo "  error-eval-example - Build error evaluation example"
	@echo "  clean             - Remove build files"
	@echo "  rebuild           - Clean and build"
	@echo "  test              - Run basic test"
	@echo "  test-all          - Run all tests"
	@echo "  debug             - Build with debug symbols"
	@echo "  profile-*         - Profile specific implementations"
	@echo ""
	@echo "Environment variables:"
	@echo "  CUTLASS_PATH - Path to CUTLASS installation (default: ~/cutlass)"

.PHONY: all clean rebuild test test-all debug help data error-eval-example

.PHONY: all clean rebuild test test-all debug help data