# Compiler settings
NVCC := nvcc
CFLAGS := -O3
INCLUDES := -I./include

# Directories
BUILD_DIR := build

# Files
SRCS := $(wildcard *.cu)
OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(SRCS))
MAIN := main

# Default target
all: setup $(MAIN)

# Create build directories if they don't exist
setup:
	mkdir -p $(BUILD_DIR)

# Compile main executable
$(MAIN): $(BUILD_DIR)/main.o $(BUILD_DIR)/gemms.o $(BUILD_DIR)/utils.o
	$(NVCC) $(CFLAGS) $^ -o $@

# Compile object files
$(BUILD_DIR)/%.o: %.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Debug build with additional info
debug: CFLAGS += -G
debug: clean all

# Profile build with line info for profiling tools
profile: CFLAGS += -lineinfo
profile: clean all

# Run the program
run: all
	./$(MAIN)

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(MAIN)

# Dependencies
$(BUILD_DIR)/main.o: main.cu
$(BUILD_DIR)/gemms.o: gemms.cu include/gemms.cuh
$(BUILD_DIR)/utils.o: utils.cu include/utils.cuh

.PHONY: all setup debug profile run clean test_naive test_tiled