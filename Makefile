# Compiler settings
NVCC := nvcc
CFLAGS := -O3
INCLUDES := -I./include

# Directories
BUILD_DIR := build
BIN_DIR := bin

# Files
SRCS := $(wildcard *.cu)
OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(SRCS))
MAIN := main

# Default target
all: setup $(BIN_DIR)/$(MAIN)

# Create build directories if they don't exist
setup:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)

# Compile main executable
$(BIN_DIR)/$(MAIN): $(BUILD_DIR)/main.o $(filter-out $(BUILD_DIR)/main.o, $(OBJS))
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
	$(BIN_DIR)/$(MAIN)

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Additional targets for specific testing
test_naive: all
	$(BIN_DIR)/$(MAIN) --naive

test_tiled: all
	$(BIN_DIR)/$(MAIN) --tiled

# Dependencies
$(BUILD_DIR)/main.o: main.cu
$(BUILD_DIR)/gemms.o: gemms.cu include/gemms.cuh
$(BUILD_DIR)/utils.o: utils.cu include/utils.cuh

.PHONY: all setup debug profile run clean test_naive test_tiled