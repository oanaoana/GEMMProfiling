# CUDA Compiler and flags
CXX = nvcc
CXXFLAGS = -O3 -std=c++17
INCLUDES = -I./include -I/home/oana/cutlass/include

# CUTLASS support
CUTLASS_PATH ?= $(HOME)/cutlass
ifneq ($(wildcard $(CUTLASS_PATH)/include),)
    $(info Building with CUTLASS support from $(CUTLASS_PATH))
endif

# Libraries and files
LIBS = -lcublas -lcurand -lcusolver
SOURCES = $(wildcard src/*.cu)
HEADERS = $(wildcard include/*.h include/*.cuh)
TARGET = main

# Main build rule
$(TARGET): $(SOURCES) $(HEADERS)
	mkdir -p data
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) $(LIBS) -o $@

# Clean
clean:
	rm -f $(TARGET) data/*.csv data/*.dat data/*.bin

help:
	@echo "Usage: make [clean|help]"
	@echo "Configure precision types in include/config.h"
	@echo "Example: ./main --error-analysis --test=tiled_mixprec --size=1024"

.PHONY: clean help