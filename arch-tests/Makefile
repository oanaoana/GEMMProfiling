# GPU Architecture Testing Makefile - Simplified for Script Usage

NVCC = nvcc
CFLAGS = -O3 -std=c++17
ARCH = -arch=sm_89  # Adjust for your GPU

# Targets - only memory_test is an executable
TARGETS = memory_test

all: $(TARGETS)

# Build memory_test which includes mat_load_patterns.cu
memory_test: memory_test.cu mat_load_patterns.cu
	$(NVCC) $(CFLAGS) $(ARCH) -o $@ $<

clean:
	rm -f $(TARGETS)

help:
	@echo "GPU Architecture Testing Makefile"
	@echo "=================================="
	@echo ""
	@echo "Build targets:"
	@echo "  all                    Build all executables"
	@echo "  memory_test            Build main memory test runner"
	@echo "  clean                  Remove all executables"
	@echo "  help                   Show this help"
	@echo ""
	@echo "Usage:"
	@echo "  make                   Build all"
	@echo "  make clean             Clean all"
	@echo ""
	@echo "Test execution (use scripts or direct commands):"
	@echo "  ./memory_test --memory-test=512 --pattern=rowmajor --config=1d-tile"
	@echo "  ./memory_test --memory-test=512 --pattern=colmajor --config=2d-tile"
	@echo ""
	@echo "Create your own test scripts for batch testing!"

# Simple test to verify build works
test-build: memory_test
	@echo "Testing build..."
	./memory_test --memory-test=64 --pattern=rowmajor --config=1d-tile
	@echo "Build test passed!"

.PHONY: all clean help test-build