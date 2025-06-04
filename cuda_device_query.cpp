#include <cuda_runtime.h>
#include <iostream>

void printCudaDeviceProperties(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device " << device << ": " << prop.name << "\n";
    std::cout << "  Global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock << "\n";
    std::cout << "  Registers per block: " << prop.regsPerBlock << "\n";
    std::cout << "  Warp size: " << prop.warpSize << "\n";
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "  Max threads dim: [" << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]\n";
    std::cout << "  Max grid size: [" << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]\n";
    std::cout << "  Clock rate: " << prop.clockRate << " kHz\n";
    std::cout << "  Multiprocessor count: " << prop.multiProcessorCount << "\n";
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
    std::cout << "  L2 Cache Size: " << prop.l2CacheSize << "\n";
}

int main() {
    int count;
    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; ++i) {
        printCudaDeviceProperties(i);
    }
    return 0;
}
