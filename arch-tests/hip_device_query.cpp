#include <hip/hip_runtime.h>
#include <iostream>

void printHipDeviceProperties(int device) {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);

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
    std::cout << "  GCN Architecture: " << prop.gcnArchName << "\n";
    std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
    std::cout << "  L2 Cache Size: " << prop.l2CacheSize << "\n";
}

int main() {
    int count;
    hipGetDeviceCount(&count);
    for (int i = 0; i < count; ++i) {
        printHipDeviceProperties(i);
    }
    return 0;
}
