#include <cstdlib>
#include <iostream>

// Phase 2 scaffolding entry point. Device discovery, pool connection, and
// mining loops are not implemented yet - see src/hardware/gpu_manager.hpp
// and src/network/stratum_client.hpp for the interfaces those stages will
// implement against.
int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    std::cout << "DeepCore " << DEEPCORE_VERSION_STRING
              << " - scaffolding build, no mining backend implemented yet.\n";

    return EXIT_SUCCESS;
}
