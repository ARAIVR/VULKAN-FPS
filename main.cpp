#include "vulkan_fps.h"

int main() {
    VulkanRenderer renderer;

    try {
        renderer.initWindow();
        renderer.initVulkan();
        renderer.mainLoop();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    renderer.cleanup();
    return EXIT_SUCCESS;
}