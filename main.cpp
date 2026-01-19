
// ======================= main.cpp  PART 1/9 =======================
// Includes, globals, camera, math helpers, utility, one-shot commands, texture upload, morph structs

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <array>
#include <optional>
#include <cstdint>
#include <string>
#include <cmath>
#include <cassert>

#include "cgltf.h"
#include "tinygltf/stb_image.h"   // STB implementation is in stb.cpp

// ----------------------------------------------------------
// Globals
// ----------------------------------------------------------

GLFWwindow* gWindow = nullptr;

vk::Instance        gInstance;
vk::PhysicalDevice  gGPU;
vk::Device          gDevice;
vk::Queue           gGraphicsQueue;
uint32_t            gGraphicsQueueFamily = 0;

vk::SurfaceKHR      gSurface;
vk::SwapchainKHR    gSwapchain;
vk::Format          gSwapFormat;
vk::Extent2D        gSwapExtent;

std::vector<vk::Image>       gSwapImages;
std::vector<vk::ImageView>   gSwapViews;
std::vector<vk::Framebuffer> gFramebuffers;

vk::RenderPass      gRenderPass;
vk::PipelineLayout  gPipelineLayout;
vk::Pipeline        gPipeline;

vk::DescriptorSetLayout          gDescSetLayout;
vk::DescriptorPool               gDescPool;
std::vector<vk::DescriptorSet>   gGlobalDescSets;

// Depth buffer
vk::Image        gDepthImage;
vk::DeviceMemory gDepthMemory;
vk::ImageView    gDepthView;
vk::Format       gDepthFormat = vk::Format::eD32Sfloat;

// Textures
vk::Image        gPlaneTexImage;
vk::DeviceMemory gPlaneTexMemory;
vk::ImageView    gPlaneTexView;
vk::Sampler      gPlaneTexSampler;

vk::Image        gModelTexImage;
vk::DeviceMemory gModelTexMemory;
vk::ImageView    gModelTexView;
vk::Sampler      gModelTexSampler;

// Shared buffers
vk::Buffer       gVertexBuffer;
vk::DeviceMemory gVertexMemory;
vk::Buffer       gIndexBuffer;
vk::DeviceMemory gIndexMemory;

vk::CommandPool                 gCmdPool;
std::vector<vk::CommandBuffer>  gCmdBuffers;

vk::Semaphore       gImgAvailable;
vk::Semaphore       gRenderFinished;
vk::Fence           gInFlight;

// UBO per swapchain image
struct GlobalUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 lightDir;
    glm::vec4 lightColor;
};
std::vector<vk::Buffer>       gUBOs;
std::vector<vk::DeviceMemory> gUBOMem;

std::chrono::steady_clock::time_point gLastFrameTime;

// ----------------------------------------------------------
// Camera
// ----------------------------------------------------------

struct Camera {
    glm::vec3 position{ 0.0f, 8.0f, -25.0f };
    float pitch = -10.0f;
    float yaw = 90.0f;
    float speed = 5.0f;
    float sensitivity = 0.1f;

    glm::mat4 getView() const {
        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        glm::vec3 dir = glm::normalize(front);
        return glm::lookAt(position, position + dir, { 0.0f, 1.0f, 0.0f });
    }
};

Camera gCamera;
double gLastX = 400.0, gLastY = 300.0;
bool   gFirstMouse = true;

// ----------------------------------------------------------
// Transforms
// ----------------------------------------------------------

struct ObjectTransform {
    glm::vec3 position = glm::vec3(0.0f);
    glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale = glm::vec3(1.0f);

    glm::mat4 toMatrix() const {
        glm::mat4 T = glm::translate(glm::mat4(1.0f), position);
        glm::mat4 R = glm::mat4_cast(rotation);
        glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
        return T * R * S;
    }
};

// ----------------------------------------------------------
// Push constants
// ----------------------------------------------------------

struct PushConstants {
    glm::mat4 model;
    int   objectID;
    float roughnessOverride;
    float metallicOverride;
};

// ----------------------------------------------------------
// Vertex / Mesh
// ----------------------------------------------------------

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
};

std::vector<Vertex>   gVertices;
std::vector<uint32_t> gIndices;

// ----------------------------------------------------------
// Object registry
// ----------------------------------------------------------

struct RenderObject {
    uint32_t vertexOffset;
    uint32_t indexOffset;
    uint32_t indexCount;
    uint32_t materialID;
    ObjectTransform transform;
};

std::vector<RenderObject> gObjects;

// ----------------------------------------------------------
// Materials
// ----------------------------------------------------------

struct Material {
    vk::ImageView view;
    vk::Sampler   sampler;
};

std::vector<Material> gMaterials;

// ----------------------------------------------------------
// Loaded image (CPU-side RGBA)
// ----------------------------------------------------------

struct LoadedImage {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels; // RGBA8
};

// ----------------------------------------------------------
// Morphing structures (from glTF morph targets + animation)
// ----------------------------------------------------------

// Morph target: per-vertex delta from base mesh
struct MorphTarget {
    std::vector<glm::vec3> deltaPos;
    std::vector<glm::vec3> deltaNormal;
};
//start
// One morphable mesh instance
struct MorphMesh {
    uint32_t baseVertexOffset;   // into gVertices
    uint32_t vertexCount;        // number of vertices in this morphable mesh

    // CPU-side base copy for morphing
    std::vector<Vertex> baseVertices;

    // Morph targets (from glTF)
    std::vector<MorphTarget> targets;

    // Active weights (same size as targets)
    std::vector<float> weights;
};

// Global morph registry
std::vector<MorphMesh> gMorphMeshes;

// Dynamic vertex buffer for morphed vertices
vk::Buffer       gMorphVertexBuffer;
vk::DeviceMemory gMorphVertexMemory;
bool             gMorphBufferCreated = false;

// ----------------------------------------------------------
// glTF morph animation (weights over time)
// ----------------------------------------------------------

struct MorphAnimationSampler {
    std::vector<float> times;                 // keyframe times
    std::vector<std::vector<float>> weights;  // per keyframe: weights for all targets
};

struct MorphAnimation {
    MorphAnimationSampler sampler;
    float duration = 0.0f;
};

MorphAnimation gMorphAnimation;   // single clip for now

// ----------------------------------------------------------
// Utility
// ----------------------------------------------------------

std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Failed to open file: " + path);
    size_t size = static_cast<size_t>(f.tellg());
    std::vector<char> buf(size);
    f.seekg(0);
    f.read(buf.data(), size);
    return buf;
}

vk::ShaderModule loadShader(const std::string& path) {
    auto code = readFile(path);
    vk::ShaderModuleCreateInfo ci({}, code.size(),
        reinterpret_cast<const uint32_t*>(code.data()));
    return gDevice.createShaderModule(ci);
}

uint32_t findMemoryType(uint32_t bits, vk::MemoryPropertyFlags props) {
    auto mem = gGPU.getMemoryProperties();
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
        if ((bits & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("No suitable memory type");
}

void createBuffer(vk::DeviceSize size,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags props,
    vk::Buffer& buf,
    vk::DeviceMemory& mem) {
    vk::BufferCreateInfo bi({}, size, usage, vk::SharingMode::eExclusive);
    buf = gDevice.createBuffer(bi);
    auto req = gDevice.getBufferMemoryRequirements(buf);
    vk::MemoryAllocateInfo ai(req.size, findMemoryType(req.memoryTypeBits, props));
    mem = gDevice.allocateMemory(ai);
    gDevice.bindBufferMemory(buf, mem, 0);
}

// ----------------------------------------------------------
// One-shot command helpers
// ----------------------------------------------------------

vk::CommandBuffer beginSingleTimeCommands() {
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = gCmdPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = 1;

    vk::CommandBuffer cmd = gDevice.allocateCommandBuffers(allocInfo)[0];

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    cmd.begin(beginInfo);
    return cmd;
}

void endSingleTimeCommands(vk::CommandBuffer cmd) {
    cmd.end();
    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    gGraphicsQueue.submit(submitInfo, nullptr);
    gGraphicsQueue.waitIdle();
    gDevice.freeCommandBuffers(gCmdPool, cmd);
}

// ----------------------------------------------------------
// Texture upload: LoadedImage -> GPU image + sampler
// ----------------------------------------------------------

void createVulkanTextureFromRGBA(
    const LoadedImage& img,
    vk::Image& outImage,
    vk::DeviceMemory& outMemory,
    vk::ImageView& outView,
    vk::Sampler& outSampler
) {
    vk::DeviceSize imageSize = (vk::DeviceSize)(img.width * img.height * 4);

    // Staging buffer
    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingMemory;
    createBuffer(
        imageSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingMemory
    );

    void* data = gDevice.mapMemory(stagingMemory, 0, imageSize);
    std::memcpy(data, img.pixels.data(), (size_t)imageSize);
    gDevice.unmapMemory(stagingMemory);

    // GPU image
    vk::ImageCreateInfo ici{};
    ici.imageType = vk::ImageType::e2D;
    ici.format = vk::Format::eR8G8B8A8Unorm;
    ici.extent = vk::Extent3D(img.width, img.height, 1);
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = vk::SampleCountFlagBits::e1;
    ici.tiling = vk::ImageTiling::eOptimal;
    ici.usage =
        vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eSampled;
    ici.initialLayout = vk::ImageLayout::eUndefined;

    outImage = gDevice.createImage(ici);

    auto req = gDevice.getImageMemoryRequirements(outImage);
    vk::MemoryAllocateInfo ai{};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(
        req.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    outMemory = gDevice.allocateMemory(ai);
    gDevice.bindImageMemory(outImage, outMemory, 0);

    // Layout transitions + copy
    vk::CommandBuffer cmd = beginSingleTimeCommands();

    vk::ImageMemoryBarrier barrier{};
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = outImage;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    // undefined -> transfer dst
    barrier.oldLayout = vk::ImageLayout::eUndefined;
    barrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.srcAccessMask = {};
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer,
        {},
        nullptr, nullptr,
        barrier
    );

    vk::BufferImageCopy copy{};
    copy.bufferOffset = 0;
    copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    copy.imageSubresource.mipLevel = 0;
    copy.imageSubresource.baseArrayLayer = 0;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = vk::Extent3D(img.width, img.height, 1);

    cmd.copyBufferToImage(
        stagingBuffer,
        outImage,
        vk::ImageLayout::eTransferDstOptimal,
        1,
        &copy
    );

    // transfer dst -> shader read
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {},
        nullptr, nullptr,
        barrier
    );

    endSingleTimeCommands(cmd);

    gDevice.destroyBuffer(stagingBuffer);
    gDevice.freeMemory(stagingMemory);

    // Image view
    vk::ImageViewCreateInfo vi{};
    vi.image = outImage;
    vi.viewType = vk::ImageViewType::e2D;
    vi.format = vk::Format::eR8G8B8A8Unorm;
    vi.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    vi.subresourceRange.baseMipLevel = 0;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.baseArrayLayer = 0;
    vi.subresourceRange.layerCount = 1;

    outView = gDevice.createImageView(vi);

    // Sampler
    vk::SamplerCreateInfo si{};
    si.magFilter = vk::Filter::eLinear;
    si.minFilter = vk::Filter::eLinear;
    si.mipmapMode = vk::SamplerMipmapMode::eLinear;
    si.addressModeU = vk::SamplerAddressMode::eRepeat;
    si.addressModeV = vk::SamplerAddressMode::eRepeat;
    si.addressModeW = vk::SamplerAddressMode::eRepeat;
    si.anisotropyEnable = VK_FALSE;
    si.maxAnisotropy = 1.0f;
    si.minLod = 0.0f;
    si.maxLod = 0.0f;

    outSampler = gDevice.createSampler(si);
}
// ======================= main.cpp  PART 2/9 =======================
// Vulkan instance, surface, GPU selection, logical device

void initWindow() {
    if (!glfwInit())
        throw std::runtime_error("GLFW init failed");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    gWindow = glfwCreateWindow(1280, 720, "Vulkan Engine", nullptr, nullptr);

    if (!gWindow)
        throw std::runtime_error("Failed to create window");
}

void createInstance() {
    vk::ApplicationInfo appInfo(
        "Vulkan Engine", 1,
        "None", 1,
        VK_API_VERSION_1_3
    );

    uint32_t extCount = 0;
    const char** extensions = glfwGetRequiredInstanceExtensions(&extCount);

    vk::InstanceCreateInfo ci(
        {}, &appInfo,
        0, nullptr,
        extCount, extensions
    );

    gInstance = vk::createInstance(ci);
}

void createSurface() {
    VkSurfaceKHR rawSurface;
    if (glfwCreateWindowSurface(gInstance, gWindow, nullptr, &rawSurface) != VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface");

    gSurface = rawSurface;
}

void pickGPU() {
    auto devices = gInstance.enumeratePhysicalDevices();
    if (devices.empty())
        throw std::runtime_error("No Vulkan-capable GPU found");

    for (auto& d : devices) {
        auto props = d.getProperties();
        if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            gGPU = d;
            std::cout << "Using discrete GPU: " << props.deviceName << "\n";
            return;
        }
    }

    gGPU = devices[0];
    std::cout << "Using fallback GPU: " << gGPU.getProperties().deviceName << "\n";
}

void createDevice() {
    auto qFamilies = gGPU.getQueueFamilyProperties();

    bool found = false;
    for (uint32_t i = 0; i < qFamilies.size(); ++i) {
        if (qFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            gGraphicsQueueFamily = i;
            found = true;
            break;
        }
    }

    if (!found)
        throw std::runtime_error("No graphics queue family found");

    float priority = 1.0f;
    vk::DeviceQueueCreateInfo qci({}, gGraphicsQueueFamily, 1, &priority);

    const char* deviceExtensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    vk::DeviceCreateInfo dci(
        {}, 1, &qci,
        0, nullptr,
        1, deviceExtensions
    );

    gDevice = gGPU.createDevice(dci);
    gGraphicsQueue = gDevice.getQueue(gGraphicsQueueFamily, 0);
}
// ======================= main.cpp  PART 3/9 =======================
// Swapchain, depth buffer, render pass, framebuffers

void createSwapchain() {
    auto caps = gGPU.getSurfaceCapabilitiesKHR(gSurface);
    auto fmts = gGPU.getSurfaceFormatsKHR(gSurface);
    auto modes = gGPU.getSurfacePresentModesKHR(gSurface);

    if (fmts.empty())
        throw std::runtime_error("No surface formats available");

    gSwapFormat = fmts[0].format;
    gSwapExtent = caps.currentExtent;

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
        imageCount = caps.maxImageCount;

    vk::SwapchainCreateInfoKHR ci{};
    ci.surface = gSurface;
    ci.minImageCount = imageCount;
    ci.imageFormat = gSwapFormat;
    ci.imageColorSpace = fmts[0].colorSpace;
    ci.imageExtent = gSwapExtent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
    ci.imageSharingMode = vk::SharingMode::eExclusive;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    ci.presentMode = vk::PresentModeKHR::eFifo;
    ci.clipped = VK_TRUE;

    gSwapchain = gDevice.createSwapchainKHR(ci);
    gSwapImages = gDevice.getSwapchainImagesKHR(gSwapchain);

    gSwapViews.resize(gSwapImages.size());
    for (size_t i = 0; i < gSwapImages.size(); ++i) {
        vk::ImageViewCreateInfo vi{};
        vi.image = gSwapImages[i];
        vi.viewType = vk::ImageViewType::e2D;
        vi.format = gSwapFormat;
        vi.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        gSwapViews[i] = gDevice.createImageView(vi);
    }
}

void createDepthResources() {
    vk::ImageCreateInfo ici{};
    ici.imageType = vk::ImageType::e2D;
    ici.format = gDepthFormat;
    ici.extent = vk::Extent3D(gSwapExtent.width, gSwapExtent.height, 1);
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = vk::SampleCountFlagBits::e1;
    ici.tiling = vk::ImageTiling::eOptimal;
    ici.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
    ici.initialLayout = vk::ImageLayout::eUndefined;

    gDepthImage = gDevice.createImage(ici);

    auto req = gDevice.getImageMemoryRequirements(gDepthImage);
    vk::MemoryAllocateInfo ai{};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(
        req.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    gDepthMemory = gDevice.allocateMemory(ai);
    gDevice.bindImageMemory(gDepthImage, gDepthMemory, 0);

    vk::ImageViewCreateInfo vi{};
    vi.image = gDepthImage;
    vi.viewType = vk::ImageViewType::e2D;
    vi.format = gDepthFormat;
    vi.subresourceRange = { vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1 };

    gDepthView = gDevice.createImageView(vi);
}

void createRenderPass() {
    vk::AttachmentDescription color{};
    color.format = gSwapFormat;
    color.samples = vk::SampleCountFlagBits::e1;
    color.loadOp = vk::AttachmentLoadOp::eClear;
    color.storeOp = vk::AttachmentStoreOp::eStore;
    color.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    color.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    color.initialLayout = vk::ImageLayout::eUndefined;
    color.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentDescription depth{};
    depth.format = gDepthFormat;
    depth.samples = vk::SampleCountFlagBits::e1;
    depth.loadOp = vk::AttachmentLoadOp::eClear;
    depth.storeOp = vk::AttachmentStoreOp::eDontCare;
    depth.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    depth.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    depth.initialLayout = vk::ImageLayout::eUndefined;
    depth.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    vk::AttachmentReference colorRef{ 0, vk::ImageLayout::eColorAttachmentOptimal };
    vk::AttachmentReference depthRef{ 1, vk::ImageLayout::eDepthStencilAttachmentOptimal };

    vk::SubpassDescription sub{};
    sub.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &colorRef;
    sub.pDepthStencilAttachment = &depthRef;

    std::array<vk::AttachmentDescription, 2> attachments{ color, depth };

    vk::RenderPassCreateInfo rp{};
    rp.attachmentCount = (uint32_t)attachments.size();
    rp.pAttachments = attachments.data();
    rp.subpassCount = 1;
    rp.pSubpasses = &sub;

    gRenderPass = gDevice.createRenderPass(rp);
}

void createFramebuffers() {
    gFramebuffers.resize(gSwapViews.size());

    for (size_t i = 0; i < gSwapViews.size(); ++i) {
        std::array<vk::ImageView, 2> attachments = {
            gSwapViews[i],
            gDepthView
        };

        vk::FramebufferCreateInfo fb{};
        fb.renderPass = gRenderPass;
        fb.attachmentCount = (uint32_t)attachments.size();
        fb.pAttachments = attachments.data();
        fb.width = gSwapExtent.width;
        fb.height = gSwapExtent.height;
        fb.layers = 1;

        gFramebuffers[i] = gDevice.createFramebuffer(fb);
    }
}
// ======================= main.cpp  PART 4/9 =======================
// Descriptor layouts, pipeline layout, graphics pipeline

void createDescriptorSetLayout() {
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings{};

    bindings[0].binding = 0;
    bindings[0].descriptorType = vk::DescriptorType::eUniformBuffer;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags =
        vk::ShaderStageFlagBits::eVertex |
        vk::ShaderStageFlagBits::eFragment;

    bindings[1].binding = 1;
    bindings[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutCreateInfo ci{};
    ci.bindingCount = (uint32_t)bindings.size();
    ci.pBindings = bindings.data();

    gDescSetLayout = gDevice.createDescriptorSetLayout(ci);
}

void createPipelineLayout() {
    vk::PushConstantRange pcRange{};
    pcRange.stageFlags =
        vk::ShaderStageFlagBits::eVertex |
        vk::ShaderStageFlagBits::eFragment;
    pcRange.offset = 0;
    pcRange.size = sizeof(PushConstants);

    vk::PipelineLayoutCreateInfo pl{};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &gDescSetLayout;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pcRange;

    gPipelineLayout = gDevice.createPipelineLayout(pl);
}

void createPipeline() {
    vk::ShaderModule vert = loadShader("vertex_pbr.spv");
    vk::ShaderModule frag = loadShader("fragment_pbr.spv");

    vk::PipelineShaderStageCreateInfo stages[2]{};
    stages[0].stage = vk::ShaderStageFlagBits::eVertex;
    stages[0].module = vert;
    stages[0].pName = "main";

    stages[1].stage = vk::ShaderStageFlagBits::eFragment;
    stages[1].module = frag;
    stages[1].pName = "main";

    vk::VertexInputBindingDescription bind{};
    bind.binding = 0;
    bind.stride = sizeof(Vertex);
    bind.inputRate = vk::VertexInputRate::eVertex;

    std::array<vk::VertexInputAttributeDescription, 3> attrs{
        vk::VertexInputAttributeDescription{
            0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)
        },
        vk::VertexInputAttributeDescription{
            1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)
        },
        vk::VertexInputAttributeDescription{
            2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, uv)
        }
    };

    vk::PipelineVertexInputStateCreateInfo vi{};
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &bind;
    vi.vertexAttributeDescriptionCount = (uint32_t)attrs.size();
    vi.pVertexAttributeDescriptions = attrs.data();

    vk::PipelineInputAssemblyStateCreateInfo ia{};
    ia.topology = vk::PrimitiveTopology::eTriangleList;

    vk::Viewport vp{};
    vp.x = 0.f;
    vp.y = 0.f;
    vp.width = (float)gSwapExtent.width;
    vp.height = (float)gSwapExtent.height;
    vp.minDepth = 0.f;
    vp.maxDepth = 1.f;

    vk::Rect2D sc{};
    sc.offset = vk::Offset2D(0, 0);
    sc.extent = gSwapExtent;

    vk::PipelineViewportStateCreateInfo vps{};
    vps.viewportCount = 1;
    vps.pViewports = &vp;
    vps.scissorCount = 1;
    vps.pScissors = &sc;

    vk::PipelineRasterizationStateCreateInfo rs{};
    rs.polygonMode = vk::PolygonMode::eFill;
    rs.cullMode = vk::CullModeFlagBits::eBack;
    rs.frontFace = vk::FrontFace::eCounterClockwise;
    rs.lineWidth = 1.0f;

    vk::PipelineMultisampleStateCreateInfo ms{};
    ms.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::PipelineDepthStencilStateCreateInfo ds{};
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = vk::CompareOp::eLess;
    ds.depthBoundsTestEnable = VK_FALSE;
    ds.stencilTestEnable = VK_FALSE;

    vk::PipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask =
        vk::ColorComponentFlagBits::eR |
        vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB |
        vk::ColorComponentFlagBits::eA;

    vk::PipelineColorBlendStateCreateInfo cb{};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    vk::GraphicsPipelineCreateInfo gp{};
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vps;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pDepthStencilState = &ds;
    gp.pColorBlendState = &cb;
    gp.layout = gPipelineLayout;
    gp.renderPass = gRenderPass;
    gp.subpass = 0;

    gPipeline = gDevice.createGraphicsPipeline({}, gp).value;

    gDevice.destroyShaderModule(vert);
    gDevice.destroyShaderModule(frag);
}
// ======================= main.cpp  PART 5/9 =======================
// Command pool, geometry (plane, cube, glTF), morph setup from glTF, shared buffers

void createCommandPool() {
    vk::CommandPoolCreateInfo ci{};
    ci.queueFamilyIndex = gGraphicsQueueFamily;
    ci.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    gCmdPool = gDevice.createCommandPool(ci);
}

void addPlaneAndCubeGeometry() {
    gVertices.clear();
    gIndices.clear();
    gObjects.clear();

    float groundHalf = 50.0f;
    float groundUV = 10.0f;

    uint32_t baseV = (uint32_t)gVertices.size();
    uint32_t baseI = (uint32_t)gIndices.size();

    gVertices.push_back({ { -groundHalf, 0, -groundHalf }, {0,1,0}, {0,0} });
    gVertices.push_back({ {  groundHalf, 0, -groundHalf }, {0,1,0}, {groundUV,0} });
    gVertices.push_back({ {  groundHalf, 0,  groundHalf }, {0,1,0}, {groundUV,groundUV} });
    gVertices.push_back({ { -groundHalf, 0,  groundHalf }, {0,1,0}, {0,groundUV} });

    gIndices.insert(gIndices.end(), {
        baseV + 0, baseV + 3, baseV + 2,
        baseV + 2, baseV + 1, baseV + 0
        });

    RenderObject ground{};
    ground.vertexOffset = baseV;
    ground.indexOffset = baseI;
    ground.indexCount = 6;
    ground.materialID = 0;
    ground.transform.position = { 0,0,0 };
    ground.transform.scale = { 1,1,1 };
    gObjects.push_back(ground);

    auto addFace = [&](glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3 n) {
        uint32_t start = (uint32_t)gVertices.size();
        gVertices.push_back({ v0, n, {0,0} });
        gVertices.push_back({ v1, n, {1,0} });
        gVertices.push_back({ v2, n, {1,1} });
        gVertices.push_back({ v3, n, {0,1} });

        gIndices.insert(gIndices.end(), {
            start + 0, start + 1, start + 2,
            start + 2, start + 3, start + 0
            });
        };

    float h = 5.0f;
    float y0 = 0.0f;
    float y1 = 10.0f;

    addFace({ -h,y1,-h }, { h,y1,-h }, { h,y1,h }, { -h,y1,h }, { 0,1,0 });
    addFace({ -h,y0,h }, { h,y0,h }, { h,y0,-h }, { -h,y0,-h }, { 0,-1,0 });
    addFace({ h,y0,-h }, { h,y0,h }, { h,y1,h }, { h,y1,-h }, { 1,0,0 });
    addFace({ -h,y0,h }, { -h,y0,-h }, { -h,y1,-h }, { -h,y1,h }, { -1,0,0 });
    addFace({ -h,y0,h }, { h,y0,h }, { h,y1,h }, { -h,y1,h }, { 0,0,1 });
    addFace({ h,y0,-h }, { -h,y0,-h }, { -h,y1,-h }, { h,y1,-h }, { 0,0,-1 });

    RenderObject cube{};
    cube.vertexOffset = baseV + 4;
    cube.indexOffset = baseI + 6;
    cube.indexCount = (uint32_t)gIndices.size() - cube.indexOffset;
    cube.materialID = 0;
    cube.transform.position = { 0,0,0 };
    cube.transform.scale = { 1,1,1 };
    gObjects.push_back(cube);
}

// Temporary storage for model vertices/indices
std::vector<Vertex>   gModelVertices;
std::vector<uint32_t> gModelIndices;
//start cut 
// Load glTF model, including morph targets and morph animation
void loadGLTFModel(const char* path) {
    cgltf_options options{};
    cgltf_data* data = nullptr;

    if (cgltf_parse_file(&options, path, &data) != cgltf_result_success)
        throw std::runtime_error("Failed to parse glTF");

    if (cgltf_load_buffers(&options, data, path) != cgltf_result_success) {
        cgltf_free(data);
        throw std::runtime_error("Failed to load glTF buffers");
    }

    if (data->meshes_count == 0) {
        cgltf_free(data);
        throw std::runtime_error("glTF has no meshes");
    }

    cgltf_mesh* mesh = &data->meshes[0];
    cgltf_primitive* prim = &mesh->primitives[0];

    const cgltf_accessor* posAcc = nullptr;
    const cgltf_accessor* normAcc = nullptr;
    const cgltf_accessor* uvAcc = nullptr;

    for (size_t i = 0; i < prim->attributes_count; ++i) {
        cgltf_attribute* a = &prim->attributes[i];
        if (a->type == cgltf_attribute_type_position) posAcc = a->data;
        if (a->type == cgltf_attribute_type_normal)   normAcc = a->data;
        if (a->type == cgltf_attribute_type_texcoord && a->index == 0) uvAcc = a->data;
    }

    if (!posAcc) {
        cgltf_free(data);
        throw std::runtime_error("glTF missing POSITION");
    }

    size_t vcount = posAcc->count;
    gModelVertices.resize(vcount);

    float tmp[4];
    for (size_t i = 0; i < vcount; ++i) {
        cgltf_accessor_read_float(posAcc, i, tmp, 3);
        gModelVertices[i].pos = { tmp[0],tmp[1],tmp[2] };

        if (normAcc) {
            cgltf_accessor_read_float(normAcc, i, tmp, 3);
            gModelVertices[i].normal = { tmp[0],tmp[1],tmp[2] };
        }
        else {
            gModelVertices[i].normal = { 0,1,0 };
        }

        if (uvAcc) {
            cgltf_accessor_read_float(uvAcc, i, tmp, 2);
            gModelVertices[i].uv = { tmp[0],tmp[1] };
        }
        else {
            gModelVertices[i].uv = { 0,0 };
        }
    }

    gModelIndices.clear();
    if (prim->indices) {
        cgltf_accessor* idx = prim->indices;
        gModelIndices.resize(idx->count);
        for (size_t i = 0; i < idx->count; ++i)
            gModelIndices[i] = (uint32_t)cgltf_accessor_read_index(idx, i);
    }
    else {
        gModelIndices.resize(vcount);
        for (size_t i = 0; i < vcount; ++i)
            gModelIndices[i] = (uint32_t)i;
    }

    // Load base color texture
    LoadedImage modelImg;

    if (prim->material &&
        prim->material->pbr_metallic_roughness.base_color_texture.texture) {

        cgltf_texture* tex = prim->material->pbr_metallic_roughness.base_color_texture.texture;
        cgltf_image* image = tex->image;

        int w = 0, h = 0, c = 0;
        stbi_uc* pixels = nullptr;

        if (image->uri && image->uri[0] != '\0') {
            pixels = stbi_load(image->uri, &w, &h, &c, STBI_rgb_alpha);
        }
        else if (image->buffer_view) {
            cgltf_buffer_view* bv = image->buffer_view;
            const unsigned char* start =
                (const unsigned char*)bv->buffer->data + bv->offset;
            pixels = stbi_load_from_memory(start, (int)bv->size, &w, &h, &c, STBI_rgb_alpha);
        }

        if (!pixels) {
            cgltf_free(data);
            throw std::runtime_error("Failed to load glTF texture");
        }

        modelImg.width = w;
        modelImg.height = h;
        modelImg.pixels.assign(pixels, pixels + (w * h * 4));
        stbi_image_free(pixels);
    }
    else {
        modelImg.width = 1;
        modelImg.height = 1;
        modelImg.pixels = { 255,255,255,255 };
    }

    createVulkanTextureFromRGBA(
        modelImg,
        gModelTexImage,
        gModelTexMemory,
        gModelTexView,
        gModelTexSampler
    );

    // ------------------------------------------------------
    // Build MorphMesh from glTF morph targets
    // ------------------------------------------------------
    gMorphMeshes.clear();
    gMorphAnimation = MorphAnimation{};

    if (prim->targets_count > 0) {
        MorphMesh mm{};
        mm.baseVertexOffset = 0; // will be adjusted after appendModelToGeometry
        mm.vertexCount = (uint32_t)vcount;

        mm.baseVertices = gModelVertices; // base mesh

        mm.targets.resize(prim->targets_count);

        for (size_t t = 0; t < prim->targets_count; ++t) {
            MorphTarget& mt = mm.targets[t];

            const cgltf_accessor* posDeltaAcc = nullptr;
            const cgltf_accessor* normDeltaAcc = nullptr;

            for (size_t a = 0; a < prim->targets[t].attributes_count; ++a) {
                cgltf_attribute* attr = &prim->targets[t].attributes[a];
                if (attr->type == cgltf_attribute_type_position)
                    posDeltaAcc = attr->data;
                if (attr->type == cgltf_attribute_type_normal)
                    normDeltaAcc = attr->data;
            }

            mt.deltaPos.resize(vcount, glm::vec3(0.0f));
            mt.deltaNormal.resize(vcount, glm::vec3(0.0f));

            if (posDeltaAcc) {
                for (size_t i = 0; i < vcount; ++i) {
                    cgltf_accessor_read_float(posDeltaAcc, i, tmp, 3);
                    mt.deltaPos[i] = glm::vec3(tmp[0], tmp[1], tmp[2]);
                }
            }

            if (normDeltaAcc) {
                for (size_t i = 0; i < vcount; ++i) {
                    cgltf_accessor_read_float(normDeltaAcc, i, tmp, 3);
                    mt.deltaNormal[i] = glm::vec3(tmp[0], tmp[1], tmp[2]);
                }
            }
        }

        mm.weights.resize(prim->targets_count, 0.0f);
        gMorphMeshes.push_back(std::move(mm));
    }

    // ------------------------------------------------------
    // Load morph weight animation (first animation, weights path)
    // ------------------------------------------------------
    if (data->animations_count > 0 && !gMorphMeshes.empty()) {
        cgltf_animation* anim = &data->animations[0];

        for (size_t c = 0; c < anim->channels_count; ++c) {
            cgltf_animation_channel* ch = &anim->channels[c];
            if (ch->target_path != cgltf_animation_path_type_weights)
                continue;

            cgltf_animation_sampler* s = ch->sampler;

            cgltf_accessor* inAcc = s->input;
            cgltf_accessor* outAcc = s->output;

            size_t keyCount = inAcc->count;
            size_t weightCountPerKey = outAcc->count / keyCount;

            gMorphAnimation.sampler.times.resize(keyCount);
            gMorphAnimation.sampler.weights.resize(keyCount);

            for (size_t k = 0; k < keyCount; ++k) {
                float tIn[4];
                cgltf_accessor_read_float(inAcc, k, tIn, 1);
                gMorphAnimation.sampler.times[k] = tIn[0];

                gMorphAnimation.sampler.weights[k].resize(weightCountPerKey);
                for (size_t w = 0; w < weightCountPerKey; ++w) {
                    float wOut[4];
                    cgltf_accessor_read_float(outAcc, k * weightCountPerKey + w, wOut, 1);
                    gMorphAnimation.sampler.weights[k][w] = wOut[0];
                }
            }

            if (!gMorphAnimation.sampler.times.empty())
                gMorphAnimation.duration = gMorphAnimation.sampler.times.back();

            break; // use first weights channel
        }
    }

    cgltf_free(data);
}

void appendModelToGeometry() {
    uint32_t baseV = (uint32_t)gVertices.size();
    uint32_t baseI = (uint32_t)gIndices.size();

    gVertices.insert(gVertices.end(), gModelVertices.begin(), gModelVertices.end());
    for (auto idx : gModelIndices)
        gIndices.push_back(baseV + idx);

    RenderObject model{};
    model.vertexOffset = baseV;
    model.indexOffset = baseI;
    model.indexCount = (uint32_t)gModelIndices.size();
    model.materialID = 1; // model uses material 1
    model.transform.position = { 0,5,0 };
    model.transform.scale = { 0.01f,0.01f,0.01f };

    gObjects.push_back(model);

    // Fix MorphMesh baseVertexOffset now that model is appended
    if (!gMorphMeshes.empty()) {
        MorphMesh& mm = gMorphMeshes[0];
        mm.baseVertexOffset = baseV;
        mm.vertexCount = (uint32_t)gModelVertices.size();
        mm.baseVertices = gModelVertices; // ensure base matches model
    }
}

void createSharedBuffers() {
    vk::DeviceSize vSize = sizeof(Vertex) * gVertices.size();
    vk::DeviceSize iSize = sizeof(uint32_t) * gIndices.size();

    createBuffer(
        vSize,
        vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent,
        gVertexBuffer, gVertexMemory
    );

    createBuffer(
        iSize,
        vk::BufferUsageFlagBits::eIndexBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent,
        gIndexBuffer, gIndexMemory
    );

    void* data = gDevice.mapMemory(gVertexMemory, 0, vSize);
    memcpy(data, gVertices.data(), (size_t)vSize);
    gDevice.unmapMemory(gVertexMemory);

    data = gDevice.mapMemory(gIndexMemory, 0, iSize);
    memcpy(data, gIndices.data(), (size_t)iSize);
    gDevice.unmapMemory(gIndexMemory);
}
// ======================= main.cpp  PART 6/9 =======================
// Morph buffer, morph animation evaluation, UBOs, descriptor sets, command buffers, sync objects, UBO update

void createMorphVertexBufferIfNeeded() {
    if (gMorphBufferCreated)
        return;

    vk::DeviceSize vSize = sizeof(Vertex) * gVertices.size();

    createBuffer(
        vSize,
        vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent,
        gMorphVertexBuffer,
        gMorphVertexMemory
    );

    // Initialize morph buffer with base vertices
    void* data = gDevice.mapMemory(gMorphVertexMemory, 0, vSize);
    memcpy(data, gVertices.data(), (size_t)vSize);
    gDevice.unmapMemory(gMorphVertexMemory);

    gMorphBufferCreated = true;
}

// Evaluate morph animation sampler at time t (looped)
void evaluateMorphAnimation(float timeSeconds, std::vector<float>& outWeights) {
    if (gMorphAnimation.sampler.times.empty())
        return;

    float duration = gMorphAnimation.duration;
    if (duration <= 0.0f)
        duration = gMorphAnimation.sampler.times.back();

    float t = fmodf(timeSeconds, duration);

    const auto& times = gMorphAnimation.sampler.times;
    const auto& weights = gMorphAnimation.sampler.weights;

    if (t <= times.front()) {
        outWeights = weights.front();
        return;
    }
    if (t >= times.back()) {
        outWeights = weights.back();
        return;
    }

    size_t k1 = 0;
    for (size_t i = 0; i < times.size() - 1; ++i) {
        if (t >= times[i] && t <= times[i + 1]) {
            k1 = i;
            break;
        }
    }

    size_t k2 = k1 + 1;
    float t0 = times[k1];
    float t1 = times[k2];
    float alpha = (t - t0) / (t1 - t0);

    const auto& w0 = weights[k1];
    const auto& w1 = weights[k2];

    size_t count = std::min(w0.size(), w1.size());
    outWeights.resize(count);

    for (size_t i = 0; i < count; ++i) {
        outWeights[i] = (1.0f - alpha) * w0[i] + alpha * w1[i];
    }
}

void updateMorphMeshes(float timeSeconds) {
    if (gMorphMeshes.empty())
        return;

    createMorphVertexBufferIfNeeded();

    std::vector<Vertex> morphed = gVertices;

    for (auto& mm : gMorphMeshes) {
        if (mm.targets.empty())
            continue;

        // Evaluate animation weights for this mesh
        std::vector<float> animWeights;
        evaluateMorphAnimation(timeSeconds, animWeights);

        // Clamp to number of targets
        size_t targetCount = mm.targets.size();
        mm.weights.assign(targetCount, 0.0f);
        for (size_t i = 0; i < std::min(targetCount, animWeights.size()); ++i) {
            mm.weights[i] = animWeights[i];
        }

        // Safety checks
        assert(mm.baseVertexOffset + mm.vertexCount <= morphed.size());
        assert(mm.baseVertices.size() == mm.vertexCount);

        for (uint32_t i = 0; i < mm.vertexCount; ++i) {
            Vertex v = mm.baseVertices[i];

            glm::vec3 pos = v.pos;
            glm::vec3 nrm = v.normal;

            for (size_t t = 0; t < mm.targets.size(); ++t) {
                float w = mm.weights[t];
                if (w == 0.0f) continue;

                const MorphTarget& mt = mm.targets[t];
                assert(i < mt.deltaPos.size());
                assert(i < mt.deltaNormal.size());

                pos += mt.deltaPos[i] * w;
                nrm += mt.deltaNormal[i] * w;
            }

            if (glm::length(nrm) > 0.0001f)
                nrm = glm::normalize(nrm);

            v.pos = pos;
            v.normal = nrm;

            morphed[mm.baseVertexOffset + i] = v;
        }
    }

    vk::DeviceSize vSize = sizeof(Vertex) * morphed.size();
    void* data = gDevice.mapMemory(gMorphVertexMemory, 0, vSize);
    memcpy(data, morphed.data(), (size_t)vSize);
    gDevice.unmapMemory(gMorphVertexMemory);
}

void createUBOs() {
    size_t count = gSwapImages.size();
    gUBOs.resize(count);
    gUBOMem.resize(count);

    for (size_t i = 0; i < count; ++i) {
        vk::DeviceSize size = sizeof(GlobalUBO);

        createBuffer(
            size,
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
            gUBOs[i],
            gUBOMem[i]
        );
    }
}

void createDescriptorSets() {
    size_t count = gSwapImages.size();

    std::array<vk::DescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
    poolSizes[0].descriptorCount = (uint32_t)count;
    poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
    poolSizes[1].descriptorCount = (uint32_t)count;

    vk::DescriptorPoolCreateInfo pi{};
    pi.maxSets = (uint32_t)count;
    pi.poolSizeCount = (uint32_t)poolSizes.size();
    pi.pPoolSizes = poolSizes.data();

    gDescPool = gDevice.createDescriptorPool(pi);

    std::vector<vk::DescriptorSetLayout> layouts(count, gDescSetLayout);
    vk::DescriptorSetAllocateInfo ai{};
    ai.descriptorPool = gDescPool;
    ai.descriptorSetCount = (uint32_t)count;
    ai.pSetLayouts = layouts.data();

    gGlobalDescSets = gDevice.allocateDescriptorSets(ai);

    for (size_t i = 0; i < count; ++i) {
        vk::DescriptorBufferInfo bi{};
        bi.buffer = gUBOs[i];
        bi.offset = 0;
        bi.range = sizeof(GlobalUBO);

        vk::WriteDescriptorSet write{};
        write.dstSet = gGlobalDescSets[i];
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = vk::DescriptorType::eUniformBuffer;
        write.pBufferInfo = &bi;

        gDevice.updateDescriptorSets(1, &write, 0, nullptr);
    }
}

void createCommandBuffers() {
    gCmdBuffers.resize(gFramebuffers.size());

    vk::CommandBufferAllocateInfo ai{};
    ai.commandPool = gCmdPool;
    ai.level = vk::CommandBufferLevel::ePrimary;
    ai.commandBufferCount = (uint32_t)gCmdBuffers.size();

    gCmdBuffers = gDevice.allocateCommandBuffers(ai);

    for (size_t i = 0; i < gCmdBuffers.size(); ++i) {
        vk::CommandBuffer cmd = gCmdBuffers[i];

        vk::CommandBufferBeginInfo bi{};
        cmd.begin(bi);

        std::array<vk::ClearValue, 2> clears{};
        clears[0].color = vk::ClearColorValue(std::array<float, 4>{0.1f, 0.1f, 0.15f, 1.0f});
        clears[1].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);

        vk::RenderPassBeginInfo rp{};
        rp.renderPass = gRenderPass;
        rp.framebuffer = gFramebuffers[i];
        rp.renderArea.offset = vk::Offset2D(0, 0);
        rp.renderArea.extent = gSwapExtent;
        rp.clearValueCount = (uint32_t)clears.size();
        rp.pClearValues = clears.data();

        cmd.beginRenderPass(rp, vk::SubpassContents::eInline);
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, gPipeline);

        cmd.bindIndexBuffer(gIndexBuffer, 0, vk::IndexType::eUint32);

        cmd.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            gPipelineLayout,
            0,
            gGlobalDescSets[i],
            nullptr
        );

        for (size_t objIndex = 0; objIndex < gObjects.size(); ++objIndex) {
            const RenderObject& obj = gObjects[objIndex];

            // materialID == 1 -> model -> morph buffer
            // else -> static vertex buffer
            vk::Buffer vb = gVertexBuffer;
            if (obj.materialID == 1 && gMorphBufferCreated) {
                vb = gMorphVertexBuffer;
            }

            vk::DeviceSize offs = 0;
            cmd.bindVertexBuffers(0, vb, offs);

            PushConstants pc{};
            pc.model = obj.transform.toMatrix();
            pc.objectID = (int)objIndex;
            pc.roughnessOverride = 1.0f;
            pc.metallicOverride = 0.0f;

            cmd.pushConstants(
                gPipelineLayout,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                0,
                sizeof(PushConstants),
                &pc
            );

            const Material& mat = gMaterials[obj.materialID];

            vk::DescriptorImageInfo ii{};
            ii.sampler = mat.sampler;
            ii.imageView = mat.view;
            ii.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

            vk::WriteDescriptorSet write{};
            write.dstSet = gGlobalDescSets[i];
            write.dstBinding = 1;
            write.descriptorCount = 1;
            write.descriptorType = vk::DescriptorType::eCombinedImageSampler;
            write.pImageInfo = &ii;

            gDevice.updateDescriptorSets(1, &write, 0, nullptr);

            cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                gPipelineLayout,
                0,
                gGlobalDescSets[i],
                nullptr
            );

            cmd.drawIndexed(obj.indexCount, 1, obj.indexOffset, 0, 0);
        }

        cmd.endRenderPass();
        cmd.end();
    }
}

void createSyncObjects() {
    vk::SemaphoreCreateInfo si{};
    gImgAvailable = gDevice.createSemaphore(si);
    gRenderFinished = gDevice.createSemaphore(si);

    vk::FenceCreateInfo fi{};
    fi.flags = vk::FenceCreateFlagBits::eSignaled;
    gInFlight = gDevice.createFence(fi);
}

void updateUBO(uint32_t imageIndex) {
    GlobalUBO ubo{};

    ubo.view = gCamera.getView();
    ubo.proj = glm::perspective(
        glm::radians(60.0f),
        gSwapExtent.width / (float)gSwapExtent.height,
        0.1f, 500.0f
    );
    ubo.proj[1][1] *= -1.0f;

    glm::vec3 lightDir = glm::normalize(glm::vec3(-0.5f, -1.0f, -0.3f));
    ubo.lightDir = glm::vec4(lightDir, 0.0f);
    ubo.lightColor = glm::vec4(3.0f, 2.6f, 2.2f, 1.0f);

    void* data = gDevice.mapMemory(gUBOMem[imageIndex], 0, sizeof(GlobalUBO));
    memcpy(data, &ubo, sizeof(GlobalUBO));
    gDevice.unmapMemory(gUBOMem[imageIndex]);
}
// ======================= main.cpp  PART 7/9 =======================
// Input system: keyboard, mouse, controller

void processKeyboard(float dt) {
    if (!gWindow) return;

    const float velocity = gCamera.speed * dt;

    auto forward = [&]() {
        glm::vec3 front;
        front.x = cos(glm::radians(gCamera.yaw)) * cos(glm::radians(gCamera.pitch));
        front.y = sin(glm::radians(gCamera.pitch));
        front.z = sin(glm::radians(gCamera.yaw)) * cos(glm::radians(gCamera.pitch));
        return glm::normalize(front);
        };

    glm::vec3 f = forward();
    glm::vec3 right = glm::normalize(glm::cross(f, glm::vec3(0, 1, 0)));

    if (glfwGetKey(gWindow, GLFW_KEY_W) == GLFW_PRESS) gCamera.position += f * velocity;
    if (glfwGetKey(gWindow, GLFW_KEY_S) == GLFW_PRESS) gCamera.position -= f * velocity;
    if (glfwGetKey(gWindow, GLFW_KEY_A) == GLFW_PRESS) gCamera.position -= right * velocity;
    if (glfwGetKey(gWindow, GLFW_KEY_D) == GLFW_PRESS) gCamera.position += right * velocity;
}

void processController(float dt) {
    if (!gWindow) return;

    if (!glfwJoystickIsGamepad(GLFW_JOYSTICK_1))
        return;

    GLFWgamepadstate state{};
    if (!glfwGetGamepadState(GLFW_JOYSTICK_1, &state))
        return;

    const float velocity = gCamera.speed * dt;

    auto forward = [&]() {
        glm::vec3 front;
        front.x = cos(glm::radians(gCamera.yaw)) * cos(glm::radians(gCamera.pitch));
        front.y = sin(glm::radians(gCamera.pitch));
        front.z = sin(glm::radians(gCamera.yaw)) * cos(glm::radians(gCamera.pitch));
        return glm::normalize(front);
        };

    glm::vec3 f = forward();
    glm::vec3 right = glm::normalize(glm::cross(f, glm::vec3(0, 1, 0)));

    float lx = state.axes[GLFW_GAMEPAD_AXIS_LEFT_X];
    float ly = state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];
    float rx = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];
    float ry = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];

    auto deadzone = [](float v, float dz) {
        return (std::fabs(v) < dz) ? 0.0f : v;
        };

    lx = deadzone(lx, 0.15f);
    ly = deadzone(ly, 0.15f);
    rx = deadzone(rx, 0.15f);
    ry = deadzone(ry, 0.15f);

    gCamera.position += f * (-ly * velocity);
    gCamera.position += right * (lx * velocity);

    float lookScale = gCamera.sensitivity * 100.0f * dt;
    gCamera.yaw += rx * lookScale;
    gCamera.pitch -= ry * lookScale;

    if (gCamera.pitch > 89.0f)  gCamera.pitch = 89.0f;
    if (gCamera.pitch < -89.0f) gCamera.pitch = -89.0f;
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (gFirstMouse) {
        gLastX = xpos;
        gLastY = ypos;
        gFirstMouse = false;
    }

    float xoffset = float(xpos - gLastX);
    float yoffset = float(gLastY - ypos);
    gLastX = xpos;
    gLastY = ypos;

    xoffset *= gCamera.sensitivity;
    yoffset *= gCamera.sensitivity;

    gCamera.yaw += xoffset;
    gCamera.pitch += yoffset;

    if (gCamera.pitch > 89.0f)  gCamera.pitch = 89.0f;
    if (gCamera.pitch < -89.0f) gCamera.pitch = -89.0f;
}

void setupInput() {
    glfwSetCursorPosCallback(gWindow, mouseCallback);
    glfwSetInputMode(gWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}
// ======================= main.cpp  PART 8/9 =======================
// Frame rendering and main loop

void drawFrame() {
    gDevice.waitForFences(gInFlight, VK_TRUE, UINT64_MAX);
    gDevice.resetFences(gInFlight);

    uint32_t imageIndex =
        gDevice.acquireNextImageKHR(
            gSwapchain,
            UINT64_MAX,
            gImgAvailable,
            {}
        ).value;

    updateUBO(imageIndex);

    vk::PipelineStageFlags waitStage =
        vk::PipelineStageFlagBits::eColorAttachmentOutput;

    vk::SubmitInfo si{};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &gImgAvailable;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &gCmdBuffers[imageIndex];
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &gRenderFinished;

    gGraphicsQueue.submit(si, gInFlight);

    vk::PresentInfoKHR pi{};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &gRenderFinished;
    pi.swapchainCount = 1;
    pi.pSwapchains = &gSwapchain;
    pi.pImageIndices = &imageIndex;

    gGraphicsQueue.presentKHR(pi);
}

float getDeltaTime() {
    auto now = std::chrono::steady_clock::now();

    if (gLastFrameTime.time_since_epoch().count() == 0) {
        gLastFrameTime = now;
        return 0.0f;
    }

    float dt = std::chrono::duration<float>(now - gLastFrameTime).count();
    gLastFrameTime = now;
    return dt;
}

void mainLoop() {
    gLastFrameTime = std::chrono::steady_clock::now();
    static float morphTime = 0.0f;

    while (!glfwWindowShouldClose(gWindow)) {
        glfwPollEvents();

        float dt = getDeltaTime();

        processKeyboard(dt);
        processController(dt);

        morphTime += dt;
        updateMorphMeshes(morphTime);  // real glTF morph animation

        drawFrame();
    }

    gDevice.waitIdle();
}
// ======================= main.cpp  PART 9/9 =======================
// Initialization, cleanup, entry point

void loadPlaneTexture(const char* path) {
    int w = 0, h = 0, c = 0;
    stbi_uc* pixels = stbi_load(path, &w, &h, &c, STBI_rgb_alpha);
    if (!pixels)
        throw std::runtime_error("Failed to load plane texture");

    LoadedImage img;
    img.width = w;
    img.height = h;
    img.pixels.assign(pixels, pixels + (w * h * 4));
    stbi_image_free(pixels);

    createVulkanTextureFromRGBA(
        img,
        gPlaneTexImage,
        gPlaneTexMemory,
        gPlaneTexView,
        gPlaneTexSampler
    );
}

void initVulkan() {
    createInstance();
    createSurface();
    pickGPU();
    createDevice();
    createSwapchain();
    createDepthResources();
    createRenderPass();
    createFramebuffers();
    createDescriptorSetLayout();
    createPipelineLayout();
    createPipeline();
    createCommandPool();

    loadPlaneTexture("grid.png");
    loadGLTFModel("dummy.glb");

    gMaterials.clear();
    gMaterials.push_back({ gPlaneTexView, gPlaneTexSampler }); // material 0
    gMaterials.push_back({ gModelTexView, gModelTexSampler }); // material 1

    addPlaneAndCubeGeometry();
    appendModelToGeometry();

    createSharedBuffers();
    createMorphVertexBufferIfNeeded();

    createUBOs();
    createDescriptorSets();

    createCommandBuffers();
    createSyncObjects();
}

void cleanup() {
    gDevice.waitIdle();

    gDevice.destroyFence(gInFlight);
    gDevice.destroySemaphore(gRenderFinished);
    gDevice.destroySemaphore(gImgAvailable);

    if (gMorphBufferCreated) {
        gDevice.destroyBuffer(gMorphVertexBuffer);
        gDevice.freeMemory(gMorphVertexMemory);
    }

    gDevice.destroyBuffer(gVertexBuffer);
    gDevice.freeMemory(gVertexMemory);
    gDevice.destroyBuffer(gIndexBuffer);
    gDevice.freeMemory(gIndexMemory);

    for (size_t i = 0; i < gUBOs.size(); ++i) {
        gDevice.destroyBuffer(gUBOs[i]);
        gDevice.freeMemory(gUBOMem[i]);
    }

    gDevice.destroySampler(gPlaneTexSampler);
    gDevice.destroyImageView(gPlaneTexView);
    gDevice.destroyImage(gPlaneTexImage);
    gDevice.freeMemory(gPlaneTexMemory);

    gDevice.destroySampler(gModelTexSampler);
    gDevice.destroyImageView(gModelTexView);
    gDevice.destroyImage(gModelTexImage);
    gDevice.freeMemory(gModelTexMemory);

    gDevice.destroyImageView(gDepthView);
    gDevice.destroyImage(gDepthImage);
    gDevice.freeMemory(gDepthMemory);

    for (auto fb : gFramebuffers)
        gDevice.destroyFramebuffer(fb);

    gDevice.destroyRenderPass(gRenderPass);

    for (auto v : gSwapViews)
        gDevice.destroyImageView(v);

    gDevice.destroySwapchainKHR(gSwapchain);

    gDevice.destroyDescriptorPool(gDescPool);
    gDevice.destroyDescriptorSetLayout(gDescSetLayout);

    gDevice.destroyPipeline(gPipeline);
    gDevice.destroyPipelineLayout(gPipelineLayout);

    gDevice.destroyCommandPool(gCmdPool);

    gDevice.destroy();
    gInstance.destroySurfaceKHR(gSurface);
    gInstance.destroy();

    glfwDestroyWindow(gWindow);
    glfwTerminate();
}

int main() {
    try {
        initWindow();
        initVulkan();
        setupInput();
        mainLoop();
        cleanup();
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
