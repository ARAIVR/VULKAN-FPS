// ======================= main.cpp (ECS + morphs) =======================
#define _CRT_SECURE_NO_WARNINGS

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
#include "tinygltf/stb_image.h"

#include "ECS.hpp"
#include "Components.hpp"
#include "Systems.hpp"

// ----------------------------------------------------------
// Globals (matching externs in Systems.hpp)
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

vk::Image        gDepthImage;
vk::DeviceMemory gDepthMemory;
vk::ImageView    gDepthView;
vk::Format       gDepthFormat = vk::Format::eD32Sfloat;

vk::Image        gPlaneTexImage;
vk::DeviceMemory gPlaneTexMemory;
vk::ImageView    gPlaneTexView;
vk::Sampler      gPlaneTexSampler;

vk::Image        gDummyTexImage;
vk::DeviceMemory gDummyTexMemory;
vk::ImageView    gDummyTexView;
vk::Sampler      gDummyTexSampler;

vk::Buffer       gVertexBuffer;
vk::DeviceMemory gVertexMemory;
vk::Buffer       gIndexBuffer;
vk::DeviceMemory gIndexMemory;

// Morph vertex buffer (dynamic, CPU-updated each frame)
vk::Buffer       gMorphVertexBuffer;
vk::DeviceMemory gMorphVertexMemory;
bool             gMorphBufferCreated = false;

vk::CommandPool                 gCmdPool;
std::vector<vk::CommandBuffer>  gCmdBuffers;

vk::Semaphore       gImgAvailable;
vk::Semaphore       gRenderFinished;
vk::Fence           gInFlight;

std::vector<vk::Buffer>       gUBOs;
std::vector<vk::DeviceMemory> gUBOMem;

std::chrono::steady_clock::time_point gLastFrameTime;

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
};

std::vector<Vertex>   gVertices;
std::vector<uint32_t> gIndices;

std::vector<Material> gMaterials;

// ----------------------------------------------------------
// Morph data (dummy.glb only)
// ----------------------------------------------------------

struct MorphTarget {
    std::vector<glm::vec3> deltaPos;
    std::vector<glm::vec3> deltaNormal;
};

struct MorphMesh {
    uint32_t baseVertexOffset = 0;
    uint32_t vertexCount = 0;
    std::vector<MorphTarget> targets;
};

MorphMesh gDummyMorph;

// Simple animation sampler for weights
struct MorphAnimationSampler {
    std::vector<float> times;                 // keyframe times
    std::vector<std::vector<float>> weights;  // [key][targetIndex]
};

struct MorphAnimation {
    MorphAnimationSampler sampler;
    float duration = 0.0f;
};

MorphAnimation gMorphAnimation;
float          gMorphTime = 0.0f;

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

void createBuffer(
    vk::DeviceSize size,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags props,
    vk::Buffer& buf,
    vk::DeviceMemory& mem
) {
    vk::BufferCreateInfo bi({}, size, usage, vk::SharingMode::eExclusive);
    buf = gDevice.createBuffer(bi);
    auto req = gDevice.getBufferMemoryRequirements(buf);
    vk::MemoryAllocateInfo ai(req.size, findMemoryType(req.memoryTypeBits, props));
    mem = gDevice.allocateMemory(ai);
    gDevice.bindBufferMemory(buf, mem, 0);
}

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

struct LoadedImage {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;
};

void createVulkanTextureFromRGBA(
    const LoadedImage& img,
    vk::Image& outImage,
    vk::DeviceMemory& outMemory,
    vk::ImageView& outView,
    vk::Sampler& outSampler
) {
    vk::DeviceSize imageSize = (vk::DeviceSize)(img.width * img.height * 4);

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

// ----------------------------------------------------------
// Vulkan setup (window, instance, device, swapchain, etc.)
// ----------------------------------------------------------

void initWindow() {
    if (!glfwInit())
        throw std::runtime_error("GLFW init failed");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    gWindow = glfwCreateWindow(1280, 720, "Vulkan ECS Morphs", nullptr, nullptr);

    if (!gWindow)
        throw std::runtime_error("Failed to create window");
}

void createInstance() {
    vk::ApplicationInfo appInfo(
        "Vulkan ECS", 1,
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

void createSwapchain() {
    auto caps = gGPU.getSurfaceCapabilitiesKHR(gSurface);
    auto fmts = gGPU.getSurfaceFormatsKHR(gSurface);

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

struct PushConstants {
    glm::mat4 model;
    int   objectID;
    float roughnessOverride;
    float metallicOverride;
};

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

// ----------------------------------------------------------
// Command pool, buffers, sync, UBOs, descriptors
// ----------------------------------------------------------

void createCommandPool() {
    vk::CommandPoolCreateInfo ci{};
    ci.queueFamilyIndex = gGraphicsQueueFamily;
    ci.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    gCmdPool = gDevice.createCommandPool(ci);
}

void allocateCommandBuffers() {
    gCmdBuffers.resize(gSwapImages.size());
    vk::CommandBufferAllocateInfo ai{};
    ai.commandPool = gCmdPool;
    ai.level = vk::CommandBufferLevel::ePrimary;
    ai.commandBufferCount = (uint32_t)gCmdBuffers.size();
    gCmdBuffers = gDevice.allocateCommandBuffers(ai);
}

void createSyncObjects() {
    vk::SemaphoreCreateInfo si{};
    gImgAvailable = gDevice.createSemaphore(si);
    gRenderFinished = gDevice.createSemaphore(si);

    vk::FenceCreateInfo fi{};
    fi.flags = vk::FenceCreateFlagBits::eSignaled;
    gInFlight = gDevice.createFence(fi);
}

void createUBOs() {
    gUBOs.resize(gSwapImages.size());
    gUBOMem.resize(gSwapImages.size());

    for (size_t i = 0; i < gUBOs.size(); ++i) {
        createBuffer(
            sizeof(GlobalUBO),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            gUBOs[i],
            gUBOMem[i]
        );
    }
}

void createDescriptorPoolAndSets() {
    std::array<vk::DescriptorPoolSize, 2> sizes{};
    sizes[0].type = vk::DescriptorType::eUniformBuffer;
    sizes[0].descriptorCount = (uint32_t)gSwapImages.size();
    sizes[1].type = vk::DescriptorType::eCombinedImageSampler;
    sizes[1].descriptorCount = (uint32_t)gSwapImages.size();

    vk::DescriptorPoolCreateInfo pi{};
    pi.maxSets = (uint32_t)gSwapImages.size();
    pi.poolSizeCount = (uint32_t)sizes.size();
    pi.pPoolSizes = sizes.data();

    gDescPool = gDevice.createDescriptorPool(pi);

    std::vector<vk::DescriptorSetLayout> layouts(gSwapImages.size(), gDescSetLayout);
    vk::DescriptorSetAllocateInfo ai{};
    ai.descriptorPool = gDescPool;
    ai.descriptorSetCount = (uint32_t)layouts.size();
    ai.pSetLayouts = layouts.data();

    gGlobalDescSets = gDevice.allocateDescriptorSets(ai);
}

void initUBODescriptorSets() {
    for (size_t i = 0; i < gGlobalDescSets.size(); ++i) {
        vk::DescriptorBufferInfo bi{};
        bi.buffer = gUBOs[i];
        bi.offset = 0;
        bi.range = sizeof(GlobalUBO);

        vk::WriteDescriptorSet write{};
        write.dstSet = gGlobalDescSets[i];
        write.dstBinding = 0;
        write.descriptorType = vk::DescriptorType::eUniformBuffer;
        write.descriptorCount = 1;
        write.pBufferInfo = &bi;

        gDevice.updateDescriptorSets(1, &write, 0, nullptr);
    }
}

void bindMaterialForDraw(uint32_t imageIndex, uint32_t materialID) {
    if (materialID >= gMaterials.size()) return;
    Material& mat = gMaterials[materialID];

    vk::DescriptorImageInfo ii{};
    ii.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    ii.imageView = mat.view;
    ii.sampler = mat.sampler;

    vk::WriteDescriptorSet write{};
    write.dstSet = gGlobalDescSets[imageIndex];
    write.dstBinding = 1;
    write.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    write.descriptorCount = 1;
    write.pImageInfo = &ii;

    gDevice.updateDescriptorSets(1, &write, 0, nullptr);
}

// ----------------------------------------------------------
// Geometry: plane + cube
// ----------------------------------------------------------

struct MeshRange {
    uint32_t vertexOffset;
    uint32_t indexOffset;
    uint32_t indexCount;
};

void addPlaneAndCubeGeometry(MeshRange& outPlane, MeshRange& outCube) {
    gVertices.clear();
    gIndices.clear();

    float groundHalf = 50.0f;
    float groundUV = 10.0f;

    outPlane.vertexOffset = (uint32_t)gVertices.size();
    outPlane.indexOffset = (uint32_t)gIndices.size();

    gVertices.push_back({ { -groundHalf, 0, -groundHalf }, {0,1,0}, {0,0} });
    gVertices.push_back({ {  groundHalf, 0, -groundHalf }, {0,1,0}, {groundUV,0} });
    gVertices.push_back({ {  groundHalf, 0,  groundHalf }, {0,1,0}, {groundUV,groundUV} });
    gVertices.push_back({ { -groundHalf, 0,  groundHalf }, {0,1,0}, {0,groundUV} });

    gIndices.insert(gIndices.end(), {
        outPlane.vertexOffset + 0, outPlane.vertexOffset + 3, outPlane.vertexOffset + 2,
        outPlane.vertexOffset + 2, outPlane.vertexOffset + 1, outPlane.vertexOffset + 0
        });

    outPlane.indexCount = 6;

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

    outCube.vertexOffset = (uint32_t)gVertices.size();
    outCube.indexOffset = (uint32_t)gIndices.size();

    float h = 5.0f;
    float y0 = 0.0f;
    float y1 = 10.0f;

    addFace({ -h,y1,-h }, { h,y1,-h }, { h,y1,h }, { -h,y1,h }, { 0,1,0 });
    addFace({ -h,y0,h }, { h,y0,h }, { h,y0,-h }, { -h,y0,-h }, { 0,-1,0 });
    addFace({ h,y0,-h }, { h,y0,h }, { h,y1,h }, { h,y1,-h }, { 1,0,0 });
    addFace({ -h,y0,h }, { -h,y0,-h }, { -h,y1,-h }, { -h,y1,h }, { -1,0,0 });
    addFace({ -h,y0,h }, { h,y0,h }, { h,y1,h }, { -h,y1,h }, { 0,0,1 });
    addFace({ h,y0,-h }, { -h,y0,-h }, { -h,y1,-h }, { h,y1,-h }, { 0,0,-1 });

    outCube.indexCount = (uint32_t)gIndices.size() - outCube.indexOffset;
}

void uploadSharedBuffers() {
    vk::DeviceSize vSize = sizeof(Vertex) * gVertices.size();
    vk::DeviceSize iSize = sizeof(uint32_t) * gIndices.size();

    vk::Buffer stagingV;
    vk::DeviceMemory stagingVMem;
    createBuffer(
        vSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingV,
        stagingVMem
    );

    void* data = gDevice.mapMemory(stagingVMem, 0, vSize);
    std::memcpy(data, gVertices.data(), (size_t)vSize);
    gDevice.unmapMemory(stagingVMem);

    vk::Buffer stagingI;
    vk::DeviceMemory stagingIMem;
    createBuffer(
        iSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingI,
        stagingIMem
    );

    data = gDevice.mapMemory(stagingIMem, 0, iSize);
    std::memcpy(data, gIndices.data(), (size_t)iSize);
    gDevice.unmapMemory(stagingIMem);

    createBuffer(
        vSize,
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        gVertexBuffer,
        gVertexMemory
    );

    createBuffer(
        iSize,
        vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        gIndexBuffer,
        gIndexMemory
    );

    vk::CommandBuffer cmd = beginSingleTimeCommands();

    vk::BufferCopy copyV{};
    copyV.size = vSize;
    cmd.copyBuffer(stagingV, gVertexBuffer, 1, &copyV);

    vk::BufferCopy copyI{};
    copyI.size = iSize;
    cmd.copyBuffer(stagingI, gIndexBuffer, 1, &copyI);

    endSingleTimeCommands(cmd);

    gDevice.destroyBuffer(stagingV);
    gDevice.freeMemory(stagingVMem);
    gDevice.destroyBuffer(stagingI);
    gDevice.freeMemory(stagingIMem);
}

// ----------------------------------------------------------
// Image loading helpers
// ----------------------------------------------------------

LoadedImage loadPNG(const char* path) {
    LoadedImage img;
    int comp = 0;
    unsigned char* data = stbi_load(path, &img.width, &img.height, &comp, 4);
    if (!data) throw std::runtime_error(std::string("Failed to load image: ") + path);
    img.pixels.assign(data, data + img.width * img.height * 4);
    stbi_image_free(data);
    return img;
}

LoadedImage loadImageFromCgltfImage(const cgltf_image* image, const char* gltfPath) {
    LoadedImage img;

    if (image->uri && image->uri[0] != '\0') {
        std::string base(gltfPath);
        size_t slash = base.find_last_of("/\\");
        std::string dir = (slash == std::string::npos) ? "" : base.substr(0, slash + 1);
        std::string fullPath = dir + image->uri;
        return loadPNG(fullPath.c_str());
    }

    if (image->buffer_view && image->buffer_view->buffer && image->buffer_view->buffer->data) {
        const cgltf_buffer_view* bv = image->buffer_view;
        const cgltf_buffer* buf = bv->buffer;
        const unsigned char* base = static_cast<const unsigned char*>(buf->data);
        const unsigned char* ptr = base + bv->offset;
        int comp = 0;
        unsigned char* data = stbi_load_from_memory(
            ptr,
            (int)bv->size,
            &img.width,
            &img.height,
            &comp,
            4
        );
        if (!data) {
            throw std::runtime_error("Failed to load embedded image from glb");
        }
        img.pixels.assign(data, data + img.width * img.height * 4);
        stbi_image_free(data);
        return img;
    }

    throw std::runtime_error("cgltf_image has no uri or buffer_view");
}

// ----------------------------------------------------------
// Minimal dummy.glb loader with morph targets
// ----------------------------------------------------------

struct GLTFLoadedMesh {
    uint32_t vertexOffset;
    uint32_t indexOffset;
    uint32_t indexCount;
};

GLTFLoadedMesh loadDummyGLB(const char* path, uint32_t& outMaterialID) {
    cgltf_options options{};
    cgltf_data* data = nullptr;
    cgltf_result res = cgltf_parse_file(&options, path, &data);
    if (res != cgltf_result_success) {
        throw std::runtime_error("Failed to parse glTF");
    }
    res = cgltf_load_buffers(&options, data, path);
    if (res != cgltf_result_success) {
        cgltf_free(data);
        throw std::runtime_error("Failed to load glTF buffers");
    }

    if (data->meshes_count == 0) {
        cgltf_free(data);
        throw std::runtime_error("dummy.glb has no meshes");
    }

    cgltf_mesh& mesh = data->meshes[0];
    if (mesh.primitives_count == 0) {
        cgltf_free(data);
        throw std::runtime_error("dummy.glb has no primitives");
    }

    cgltf_primitive& prim = mesh.primitives[0];

    cgltf_accessor* posAcc = nullptr;
    cgltf_accessor* normAcc = nullptr;
    cgltf_accessor* uvAcc = nullptr;
    for (size_t i = 0; i < prim.attributes_count; ++i) {
        cgltf_attribute& a = prim.attributes[i];
        if (a.type == cgltf_attribute_type_position) posAcc = a.data;
        if (a.type == cgltf_attribute_type_normal)   normAcc = a.data;
        if (a.type == cgltf_attribute_type_texcoord && a.index == 0) uvAcc = a.data;
    }
    if (!posAcc || !normAcc || !uvAcc || !prim.indices) {
        cgltf_free(data);
        throw std::runtime_error("dummy.glb missing pos/norm/uv/indices");
    }

    uint32_t baseV = (uint32_t)gVertices.size();
    uint32_t baseI = (uint32_t)gIndices.size();

    size_t vCount = posAcc->count;
    gVertices.resize(baseV + vCount);

    auto readVec3 = [](cgltf_accessor* acc, size_t index) {
        cgltf_float tmp[4];
        cgltf_accessor_read_float(acc, index, tmp, 4);
        return glm::vec3(tmp[0], tmp[1], tmp[2]);
        };
    auto readVec2 = [](cgltf_accessor* acc, size_t index) {
        cgltf_float tmp[4];
        cgltf_accessor_read_float(acc, index, tmp, 4);
        return glm::vec2(tmp[0], tmp[1]);
        };

    for (size_t i = 0; i < vCount; ++i) {
        glm::vec3 p = readVec3(posAcc, i);
        glm::vec3 n = readVec3(normAcc, i);
        glm::vec2 uv = readVec2(uvAcc, i);
        gVertices[baseV + i] = { p, n, uv };
    }

    size_t indexCount = prim.indices->count;
    gIndices.resize(baseI + indexCount);
    for (size_t i = 0; i < indexCount; ++i) {
        gIndices[baseI + i] = (uint32_t)cgltf_accessor_read_index(prim.indices, i);
    }

    // Material + texture (proper UV-based sampling)
    vk::ImageView texView = gPlaneTexView;
    vk::Sampler   texSampler = gPlaneTexSampler;

    if (prim.material && prim.material->pbr_metallic_roughness.base_color_texture.texture) {
        cgltf_texture* tex = prim.material->pbr_metallic_roughness.base_color_texture.texture;
        if (tex->image) {
            LoadedImage img = loadImageFromCgltfImage(tex->image, path);
            createVulkanTextureFromRGBA(
                img,
                gDummyTexImage,
                gDummyTexMemory,
                gDummyTexView,
                gDummyTexSampler
            );
            texView = gDummyTexView;
            texSampler = gDummyTexSampler;
        }
    }

    outMaterialID = (uint32_t)gMaterials.size();
    gMaterials.push_back(Material{ texView, texSampler });

    // --------- Morph targets (POSITION / NORMAL deltas) ----------
    gDummyMorph.baseVertexOffset = baseV;
    gDummyMorph.vertexCount = (uint32_t)vCount;
    gDummyMorph.targets.clear();

    if (prim.targets_count > 0) {
        gDummyMorph.targets.resize(prim.targets_count);
        for (size_t t = 0; t < prim.targets_count; ++t) {
            MorphTarget& mt = gDummyMorph.targets[t];
            mt.deltaPos.resize(vCount, glm::vec3(0.0f));
            mt.deltaNormal.resize(vCount, glm::vec3(0.0f));

            cgltf_accessor* posDeltaAcc = nullptr;
            cgltf_accessor* normDeltaAcc = nullptr;

            for (size_t a = 0; a < prim.targets[t].attributes_count; ++a) {
                cgltf_attribute& ta = prim.targets[t].attributes[a];
                if (ta.type == cgltf_attribute_type_position) posDeltaAcc = ta.data;
                if (ta.type == cgltf_attribute_type_normal)   normDeltaAcc = ta.data;
            }

            if (posDeltaAcc) {
                for (size_t i = 0; i < vCount; ++i) {
                    mt.deltaPos[i] = readVec3(posDeltaAcc, i);
                }
            }
            if (normDeltaAcc) {
                for (size_t i = 0; i < vCount; ++i) {
                    mt.deltaNormal[i] = readVec3(normDeltaAcc, i);
                }
            }
        }
    }

    // --------- Animation (WEIGHTS) ----------
    gMorphAnimation = MorphAnimation{};
    if (data->animations_count > 0 && prim.targets_count > 0) {
        cgltf_animation& anim = data->animations[0];

        for (size_t c = 0; c < anim.channels_count; ++c) {
            cgltf_animation_channel& ch = anim.channels[c];
            if (!ch.target_node || !ch.sampler) continue;
            if (ch.target_path != cgltf_animation_path_type_weights) continue;
            if (!ch.target_node->mesh || ch.target_node->mesh != &mesh) continue;

            cgltf_animation_sampler* s = ch.sampler;
            cgltf_accessor* inAcc = s->input;
            cgltf_accessor* outAcc = s->output;
            if (!inAcc || !outAcc) continue;

            size_t keyCount = inAcc->count;
            gMorphAnimation.sampler.times.resize(keyCount);
            gMorphAnimation.sampler.weights.resize(keyCount);

            // times
            for (size_t k = 0; k < keyCount; ++k) {
                cgltf_float tmp[4];
                cgltf_accessor_read_float(inAcc, k, tmp, 4);
                gMorphAnimation.sampler.times[k] = tmp[0];
            }

            // weights: outAcc has keyCount * targetCount floats
            size_t targetCount = prim.targets_count;
            for (size_t k = 0; k < keyCount; ++k) {
                std::vector<float> w(targetCount, 0.0f);
                cgltf_accessor_read_float(outAcc, k * targetCount, w.data(), (cgltf_size)targetCount);
                gMorphAnimation.sampler.weights[k] = std::move(w);
            }

            if (keyCount > 0) {
                gMorphAnimation.duration = gMorphAnimation.sampler.times.back();
            }
            break;
        }
    }

    GLTFLoadedMesh result{};
    result.vertexOffset = baseV;
    result.indexOffset = baseI;
    result.indexCount = (uint32_t)indexCount;

    cgltf_free(data);
    return result;
}

// ----------------------------------------------------------
// Morph buffer + animation update
// ----------------------------------------------------------

void createMorphVertexBufferIfNeeded() {
    if (gMorphBufferCreated) return;
    if (gVertices.empty()) return;

    vk::DeviceSize size = sizeof(Vertex) * gVertices.size();
    createBuffer(
        size,
        vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        gMorphVertexBuffer,
        gMorphVertexMemory
    );

    void* data = gDevice.mapMemory(gMorphVertexMemory, 0, size);
    std::memcpy(data, gVertices.data(), (size_t)size);
    gDevice.unmapMemory(gMorphVertexMemory);

    gMorphBufferCreated = true;
}

void evaluateMorphAnimation(float time, std::vector<float>& outWeights) {
    outWeights.clear();
    if (gMorphAnimation.sampler.times.empty() ||
        gMorphAnimation.sampler.weights.empty() ||
        gMorphAnimation.duration <= 0.0f) {
        return;
    }

    float t = std::fmod(time, gMorphAnimation.duration);
    auto& times = gMorphAnimation.sampler.times;
    auto& weights = gMorphAnimation.sampler.weights;

    size_t count = times.size();
    if (count == 1) {
        outWeights = weights[0];
        return;
    }

    size_t i1 = 0;
    while (i1 < count && times[i1] < t) ++i1;
    if (i1 == 0) {
        outWeights = weights[0];
        return;
    }
    if (i1 >= count) {
        outWeights = weights[count - 1];
        return;
    }

    size_t i0 = i1 - 1;
    float t0 = times[i0];
    float t1 = times[i1];
    float alpha = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0f;

    auto& w0 = weights[i0];
    auto& w1 = weights[i1];
    size_t n = std::min(w0.size(), w1.size());
    outWeights.resize(n);
    for (size_t i = 0; i < n; ++i) {
        outWeights[i] = (1.0f - alpha) * w0[i] + alpha * w1[i];
    }
}

void updateMorphMeshes(float dt) {
    gMorphTime += dt;

    if (gDummyMorph.vertexCount == 0 || gDummyMorph.targets.empty())
        return;

    createMorphVertexBufferIfNeeded();
    if (!gMorphBufferCreated) return;

    // Start from base geometry
    std::vector<Vertex> morphed = gVertices;

    // Evaluate animation weights
    std::vector<float> animWeights;
    evaluateMorphAnimation(gMorphTime, animWeights);

    size_t targetCount = gDummyMorph.targets.size();
    if (animWeights.size() < targetCount) {
        animWeights.resize(targetCount, 0.0f);
    }

    uint32_t base = gDummyMorph.baseVertexOffset;
    uint32_t vCount = gDummyMorph.vertexCount;

    for (uint32_t i = 0; i < vCount; ++i) {
        Vertex v = gVertices[base + i];
        glm::vec3 pos = v.pos;
        glm::vec3 nrm = v.normal;

        for (size_t t = 0; t < targetCount; ++t) {
            float w = animWeights[t];
            if (w == 0.0f) continue;
            pos += gDummyMorph.targets[t].deltaPos[i] * w;
            nrm += gDummyMorph.targets[t].deltaNormal[i] * w;
        }

        if (glm::length(nrm) > 0.0001f) nrm = glm::normalize(nrm);
        morphed[base + i] = { pos, nrm, v.uv };
    }

    vk::DeviceSize size = sizeof(Vertex) * morphed.size();
    void* data = gDevice.mapMemory(gMorphVertexMemory, 0, size);
    std::memcpy(data, morphed.data(), (size_t)size);
    gDevice.unmapMemory(gMorphVertexMemory);
}

// ----------------------------------------------------------
// Main
// ----------------------------------------------------------

int main() {
    try {
        initWindow();
        createInstance();
        createSurface();
        pickGPU();
        createDevice();
        createSwapchain();
        createDepthResources();
        createRenderPass();
        createDescriptorSetLayout();
        createPipelineLayout();
        createPipeline();
        createFramebuffers();
        createCommandPool();
        allocateCommandBuffers();
        createSyncObjects();
        createUBOs();
        createDescriptorPoolAndSets();

        // Load grid.png for plane/cube
        LoadedImage gridImg = loadPNG("grid.png");
        createVulkanTextureFromRGBA(
            gridImg,
            gPlaneTexImage,
            gPlaneTexMemory,
            gPlaneTexView,
            gPlaneTexSampler
        );
        uint32_t gridMatID = (uint32_t)gMaterials.size();
        gMaterials.push_back(Material{ gPlaneTexView, gPlaneTexSampler });

        // Geometry: plane + cube (separate ranges)
        MeshRange planeRange{}, cubeRange{};
        addPlaneAndCubeGeometry(planeRange, cubeRange);

        // Load dummy.glb (appends vertices/indices, creates its own sampler if texture present)
        uint32_t dummyMatID = 0;
        GLTFLoadedMesh dummyMesh = loadDummyGLB("dummy.glb", dummyMatID);

        // Upload combined buffers
        uploadSharedBuffers();

        // Init UBO descriptors (binding 0 only)
        initUBODescriptorSets();

        // ECS setup
        ECSRegistry   reg;
        ECSComponents comps;

        // Camera entity
        Entity cam = ecsCreateEntity(reg);
        TransformComponent camT{};
        camT.position = glm::vec3(0.0f, 8.0f, -25.0f);
        CameraComponent camC{};
        camC.pitch = -10.0f;
        camC.yaw = 90.0f;
        camC.fov = 60.0f;
        camC.speed = 10.0f;
        camC.sensitivity = 0.1f;
        InputComponent camInput{};

        comps.transforms.add(cam, camT);
        comps.cameras.add(cam, camC);
        comps.inputs.add(cam, camInput);

        // Plane entity (grid.png)
        Entity planeE = ecsCreateEntity(reg);
        TransformComponent planeT{};
        planeT.position = glm::vec3(0.0f);
        planeT.scale = glm::vec3(1.0f);
        MeshComponent planeM{};
        planeM.vertexOffset = planeRange.vertexOffset;
        planeM.indexOffset = planeRange.indexOffset;
        planeM.indexCount = planeRange.indexCount;
        planeM.materialID = gridMatID;
        MaterialComponent planeMat{};
        planeMat.materialID = gridMatID;

        comps.transforms.add(planeE, planeT);
        comps.meshes.add(planeE, planeM);
        comps.materials.add(planeE, planeMat);

        // Cube entity (grid.png)
        Entity cubeE = ecsCreateEntity(reg);
        TransformComponent cubeT{};
        cubeT.position = glm::vec3(0.0f, 0.0f, 0.0f);
        cubeT.scale = glm::vec3(1.0f);
        MeshComponent cubeM{};
        cubeM.vertexOffset = cubeRange.vertexOffset;
        cubeM.indexOffset = cubeRange.indexOffset;
        cubeM.indexCount = cubeRange.indexCount;
        cubeM.materialID = gridMatID;
        MaterialComponent cubeMat{};
        cubeMat.materialID = gridMatID;

        comps.transforms.add(cubeE, cubeT);
        comps.meshes.add(cubeE, cubeM);
        comps.materials.add(cubeE, cubeMat);

        // dummy.glb entity (its own texture, morphed)
        Entity dummyE = ecsCreateEntity(reg);
        TransformComponent dummyT{};
        dummyT.position = glm::vec3(0.0f, 5.0f, 0.0f);   // above ground
        dummyT.scale = glm::vec3(0.01f, 0.01f, 0.01f);   // glTF units
        MeshComponent dummyMC{};
        dummyMC.vertexOffset = dummyMesh.vertexOffset;
        dummyMC.indexOffset = dummyMesh.indexOffset;
        dummyMC.indexCount = dummyMesh.indexCount;
        dummyMC.materialID = dummyMatID;
        MaterialComponent dummyMatC{};
        dummyMatC.materialID = dummyMatID;

        comps.transforms.add(dummyE, dummyT);
        comps.meshes.add(dummyE, dummyMC);
        comps.materials.add(dummyE, dummyMatC);

        gLastFrameTime = std::chrono::steady_clock::now();

        while (!glfwWindowShouldClose(gWindow)) {
            glfwPollEvents();

            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - gLastFrameTime).count();
            gLastFrameTime = now;

            updateInputSystem(reg, comps, dt);
            updateMorphMeshes(dt);      // morph dummy.glb on CPU, upload to GPU
            renderFrame(reg, comps);
        }

        gDevice.waitIdle();
        std::cout << "Shutdown clean.\n";

        // Cleanup
        if (gMorphBufferCreated) {
            gDevice.destroyBuffer(gMorphVertexBuffer);
            gDevice.freeMemory(gMorphVertexMemory);
        }

        gDevice.destroyFence(gInFlight);
        gDevice.destroySemaphore(gRenderFinished);
        gDevice.destroySemaphore(gImgAvailable);

        for (size_t i = 0; i < gUBOs.size(); ++i) {
            gDevice.destroyBuffer(gUBOs[i]);
            gDevice.freeMemory(gUBOMem[i]);
        }

        gDevice.destroySampler(gPlaneTexSampler);
        gDevice.destroyImageView(gPlaneTexView);
        gDevice.destroyImage(gPlaneTexImage);
        gDevice.freeMemory(gPlaneTexMemory);

        gDevice.destroySampler(gDummyTexSampler);
        gDevice.destroyImageView(gDummyTexView);
        gDevice.destroyImage(gDummyTexImage);
        gDevice.freeMemory(gDummyTexMemory);

        gDevice.destroyBuffer(gVertexBuffer);
        gDevice.freeMemory(gVertexMemory);
        gDevice.destroyBuffer(gIndexBuffer);
        gDevice.freeMemory(gIndexMemory);

        gDevice.destroyImageView(gDepthView);
        gDevice.destroyImage(gDepthImage);
        gDevice.freeMemory(gDepthMemory);

        for (auto fb : gFramebuffers) gDevice.destroyFramebuffer(fb);
        gDevice.destroyRenderPass(gRenderPass);
        for (auto v : gSwapViews) gDevice.destroyImageView(v);
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
    catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
//new