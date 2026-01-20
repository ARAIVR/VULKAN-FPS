// ======================= Systems.hpp =======================
#pragma once
#include "ECS.hpp"
#include "Components.hpp"
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.hpp>
#include <vector>

// ==========================================================
//  MATERIAL TYPE
// ==========================================================
struct Material {
    vk::ImageView view;
    vk::Sampler   sampler;
};

extern std::vector<Material> gMaterials;

// ==========================================================
//  EXTERNAL VULKAN GLOBALS
// ==========================================================
extern GLFWwindow* gWindow;
extern vk::Device  gDevice;
extern vk::Queue   gGraphicsQueue;
extern vk::PipelineLayout gPipelineLayout;
extern vk::Pipeline       gPipeline;
extern std::vector<vk::Framebuffer> gFramebuffers;
extern vk::RenderPass     gRenderPass;
extern vk::Extent2D       gSwapExtent;
extern std::vector<vk::CommandBuffer> gCmdBuffers;
extern std::vector<vk::Buffer>        gUBOs;
extern std::vector<vk::DeviceMemory>  gUBOMem;
extern vk::Semaphore gImgAvailable;
extern vk::Semaphore gRenderFinished;
extern vk::Fence     gInFlight;
extern vk::SwapchainKHR gSwapchain;
extern vk::DescriptorSetLayout gDescSetLayout;
extern vk::DescriptorPool      gDescPool;
extern std::vector<vk::DescriptorSet> gGlobalDescSets;
extern vk::Buffer       gVertexBuffer;
extern vk::Buffer       gIndexBuffer;

// ==========================================================
//  GLOBAL UBO
// ==========================================================
struct GlobalUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 lightDir;
    glm::vec4 lightColor;
};

// ==========================================================
//  PER-IMAGE MATERIAL BINDING (implemented in main.cpp)
// ==========================================================
void bindMaterialForDraw(uint32_t imageIndex, uint32_t materialID);

// ==========================================================
//  INPUT SYSTEM
// ==========================================================
inline void updateInputSystem(
    ECSRegistry& reg,
    ECSComponents& comps,
    float dt
) {
    Entity controlled = INVALID_ENTITY;
    for (auto& kv : comps.inputs.data) {
        controlled = kv.first;
        break;
    }
    if (controlled == INVALID_ENTITY) return;

    auto* t = comps.transforms.get(controlled);
    auto* c = comps.cameras.get(controlled);
    if (!t || !c) return;

    // Keyboard movement
    glm::vec3 move(0.0f);
    if (glfwGetKey(gWindow, GLFW_KEY_W) == GLFW_PRESS) move.z += 1.0f;
    if (glfwGetKey(gWindow, GLFW_KEY_S) == GLFW_PRESS) move.z -= 1.0f;
    if (glfwGetKey(gWindow, GLFW_KEY_A) == GLFW_PRESS) move.x -= 1.0f;
    if (glfwGetKey(gWindow, GLFW_KEY_D) == GLFW_PRESS) move.x += 1.0f;

    // Mouse look
    static double lastX = 400.0, lastY = 300.0;
    static bool firstMouse = true;
    double xpos, ypos;
    glfwGetCursorPos(gWindow, &xpos, &ypos);
    if (firstMouse) {
        lastX = xpos; lastY = ypos;
        firstMouse = false;
    }
    float xoffset = float(xpos - lastX);
    float yoffset = float(lastY - ypos);
    lastX = xpos; lastY = ypos;

    xoffset *= c->sensitivity;
    yoffset *= c->sensitivity;

    c->yaw += xoffset;
    c->pitch += yoffset;
    if (c->pitch > 89.0f)  c->pitch = 89.0f;
    if (c->pitch < -89.0f) c->pitch = -89.0f;

    // Gamepad support (Xbox controller)
    if (glfwJoystickIsGamepad(GLFW_JOYSTICK_1)) {
        GLFWgamepadstate state;
        if (glfwGetGamepadState(GLFW_JOYSTICK_1, &state)) {
            float lx = state.axes[GLFW_GAMEPAD_AXIS_LEFT_X];
            float ly = state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];
            float rx = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];
            float ry = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];

            // Deadzone
            const float dead = 0.15f;
            auto applyDead = [&](float v) { return (fabs(v) < dead) ? 0.0f : v; };
            lx = applyDead(lx);
            ly = applyDead(ly);
            rx = applyDead(rx);
            ry = applyDead(ry);

            // Movement: left stick (match previous behavior)
            move.x += lx;
            move.z -= ly;

            // Look: right stick scaled by sensitivity and dt (match non-ECS code)
            float lookScale = c->sensitivity * 100.0f * dt;
            c->yaw += rx * lookScale;
            c->pitch -= ry * lookScale;
            if (c->pitch > 89.0f)  c->pitch = 89.0f;
            if (c->pitch < -89.0f) c->pitch = -89.0f;
        }
    }

    // Apply movement in world space relative to camera yaw
    if (glm::length(move) > 0.0001f) {
        move = glm::normalize(move);
        float yawRad = glm::radians(c->yaw);
        glm::vec3 forward(cos(yawRad), 0.0f, sin(yawRad));
        glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
        glm::vec3 dir = forward * move.z + right * move.x;
        t->position += dir * c->speed * dt;
    }
}

// ==========================================================
//  CAMERA VIEW MATRIX
// ==========================================================
inline glm::mat4 buildViewMatrix(const TransformComponent& t, const CameraComponent& c) {
    glm::vec3 front;
    front.x = cos(glm::radians(c.yaw)) * cos(glm::radians(c.pitch));
    front.y = sin(glm::radians(c.pitch));
    front.z = sin(glm::radians(c.yaw)) * cos(glm::radians(c.pitch));
    glm::vec3 dir = glm::normalize(front);
    return glm::lookAt(t.position, t.position + dir, glm::vec3(0, 1, 0));
}

// ==========================================================
//  RENDER SYSTEM (bind descriptor after update)
// ==========================================================
inline void recordCommandBuffer(
    vk::CommandBuffer cmd,
    uint32_t imageIndex,
    ECSRegistry& reg,
    ECSComponents& comps
) {
    vk::CommandBufferBeginInfo bi{};
    cmd.begin(bi);

    std::array<vk::ClearValue, 2> clears{};
    clears[0].color = vk::ClearColorValue(std::array<float, 4>{0.02f, 0.02f, 0.03f, 1.0f});
    clears[1].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);

    vk::RenderPassBeginInfo rp{};
    rp.renderPass = gRenderPass;
    rp.framebuffer = gFramebuffers[imageIndex];
    rp.renderArea.offset = vk::Offset2D(0, 0);
    rp.renderArea.extent = gSwapExtent;
    rp.clearValueCount = (uint32_t)clears.size();
    rp.pClearValues = clears.data();

    cmd.beginRenderPass(rp, vk::SubpassContents::eInline);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, gPipeline);

    vk::DeviceSize offsets[] = { 0 };
    cmd.bindVertexBuffers(0, 1, &gVertexBuffer, offsets);
    cmd.bindIndexBuffer(gIndexBuffer, 0, vk::IndexType::eUint32);

    // Update UBO from camera component
    Entity camEntity = INVALID_ENTITY;
    for (auto& kv : comps.cameras.data) {
        camEntity = kv.first;
        break;
    }

    if (camEntity != INVALID_ENTITY) {
        auto* t = comps.transforms.get(camEntity);
        auto* c = comps.cameras.get(camEntity);
        if (t && c) {
            GlobalUBO ubo{};
            ubo.view = buildViewMatrix(*t, *c);
            float aspect = (float)gSwapExtent.width / (float)gSwapExtent.height;
            ubo.proj = glm::perspective(glm::radians(c->fov), aspect, 0.1f, 500.0f);
            ubo.proj[1][1] *= -1.0f;
            ubo.lightDir = glm::vec4(glm::normalize(glm::vec3(0.3f, -1.0f, 0.2f)), 0.0f);
            ubo.lightColor = glm::vec4(1.0f);

            void* data = gDevice.mapMemory(gUBOMem[imageIndex], 0, sizeof(GlobalUBO));
            std::memcpy(data, &ubo, sizeof(GlobalUBO));
            gDevice.unmapMemory(gUBOMem[imageIndex]);
        }
    }

    struct PushConstants {
        glm::mat4 model;
        int   objectID;
        float roughnessOverride;
        float metallicOverride;
    } pc;

    int objectID = 0;
    for (auto e : reg.entities) {
        auto* t = comps.transforms.get(e);
        auto* m = comps.meshes.get(e);
        auto* matC = comps.materials.get(e);
        if (!t || !m || !matC) continue;

        // Update descriptor set binding 1 for this material
        bindMaterialForDraw(imageIndex, matC->materialID);

        // Bind descriptor set AFTER updating it (per-draw)
        cmd.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            gPipelineLayout,
            0,
            1,
            &gGlobalDescSets[imageIndex],
            0,
            nullptr
        );

        pc.model = t->toMatrix();
        pc.objectID = objectID++;
        pc.roughnessOverride = -1.0f;
        pc.metallicOverride = -1.0f;

        cmd.pushConstants(
            gPipelineLayout,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0,
            sizeof(PushConstants),
            &pc
        );

        cmd.drawIndexed(m->indexCount, 1, m->indexOffset, m->vertexOffset, 0);
    }

    cmd.endRenderPass();
    cmd.end();
}

inline void renderFrame(
    ECSRegistry& reg,
    ECSComponents& comps
) {
    gDevice.waitForFences(1, &gInFlight, VK_TRUE, UINT64_MAX);
    gDevice.resetFences(1, &gInFlight);

    uint32_t imageIndex = gDevice.acquireNextImageKHR(
        gSwapchain,
        UINT64_MAX,
        gImgAvailable,
        nullptr
    ).value;

    gCmdBuffers[imageIndex].reset();
    recordCommandBuffer(gCmdBuffers[imageIndex], imageIndex, reg, comps);

    vk::Semaphore waitSemaphores[] = { gImgAvailable };
    vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
    vk::Semaphore signalSemaphores[] = { gRenderFinished };

    vk::SubmitInfo submit{};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = waitSemaphores;
    submit.pWaitDstStageMask = waitStages;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &gCmdBuffers[imageIndex];
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = signalSemaphores;

    gGraphicsQueue.submit(1, &submit, gInFlight);

    vk::PresentInfoKHR present{};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = signalSemaphores;
    present.swapchainCount = 1;
    present.pSwapchains = &gSwapchain;
    present.pImageIndices = &imageIndex;

    gGraphicsQueue.presentKHR(present);
}
