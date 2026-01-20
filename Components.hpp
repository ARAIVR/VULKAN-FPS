#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>
#include <unordered_map>
#include <vulkan/vulkan.hpp>

// Forward
struct MeshGPUHandle;

// ---------------- Transform ----------------
struct TransformComponent {
    glm::vec3 position{ 0.0f };
    glm::quat rotation{ 1.0f,0.0f,0.0f,0.0f };
    glm::vec3 scale{ 1.0f };

    glm::mat4 toMatrix() const {
        glm::mat4 T = glm::translate(glm::mat4(1.0f), position);
        glm::mat4 R = glm::mat4_cast(rotation);
        glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
        return T * R * S;
    }
};

// ---------------- Camera ----------------
struct CameraComponent {
    float pitch = -10.0f;
    float yaw = 90.0f;
    float fov = 60.0f;
    float speed = 5.0f;
    float sensitivity = 0.1f;
};

// ---------------- Mesh ----------------
struct MeshComponent {
    uint32_t vertexOffset = 0;
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;
    uint32_t materialID = 0;
};

// ---------------- Material ----------------
struct MaterialComponent {
    uint32_t materialID = 0;
};

// ---------------- Input ----------------
struct InputComponent {
    // nothing yet; marker for controlled entity
};

// ---------------- Component storage ----------------
template<typename T>
struct ComponentPool {
    std::unordered_map<Entity, T> data;

    bool has(Entity e) const {
        return data.find(e) != data.end();
    }
    T* get(Entity e) {
        auto it = data.find(e);
        if (it == data.end()) return nullptr;
        return &it->second;
    }
    const T* get(Entity e) const {
        auto it = data.find(e);
        if (it == data.end()) return nullptr;
        return &it->second;
    }
    void add(Entity e, const T& c) {
        data[e] = c;
    }
};

struct ECSComponents {
    ComponentPool<TransformComponent> transforms;
    ComponentPool<CameraComponent>    cameras;
    ComponentPool<MeshComponent>      meshes;
    ComponentPool<MaterialComponent>  materials;
    ComponentPool<InputComponent>     inputs;
};
