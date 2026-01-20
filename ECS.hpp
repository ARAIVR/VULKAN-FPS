#pragma once
#include <cstdint>
#include <vector>
#include <array>
#include <functional>

using Entity = uint32_t;
static const Entity INVALID_ENTITY = 0xffffffffu;

struct ECSRegistry {
    Entity nextEntity = 1;
    std::vector<Entity> entities;
};

inline Entity ecsCreateEntity(ECSRegistry& reg) {
    Entity e = reg.nextEntity++;
    reg.entities.push_back(e);
    return e;
}
