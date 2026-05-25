#pragma once
#define VK_NO_PROTOTYPES
#include <cstdint>    
#include <string>   
#include <vulkan/vulkan.h> 
#include <glm/ext/vector_int2.hpp>

struct CubemapRenderTarget;

struct ComputePushConstants {
    glm::ivec2 uFaceSize;
    int uNumTriangles;
};
