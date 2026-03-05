#include <Rendering/PMVC/CpuComputeStrategy.h>
#include <Rendering/PMVC/CubemapRenderInstance.h>
#include <Rendering/Utils/VulkanUtils.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace
{
constexpr float kDepthEpsilon = 0.999999f;

float AreaElement(float x, float y)
{
    return std::atan2(x * y, std::sqrt(x * x + y * y + 1.0f));
}
}

uint32_t CpuComputeStrategy::RequiredRenderTargetCount() const
{
    return _targetCount;
}

void CpuComputeStrategy::Initialize()
{
    _targetCount = 1;

    _lambdaResults.resize(_deformableMesh._vertices.rows(), _cageMesh._vertices.rows());
    _lambdaResults.setZero();
    _wsumResults.assign(_deformableMesh._vertices.rows(), 0.0f);

    _vertexList.clear();
    _vertexList.reserve(_cageMesh._faces.size());
    for (const auto faceIndex : _cageMesh._faces)
    {
        _vertexList.push_back(static_cast<uint32_t>(faceIndex));
    }

    _depthFormat = _device->FindDepthFormat();
    _depthBytesPerTexel = (_depthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT) ? 8u : 4u;

    _solidAngles.resize(6ull * _faceSize * _faceSize);
    for (uint32_t face = 0; face < 6; ++face)
    {
        for (uint32_t y = 0; y < _faceSize; ++y)
        {
            for (uint32_t x = 0; x < _faceSize; ++x)
            {
                const size_t idx = size_t(face) * _faceSize * _faceSize + size_t(y) * _faceSize + x;
                _solidAngles[idx] = ComputeSolidAngle(x, y);
            }
        }
    }

    VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.queueFamilyIndex = _transferQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(_device, &poolInfo, nullptr, &_commandPool));

    VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.commandPool = _commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(_device, &allocInfo, &_commandBuffer));

    vkGetDeviceQueue(_device, _transferQueueFamily, 0, &_queue);

    _slots.resize(_targetCount);

    const VkDeviceSize colorSize = VkDeviceSize(_faceSize) * _faceSize * 6 * 4 * sizeof(float);
    const VkDeviceSize depthSize = VkDeviceSize(_faceSize) * _faceSize * 6 * _depthBytesPerTexel;

    for (SlotReadback& slot : _slots)
    {
        VkBufferCreateInfo colorBufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        colorBufferInfo.size = colorSize;
        colorBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        colorBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(_device, &colorBufferInfo, nullptr, &slot.colorBuffer));

        VkMemoryRequirements memReq{};
        vkGetBufferMemoryRequirements(_device, slot.colorBuffer, &memReq);

        VkMemoryAllocateInfo alloc{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        alloc.allocationSize = memReq.size;
        alloc.memoryTypeIndex = FindMemoryType(
            memReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(_device, &alloc, nullptr, &slot.colorMemory));
        VK_CHECK(vkBindBufferMemory(_device, slot.colorBuffer, slot.colorMemory, 0));

        VkBufferCreateInfo depthBufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        depthBufferInfo.size = depthSize;
        depthBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        depthBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(_device, &depthBufferInfo, nullptr, &slot.depthBuffer));

        vkGetBufferMemoryRequirements(_device, slot.depthBuffer, &memReq);
        alloc.allocationSize = memReq.size;
        alloc.memoryTypeIndex = FindMemoryType(
            memReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(_device, &alloc, nullptr, &slot.depthMemory));
        VK_CHECK(vkBindBufferMemory(_device, slot.depthBuffer, slot.depthMemory, 0));
    }
}

void CpuComputeStrategy::WaitForTargetReuse(uint32_t,
    VkSemaphore,
    uint64_t)
{
}

void CpuComputeStrategy::DispatchAfterRender(
    uint32_t deformableIndex,
    uint32_t slot,
    VkSemaphore,
    const CubemapRenderTarget& target)
{
    const SlotReadback& readbackSlot = _slots[slot];
    CopyImagesToStaging(target, readbackSlot);
    ComputeOnCpu(deformableIndex, readbackSlot);
}

uint64_t CpuComputeStrategy::GetSlotCompletionValue(uint32_t) const
{
    return 0;
}

void CpuComputeStrategy::Readback(uint32_t,
    uint32_t,
    const std::string&)
{
}

void CpuComputeStrategy::WaitAll(VkSemaphore)
{
}

Eigen::MatrixXd CpuComputeStrategy::Readback()
{
    for (int i = 0; i < _lambdaResults.rows(); ++i)
    {
        const float wsum = _wsumResults[i];
        if (wsum > 0.0f)
        {
            _lambdaResults.row(i) /= wsum;
        }
    }

    return _lambdaResults;
}

uint32_t CpuComputeStrategy::FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const
{
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(_device->GetPhysicalDeviceHandle(), &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
    {
        if ((typeBits & (1u << i)) && ((memProps.memoryTypes[i].propertyFlags & props) == props))
        {
            return i;
        }
    }

    throw std::runtime_error("CpuComputeStrategy: no suitable staging memory type");
}

float CpuComputeStrategy::ComputeSolidAngle(uint32_t texelX, uint32_t texelY) const
{
    const float size = static_cast<float>(_faceSize);
    const float u = (2.0f * (static_cast<float>(texelX) + 0.5f) / size) - 1.0f;
    const float v = (2.0f * (static_cast<float>(texelY) + 0.5f) / size) - 1.0f;
    const float invResolution = 1.0f / size;

    const float x0 = u - invResolution;
    const float y0 = v - invResolution;
    const float x1 = u + invResolution;
    const float y1 = v + invResolution;

    return AreaElement(x0, y0) - AreaElement(x0, y1)
        - AreaElement(x1, y0) + AreaElement(x1, y1);
}

float CpuComputeStrategy::DecodeDepthSample(const uint8_t* texel) const
{
    switch (_depthFormat)
    {
    case VK_FORMAT_D32_SFLOAT:
        return *reinterpret_cast<const float*>(texel);
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return *reinterpret_cast<const float*>(texel);
    case VK_FORMAT_D24_UNORM_S8_UINT:
    {
        const uint32_t packed = *reinterpret_cast<const uint32_t*>(texel);
        const uint32_t depth24 = packed & 0x00FFFFFFu;
        return static_cast<float>(depth24) / 16777215.0f;
    }
    default:
        return 1.0f;
    }
}

void CpuComputeStrategy::CopyImagesToStaging(const CubemapRenderTarget& target, const SlotReadback& slot)
{
    const VkDeviceSize colorFaceSize = VkDeviceSize(_faceSize) * _faceSize * 4 * sizeof(float);
    const VkDeviceSize depthFaceSize = VkDeviceSize(_faceSize) * _faceSize * _depthBytesPerTexel;

    VK_CHECK(vkResetCommandBuffer(_commandBuffer, 0));

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(_commandBuffer, &beginInfo));

    VkImageMemoryBarrier2 barriers[2]{};
    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
    barriers[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barriers[0].image = target.cubemapImage;
    barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barriers[0].subresourceRange.baseMipLevel = 0;
    barriers[0].subresourceRange.levelCount = 1;
    barriers[0].subresourceRange.baseArrayLayer = 0;
    barriers[0].subresourceRange.layerCount = 6;

    barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    barriers[1].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    barriers[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barriers[1].image = target.depthImage;
    barriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barriers[1].subresourceRange.baseMipLevel = 0;
    barriers[1].subresourceRange.levelCount = 1;
    barriers[1].subresourceRange.baseArrayLayer = 0;
    barriers[1].subresourceRange.layerCount = 6;

    if (VulkanUtils::FormatHasStencilComponent(_depthFormat))
    {
        barriers[1].subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }

    VkDependencyInfo barrierInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    barrierInfo.imageMemoryBarrierCount = 2;
    barrierInfo.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(_commandBuffer, &barrierInfo);

    std::array<VkBufferImageCopy, 6> colorRegions{};
    std::array<VkBufferImageCopy, 6> depthRegions{};
    for (uint32_t face = 0; face < 6; ++face)
    {
        colorRegions[face].bufferOffset = colorFaceSize * face;
        colorRegions[face].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        colorRegions[face].imageSubresource.mipLevel = 0;
        colorRegions[face].imageSubresource.baseArrayLayer = face;
        colorRegions[face].imageSubresource.layerCount = 1;
        colorRegions[face].imageExtent = { _faceSize, _faceSize, 1 };

        depthRegions[face].bufferOffset = depthFaceSize * face;
        depthRegions[face].imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        depthRegions[face].imageSubresource.mipLevel = 0;
        depthRegions[face].imageSubresource.baseArrayLayer = face;
        depthRegions[face].imageSubresource.layerCount = 1;
        depthRegions[face].imageExtent = { _faceSize, _faceSize, 1 };
    }

    vkCmdCopyImageToBuffer(
        _commandBuffer,
        target.cubemapImage,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        slot.colorBuffer,
        static_cast<uint32_t>(colorRegions.size()),
        colorRegions.data());

    vkCmdCopyImageToBuffer(
        _commandBuffer,
        target.depthImage,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        slot.depthBuffer,
        static_cast<uint32_t>(depthRegions.size()),
        depthRegions.data());

    barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barriers[1].srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    vkCmdPipelineBarrier2(_commandBuffer, &barrierInfo);

    VK_CHECK(vkEndCommandBuffer(_commandBuffer));

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &_commandBuffer;
    VK_CHECK(vkQueueSubmit(_queue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(_queue));
}

void CpuComputeStrategy::ComputeOnCpu(uint32_t deformableIndex, const SlotReadback& slot)
{
    void* colorRaw = nullptr;
    void* depthRaw = nullptr;

    const VkDeviceSize colorSize = VkDeviceSize(_faceSize) * _faceSize * 6 * 4 * sizeof(float);
    const VkDeviceSize depthSize = VkDeviceSize(_faceSize) * _faceSize * 6 * _depthBytesPerTexel;

    VK_CHECK(vkMapMemory(_device, slot.colorMemory, 0, colorSize, 0, &colorRaw));
    VK_CHECK(vkMapMemory(_device, slot.depthMemory, 0, depthSize, 0, &depthRaw));

    const float* colors = reinterpret_cast<const float*>(colorRaw);
    const uint8_t* depthBytes = reinterpret_cast<const uint8_t*>(depthRaw);

    std::vector<float> lambda(_cageMesh._vertices.rows(), 0.0f);
    float wsum = 0.0f;

    const uint32_t numTriangles = static_cast<uint32_t>(_cageMesh._faces.size() / 3);
    const size_t texelCountPerFace = static_cast<size_t>(_faceSize) * _faceSize;

    for (uint32_t face = 0; face < 6; ++face)
    {
        for (uint32_t y = 0; y < _faceSize; ++y)
        {
            for (uint32_t x = 0; x < _faceSize; ++x)
            {
                const size_t texelIdx = size_t(face) * texelCountPerFace + size_t(y) * _faceSize + x;
                const size_t colorBase = texelIdx * 4;

                const float b0 = colors[colorBase + 0];
                const float b1 = colors[colorBase + 1];
                const float b2 = colors[colorBase + 2];

                const uint32_t tri = static_cast<uint32_t>(colors[colorBase + 3] * float(numTriangles) + 0.5f);
                if (tri >= numTriangles)
                {
                    continue;
                }

                const float depth = DecodeDepthSample(depthBytes + texelIdx * _depthBytesPerTexel);
                if (depth >= kDepthEpsilon)
                {
                    continue;
                }

                const float depthWeight = 1.0f - depth;
                const float solidAngle = _solidAngles[texelIdx];
                const float w = solidAngle * depthWeight;
                if (w <= 0.0f)
                {
                    continue;
                }

                const uint32_t i0 = _vertexList[tri * 3 + 0];
                const uint32_t i1 = _vertexList[tri * 3 + 1];
                const uint32_t i2 = _vertexList[tri * 3 + 2];

                lambda[i0] += b0 * w;
                lambda[i1] += b1 * w;
                lambda[i2] += b2 * w;

                wsum += w;
            }
        }
    }

    for (size_t c = 0; c < lambda.size(); ++c)
    {
        _lambdaResults(deformableIndex, static_cast<Eigen::Index>(c)) = lambda[c];
    }
    _wsumResults[deformableIndex] = wsum;

    vkUnmapMemory(_device, slot.colorMemory);
    vkUnmapMemory(_device, slot.depthMemory);
}
