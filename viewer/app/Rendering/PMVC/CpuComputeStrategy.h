#pragma once

#include <Rendering/PMVC/IComputeStrategy.h>
#include <Rendering/Core/Device.h>
#include <Mesh/GeometryUtils.h>
#include <Eigen/Core>
#include <vector>

class CpuComputeStrategy final : public ICubemapComputeStrategy
{
public:
    CpuComputeStrategy(
        RenderResourceRef<Device> device,
        uint32_t transferQueueFamily,
        uint32_t faceSize,
        VkFormat format,
        EigenMesh& cageMesh,
        EigenMesh& deformableMesh)
        : _device(device)
        , _transferQueueFamily(transferQueueFamily)
        , _faceSize(faceSize)
        , _format(format)
        , _cageMesh(cageMesh)
        , _deformableMesh(deformableMesh)
    {
    }

    uint32_t RequiredRenderTargetCount() const override;

    void Initialize() override;

    void WaitForTargetReuse(uint32_t targetIndex,
        VkSemaphore timeline,
        uint64_t slotDoneValue);

    void DispatchAfterRender(
        uint32_t deformableIndex,
        uint32_t slot,
        VkSemaphore timeline,
		const CubemapRenderTarget& target);

    uint64_t GetSlotCompletionValue(uint32_t targetIndex) const;

    void Readback(uint32_t cubemapIdx,
        uint32_t targetIndex,
        const std::string& filename);

    void WaitAll(VkSemaphore timeline);

    Eigen::MatrixXd Readback();

    void ConsumeSlot(
        uint32_t deformableIndex,
        uint32_t slot,
        VkSemaphore timeline) { }

private:
    struct SlotReadback
    {
        VkBuffer colorBuffer = VK_NULL_HANDLE;
        VkDeviceMemory colorMemory = VK_NULL_HANDLE;
        VkBuffer depthBuffer = VK_NULL_HANDLE;
        VkDeviceMemory depthMemory = VK_NULL_HANDLE;
    };

    uint32_t FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const;
    float ComputeSolidAngle(uint32_t texelX, uint32_t texelY) const;
    float DecodeDepthSample(const uint8_t* texel) const;
    void CopyImagesToStaging(const CubemapRenderTarget& target, const SlotReadback& slot);
    void ComputeOnCpu(uint32_t deformableIndex, const SlotReadback& slot);

    RenderResourceRef<Device> _device;
    uint32_t _transferQueueFamily = 0;
    uint32_t _faceSize = 0;
    VkFormat _format = VK_FORMAT_UNDEFINED;
    VkFormat _depthFormat = VK_FORMAT_UNDEFINED;
    uint32_t _depthBytesPerTexel = 4;

    VkCommandPool _commandPool = VK_NULL_HANDLE;
    VkCommandBuffer _commandBuffer = VK_NULL_HANDLE;
    VkQueue _queue = VK_NULL_HANDLE;

    std::vector<SlotReadback> _slots;
    std::vector<float> _solidAngles;
    std::vector<uint32_t> _vertexList;

    Eigen::MatrixXd _lambdaResults;
    std::vector<float> _wsumResults;

    EigenMesh& _cageMesh;
    EigenMesh& _deformableMesh;

    uint32_t _targetCount = 0;
};
