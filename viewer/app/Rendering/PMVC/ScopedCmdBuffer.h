#pragma once

#include <Rendering/Core/Device.h>
#include <vulkan/vulkan.h>

class ScopedCmdBuffer
{
public:
    ScopedCmdBuffer(
        RenderResourceRef<Device> device,
        VkCommandPool pool)
        : _device(device), _pool(pool)
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = _pool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VK_CHECK(vkAllocateCommandBuffers(_device, &allocInfo, &_cmd));

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VK_CHECK(vkBeginCommandBuffer(_cmd, &beginInfo));
    }

    VkCommandBuffer Get() const { return _cmd; }

    void SubmitAndWait(VkQueue queue)
    {
        VK_CHECK(vkEndCommandBuffer(_cmd));

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &_cmd;

        VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue));

        _submitted = true;
    }

    void Submit(VkQueue queue, VkFence fence = VK_NULL_HANDLE)
    {
        VK_CHECK(vkEndCommandBuffer(_cmd));

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &_cmd;

        VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));

        _submitted = true;
    }

    void SubmitTimeline(
        VkQueue queue,
        VkSemaphore waitSemaphore,
        uint64_t waitValue,
        VkPipelineStageFlags2 waitStageMask,
        VkSemaphore signalSemaphore,
        uint64_t signalValue,
        VkPipelineStageFlags2 signalStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
    {
        VK_CHECK(vkEndCommandBuffer(_cmd));

        VkCommandBufferSubmitInfo cmdInfo{};
        cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        cmdInfo.commandBuffer = _cmd;

        VkSemaphoreSubmitInfo waitInfo{};
        waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        waitInfo.semaphore = waitSemaphore;
        waitInfo.value = waitValue;
        waitInfo.stageMask = waitStageMask;
        waitInfo.deviceIndex = 0;

        VkSemaphoreSubmitInfo signalInfo{};
        signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        signalInfo.semaphore = signalSemaphore;
        signalInfo.value = signalValue;
        signalInfo.stageMask = signalStageMask;
        signalInfo.deviceIndex = 0;

        VkSubmitInfo2 submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        submitInfo.waitSemaphoreInfoCount = 1;
        submitInfo.pWaitSemaphoreInfos = &waitInfo;
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdInfo;
        submitInfo.signalSemaphoreInfoCount = 1;
        submitInfo.pSignalSemaphoreInfos = &signalInfo;

        VK_CHECK(vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE));

        _submitted = true;
    }

    ~ScopedCmdBuffer()
    {
        if (_cmd != VK_NULL_HANDLE)
            vkFreeCommandBuffers(_device, _pool, 1, &_cmd);
    }

private:
    RenderResourceRef<Device> _device;
    VkCommandPool _pool;
    VkCommandBuffer _cmd = VK_NULL_HANDLE;
    bool _submitted = false;
};