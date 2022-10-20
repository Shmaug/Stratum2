#pragma once

#include "Device.hpp"

namespace tinyvkpt {

class ComputePipeline : public Device::Resource {
public:
	Buffer(Shader& shader);

	inline vk::raii::Buffer& operator*() { return mPipeline; }
	inline vk::raii::Buffer* operator->() { return &mPipeline; }
	inline const vk::raii::Buffer& operator*() const { return mPipeline; }
	inline const vk::raii::Buffer* operator->() const { return &mPipeline; }

private:
	vk::raii::Pipeline mPipeline;
};

}