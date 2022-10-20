#pragma once

#include <unordered_set>
#include <ranges>

#include <Utils/utils.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace tinyvkpt {

class Instance {
public:
	static bool sDisableDebugCallback;

	Instance(const vector<string>& args);

	inline vk::raii::Instance& operator*() { return mInstance; }
	inline vk::raii::Instance* operator->() { return &mInstance; }
	inline const vk::raii::Instance& operator*() const { return mInstance; }
	inline const vk::raii::Instance* operator->() const { return &mInstance; }
	inline vk::raii::Context& context() { return mContext; }
	inline const vk::raii::Context& context() const { return mContext; }

	inline uint32_t vulkanVersion() const { return mVulkanApiVersion; }

	inline optional<string> findArgument(const string& name) const {
		auto[first,last] = mOptions.equal_range(name);
		if (first == last) return nullopt;
		return first->second;
	}
	inline auto findArguments(const string& name) const {
		auto[first,last] = mOptions.equal_range(name);
		return ranges::subrange(first,last) | views::values;
	}
	inline const unordered_set<string>& validationLayers() const { return mValidationLayers; }

private:
	vk::raii::Context mContext;
	vk::raii::Instance mInstance;

	unordered_set<string> mValidationLayers;

	vector<string> mCommandLine;
	unordered_multimap<string, string> mOptions;
	uint32_t mVulkanApiVersion;

	vk::raii::DebugUtilsMessengerEXT mDebugMessenger;
};

}