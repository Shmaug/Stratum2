#pragma once

#include <chrono>
#include <unordered_set>

#include <vulkan/vulkan_raii.hpp>

#include "fwd.hpp"
#include "math.hpp"

struct GLFWwindow;

namespace stm2 {

class Window {
public:
	Instance& mInstance;

	Window(Instance& instance, const string& title, const vk::Extent2D& extent);
	~Window();

	inline GLFWwindow* window() const { return mWindow; }
	inline const string& title() const { return mTitle; }
	inline const vk::raii::SurfaceKHR& surface() const { return mSurface; }
	inline vk::raii::SurfaceKHR& surface() { return mSurface; }
	inline const vk::Extent2D& extent() const {return mClientExtent; }
	tuple<vk::raii::PhysicalDevice, uint32_t> findPhysicalDevice() const;
	vector<uint32_t> queueFamilies(const vk::raii::PhysicalDevice& physicalDevice) const;

	bool isOpen() const;

	void resize(const vk::Extent2D& extent) const;

	void fullscreen(const bool fs);
	inline bool fullscreen() const { return mFullscreen; }

	inline bool wantsRepaint() { return mRepaint; }

	unordered_set<string>& droppedFiles() { return mDroppedFiles; }

	void drawGui();

private:
	GLFWwindow* mWindow;
	vk::raii::SurfaceKHR mSurface;

	string mTitle;
	vk::Extent2D mClientExtent;
	vk::Rect2D mRestoreRect;

	bool mFullscreen = false;
	bool mRecreateSwapchain = false;
	bool mRepaint = false;

	unordered_set<string> mDroppedFiles;

	void createSwapchain();

	static void windowSizeCallback(GLFWwindow* window, int width, int height);
	static void dropCallback(GLFWwindow* window, int count, const char** paths);
};

}