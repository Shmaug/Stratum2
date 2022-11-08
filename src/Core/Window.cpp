#include "Window.hpp"
#include "Instance.hpp"

#include <stdexcept>

// vulkan.h needed by glfw.h
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

namespace tinyvkpt {

void Window::windowSizeCallback(GLFWwindow* window, int width, int height) {
	Window* w = (Window*)glfwGetWindowUserPointer(window);
	w->mClientExtent = vk::Extent2D{ (uint32_t)width, (uint32_t)height };
}
void Window::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	Window* w = (Window*)glfwGetWindowUserPointer(window);
	if (action == GLFW_PRESS || action == GLFW_REPEAT)
		w->mInputState.mButtons.emplace(key);
	else if (action == GLFW_RELEASE)
		w->mInputState.mButtons.erase(key);
}
void Window::characterCallback(GLFWwindow* window, unsigned int codepoint) {
	Window* w = (Window*)glfwGetWindowUserPointer(window);
	w->mInputState.mInputCharacters.push_back(codepoint);
}
void Window::cursorPositionCallback(GLFWwindow* window, double x, double y) {
	Window* w = (Window*)glfwGetWindowUserPointer(window);
	w->mInputState.mCursorPos = float2((float)x, (float)y);
}
void Window::dropCallback(GLFWwindow* window, int count, const char** paths) {
	Window* w = (Window*)glfwGetWindowUserPointer(window);
	for (int i = 0; i < count; i++)
		w->mInputState.mInputFiles.emplace_back(paths[i]);
}

void errorCallback(int code, const char* msg) {
	cerr << msg;
	throw runtime_error(msg);
}

Window::Window(Instance& instance, const string& title, const vk::Extent2D& extent) : mInstance(instance), mTitle(title), mSurface(nullptr) {
	if (glfwInit() != GLFW_TRUE) {
		cerr << "Error: Failed to initialize GLFW" << endl;
		throw runtime_error("Failed to initialized GLFW");
	}

	glfwSetErrorCallback(errorCallback);

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_REFRESH_RATE , GLFW_DONT_CARE);
	mWindow = glfwCreateWindow(extent.width, extent.height, mTitle.c_str(), NULL, NULL);
	glfwSetWindowUserPointer(mWindow, this);

	VkSurfaceKHR surface;
	if (glfwCreateWindowSurface(**mInstance, mWindow, NULL, &surface))
		throw runtime_error("Failed to create surface");
	mSurface = vk::raii::SurfaceKHR(*instance, surface);

	glfwSetFramebufferSizeCallback(mWindow, Window::windowSizeCallback);
	glfwSetKeyCallback            (mWindow, Window::keyCallback);
	glfwSetCharCallback           (mWindow, Window::characterCallback);
	glfwSetCursorPosCallback      (mWindow, Window::cursorPositionCallback);
	glfwSetDropCallback           (mWindow, Window::dropCallback);
}
Window::~Window() {
	glfwDestroyWindow(mWindow);
}

tuple<vk::raii::PhysicalDevice, uint32_t> Window::findPhysicalDevice() const {
	vk::raii::PhysicalDevices physicalDevices(*mInstance);
	for (const vk::raii::PhysicalDevice physicalDevice : physicalDevices) {
		const auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
		for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
			if (glfwGetPhysicalDevicePresentationSupport(**mInstance, *physicalDevice, i))
				return tie(physicalDevice, i);
	}
	return tuple<vk::raii::PhysicalDevice, uint32_t>( vk::raii::PhysicalDevice(nullptr), uint32_t(-1) );
}

vector<uint32_t> Window::queueFamilies(const vk::raii::PhysicalDevice& physicalDevice) const {
	vector<uint32_t> families;
	const auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
	for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
		if (glfwGetPhysicalDevicePresentationSupport(**mInstance, *physicalDevice, i))
			families.emplace_back(i);
	return families;
}

bool Window::isOpen() const {
	return !glfwWindowShouldClose(mWindow);
}

void Window::resize(const vk::Extent2D& extent) const {
	glfwSetWindowSize(mWindow, extent.width, extent.height);
}

void Window::fullscreen(const bool fs) {
	mFullscreen = fs;
	if (mFullscreen) {
		glfwGetWindowPos(mWindow, &mRestoreRect.offset.x, &mRestoreRect.offset.y);
		glfwGetWindowSize(mWindow, reinterpret_cast<int*>(&mRestoreRect.extent.width), reinterpret_cast<int*>(&mRestoreRect.extent.height));
		GLFWmonitor* monitor = glfwGetWindowMonitor(mWindow);
		const GLFWvidmode* mode = glfwGetVideoMode(monitor);
		glfwWindowHint(GLFW_RED_BITS, mode->redBits);
		glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
		glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
		glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
		glfwSetWindowMonitor(mWindow, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
	} else {
		glfwSetWindowMonitor(mWindow, nullptr, mRestoreRect.offset.x, mRestoreRect.offset.y, mRestoreRect.extent.width, mRestoreRect.extent.height, 0);
	}
}

}