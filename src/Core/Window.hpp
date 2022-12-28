#pragma once

#include <chrono>
#include <unordered_set>

#include <vulkan/vulkan_raii.hpp>

#include "fwd.hpp"
#include "hlslmath.hpp"

struct GLFWwindow;

namespace stm2 {

// GLFW_KEY_***
typedef uint32_t KeyCode;

class MouseKeyboardState {
public:
	inline float2& cursor_pos() { return mCursorPos; }
	inline const unordered_set<KeyCode>& buttons() const { return mButtons; }
	inline const float2& cursorPos() const { return mCursorPos; }
	inline const string& inputCharacters() const { return mInputCharacters; };
	inline bool held(const KeyCode key) const { return mButtons.count(key); }
	inline const vector<string>& files() const { return mInputFiles; }

private:
	friend class Window;

	float2 mCursorPos;
	unordered_set<KeyCode> mButtons;
	string mInputCharacters;
	vector<string> mInputFiles;

	inline void clear() {
		mInputCharacters.clear();
		mInputFiles.clear();
	}
};

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

	inline const MouseKeyboardState& inputState() const { return mInputState; }
	inline const MouseKeyboardState& inputStatePrev() const { return mInputStatePrev; }
	inline bool held    (const KeyCode& key) const { return  mInputState.held(key); }
	inline bool pressed (const KeyCode& key) const { return  mInputState.held(key) && !mInputStatePrev.held(key); }
	inline bool released(const KeyCode& key) const { return !mInputState.held(key) &&  mInputStatePrev.held(key); }

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

	MouseKeyboardState mInputState, mInputStatePrev;

	void createSwapchain();

	static void windowSizeCallback(GLFWwindow* window, int width, int height);
	static void dropCallback(GLFWwindow* window, int count, const char** paths);
};

}