#pragma once

#include <chrono>
#include <unordered_set>

#include <vulkan/vulkan_raii.hpp>

#include <Utils/math.hpp>

struct GLFWwindow;

namespace tinyvkpt {

// GLFW_KEY_***
typedef uint32_t KeyCode;

class MouseKeyboardState {
public:
	inline float2& cursor_pos() { return mCursorPos; }
	inline const unordered_set<KeyCode>& buttons() const { return mButtons; }
	inline const float2& cursorPos() const { return mCursorPos; }
	inline const float2& cursorDelta() const { return mCursorDelta; }
	inline float scrollDelta() const { return mScrollDelta; }
	inline const string& input_characters() const { return mInputCharacters; };
	inline bool held(KeyCode key) const { return mButtons.count(key); }
	inline const vector<string>& files() const { return mInputFiles; }

private:
	friend class Instance;
	friend class Window;

	float2 mCursorPos;
	float2 mCursorDelta;
	float mScrollDelta;
	unordered_set<KeyCode> mButtons;
	string mInputCharacters;
	vector<string> mInputFiles;

	inline void clear() {
		mCursorDelta = float2::Zero();
		mScrollDelta = 0;
		mInputCharacters.clear();
		mInputFiles.clear();
	}
	inline void addCursorDelta(const float2& delta) { mCursorDelta += delta; }
	inline void addScrollDelta(float delta) { mScrollDelta += delta; }
	inline void addInputCharacter(char c) { mInputCharacters.push_back(c); };
	inline void setButton(KeyCode key) { mButtons.emplace(key); }
	inline void unsetButton(KeyCode key) { mButtons.erase(key); }
};

class Window {
public:
	Instance& mInstance;

	Window(Instance& instance, const string& title, const vk::Extent2D& extent);
	~Window();

	inline GLFWwindow* window() const { return mWindow; }
	inline const string& title() const { return mTitle; }
	inline const vk::Rect2D& clientRect() const { return mClientRect; };
	inline const vk::raii::SurfaceKHR& surface() const { return mSurface; }
	inline vk::raii::SurfaceKHR& surface() { return mSurface; }
	inline vk::SurfaceFormatKHR surfaceFormat() const { return mSurfaceFormat; }
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

private:
	GLFWwindow* mWindow;
	vk::raii::SurfaceKHR mSurface;
	vk::SurfaceFormatKHR mSurfaceFormat;

	bool mFullscreen = false;
	bool mRecreateSwapchain = false;
	bool mRepaint = false;
	vk::Rect2D mRestoreRect;
	vk::Rect2D mClientRect;
	string mTitle;

	MouseKeyboardState mInputState, mInputStatePrev;

	void createSwapchain();

	static void windowSizeCallback(GLFWwindow* window, int width, int height);
	static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void characterCallback(GLFWwindow* window, unsigned int codepoint);
	static void cursor_positionCallback(GLFWwindow* window, double x, double y);
	static void dropCallback(GLFWwindow* window, int count, const char** paths);
};

}