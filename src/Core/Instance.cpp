#include "Instance.hpp"

#include <imgui/imgui.h>
#include <GLFW/glfw3.h>

namespace stm2 {

#define RESET       "\033[0m"
#define BLACK       "\033[30m"
#define RED         "\033[31m"
#define GREEN       "\033[32m"
#define YELLOW      "\033[33m"
#define BLUE        "\033[34m"
#define MAGENTA     "\033[35m"
#define CYAN        "\033[36m"
#define WHITE       "\033[37m"
#define BOLDBLACK   "\033[1m\033[30m"
#define BOLDRED     "\033[1m\033[31m"
#define BOLDGREEN   "\033[1m\033[32m"
#define BOLDYELLOW  "\033[1m\033[33m"
#define BOLDBLUE    "\033[1m\033[34m"
#define BOLDMAGENTA "\033[1m\033[35m"
#define BOLDCYAN    "\033[1m\033[36m"
#define BOLDWHITE   "\033[1m\033[37m"

bool Instance::sDisableDebugCallback = false;

// Debug messenger functions
VKAPI_ATTR vk::Bool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
	if (Instance::sDisableDebugCallback) return VK_FALSE;

	string msgstr = pCallbackData->pMessage;
	
	{ // skip past ' ... | MessageID = ... | '
		const size_t offset = msgstr.find_last_of("|");
		if (offset != string::npos)
			msgstr = msgstr.substr(offset + 2); // skip '| '
	}

	string specstr;
	{ // Separately print 'The Vulkan spec states: '
		const size_t offset = msgstr.find("The Vulkan spec states:");
		if (offset != string::npos) {
			specstr = msgstr.substr(offset);
			msgstr = msgstr.substr(0, offset);
		}
	}

	auto print_fn = [&](ostream& stream) {
		stream << pCallbackData->pMessageIdName << ": " << endl;
		stream << "\t" << BOLDWHITE << msgstr << RESET << endl;
		if (!specstr.empty())
			stream << "\t" << specstr << endl;
	};

	if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
		print_fn(cerr << BOLDRED);
	else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
		print_fn(cerr << BOLDYELLOW);
	else
		print_fn(cout << BOLDCYAN);

	return VK_FALSE;
}

Instance::Instance(const vector<string>& args) : mInstance(nullptr), mDebugMessenger(nullptr), mCommandLine(args) {
	mContext = vk::raii::Context();
	
	for (const string& arg : mCommandLine | views::drop(1)) {
		size_t o = 0;
		if (arg.starts_with("--"))
			o = 2;
		else if (arg.starts_with("-") || arg.starts_with("/"))
			o = 1;

		if (size_t sep = arg.find('='); sep != string::npos)
			mOptions.emplace(arg.substr(o,sep-o), arg.substr(sep+1));
		else if (size_t sep = arg.find(':'); sep != string::npos)
			mOptions.emplace(arg.substr(o,sep-o), arg.substr(sep+1));
		else
			mOptions.emplace(arg.substr(o), "");
	}

	const bool debugMessenger = findArgument("debugMessenger").has_value();

	// Parse validation layers

	for (const auto& layer : findArguments("validationLayer")) mValidationLayers.emplace(layer);
	if (debugMessenger) mValidationLayers.emplace("VK_LAYER_KHRONOS_validation");

	// Parse instance extensions

	unordered_set<string> instanceExtensions = { VK_KHR_SURFACE_EXTENSION_NAME };
#ifdef _WIN32
	instanceExtensions.emplace(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#endif
#ifdef __linux
	instanceExtensions.emplace(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif
	for (const auto& ext : findArguments("instanceExtension"))
		instanceExtensions.emplace(ext);

	uint32_t count;
	const char** exts = glfwGetRequiredInstanceExtensions(&count);
	for (uint32_t i = 0; i < count; i++)
		instanceExtensions.emplace(exts[i]);

	// Remove unsupported layers

	if (mValidationLayers.size()) {
		unordered_set<string> available;
		for (const auto& layer : mContext.enumerateInstanceLayerProperties()) available.emplace(layer.layerName.data());
		for (auto it = mValidationLayers.begin(); it != mValidationLayers.end();)
			if (available.find(*it) == available.end()) {
				cerr << "Warning: Removing unsupported validation layer: " << it->c_str() << endl;
				it = mValidationLayers.erase(it);
			} else
				it++;
	}

	// Add debug extensions if needed

	if (mValidationLayers.contains("VK_LAYER_KHRONOS_validation")) {
		instanceExtensions.emplace(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
		instanceExtensions.emplace(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		instanceExtensions.emplace(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
	}

	vector<const char*> instanceExts;
	for (const string& s : instanceExtensions) instanceExts.push_back(s.c_str());

	vector<const char*> validationLayers;
	for (const string& v : mValidationLayers) validationLayers.push_back(v.c_str());

	// create instance

	vk::ApplicationInfo appInfo = {};
	appInfo.pApplicationName = "stratum2";
	appInfo.applicationVersion = VK_MAKE_VERSION(0, 0, 0);
	appInfo.pEngineName = "stratum2";
	appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
	appInfo.apiVersion = mVulkanApiVersion = mContext.enumerateInstanceVersion();
	mInstance = vk::raii::Instance(mContext, vk::InstanceCreateInfo({}, &appInfo, validationLayers, instanceExts));

	cout << "Vulkan " << VK_VERSION_MAJOR(mVulkanApiVersion) << "." << VK_VERSION_MINOR(mVulkanApiVersion) << "." << VK_VERSION_PATCH(mVulkanApiVersion) << endl;

	if (debugMessenger) {
		cout << "Creating debug messenger" << endl;
		mDebugMessenger = vk::raii::DebugUtilsMessengerEXT(mInstance, vk::DebugUtilsMessengerCreateInfoEXT({},
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
			vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
			DebugCallback));
	}
}

void Instance::drawGui() {
	ImGui::Text("Vulkan %u.%u.%u",
		VK_API_VERSION_MAJOR(mVulkanApiVersion),
		VK_API_VERSION_MINOR(mVulkanApiVersion),
		VK_API_VERSION_PATCH(mVulkanApiVersion));
}

}