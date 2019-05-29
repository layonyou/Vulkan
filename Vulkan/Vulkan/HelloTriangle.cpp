#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <vector>
#include <optional>
#include <set>
#include <algorithm>

//Constants for window size
const int WIDTH = 1280;
const int HEIGHT = 720;
//Constant vector for LunarG sstandard validation layers
const std::vector<const char*> validationLayers = {
   "VK_LAYER_LUNARG_standard_validation"
};
//Constant vector of required device extensions
const std::vector<const char*> deviceExtensions = {
   VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

//Enable validation layers based on debug or release mode
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

//Proxy function to load function to create debug object
VkResult CreateDebugUtilsMessengerEXT(
   VkInstance instance, 
   const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, 
   const VkAllocationCallbacks* pAllocator, 
   VkDebugUtilsMessengerEXT* pDebugMessenger) {
   auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
   if (func != nullptr) {
      return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
   }
   else {
      return VK_ERROR_EXTENSION_NOT_PRESENT;
   }
}

//Proxy function to load function to destroy debug object
void DestroyDebugUtilsMessengerEXT(
   VkInstance instance, 
   VkDebugUtilsMessengerEXT debugMessenger, 
   const VkAllocationCallbacks* pAllocator) {
   auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
   if (func != nullptr) {
      func(instance, debugMessenger, pAllocator);
   }
}

//Struct for queue family that each have a subset of commands
struct QueueFamilyIndices {
   
   std::optional<uint32_t> graphicsFamily;
   std::optional<uint32_t> presentFamily;

   bool isComplete() {
      
      return graphicsFamily.has_value() && presentFamily.has_value();

   }

};
//Struct to contain details for creating a swap chain
struct SwapChainSupportDetails {
   VkSurfaceCapabilitiesKHR capabilities;
   std::vector<VkSurfaceFormatKHR> formats;
   std::vector<VkPresentModeKHR> presentModes;
};

class HelloTriangleApplication 
{
public:

   void run() {

      initWindow();
      initVulkan();
      mainLoop();
      cleanup();

   }

private:

   //Reference for window
   GLFWwindow* window;
   VkInstance instance;
   VkDebugUtilsMessengerEXT debugMessenger;
   VkSurfaceKHR surface;
   VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
   VkDevice device;
   VkQueue graphicsQueue;
   VkQueue presentQueue;
   VkSwapchainKHR swapChain;
   std::vector<VkImage> swapChainImages;
   VkFormat swapChainImageFormat;
   VkExtent2D swapChainExtent;
   std::vector<VkImageView> swapChainImageViews;

   void initWindow() {

      //Initialize glfw
      glfwInit();
      //Window hint for glfw to not use an OpenGL context
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
      //Window hint for glfw to not allow resizing of the created window
      glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
      //Store created window
      window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

   }

   void initVulkan() {

      createInstance();
      setupDebugMessenger();
      createSurface();
      pickPhysicalDevice();
      createLogicalDevice();
      createSwapChain();
      createImageViews();

   }

   void mainLoop() {

      while (!glfwWindowShouldClose(window)) {
         glfwPollEvents();
      }

   }

   void cleanup() {

      for (auto imageView : swapChainImageViews) {

         vkDestroyImageView(device, imageView, nullptr);

      }
      vkDestroySwapchainKHR(device, swapChain, nullptr);
      vkDestroyDevice(device, nullptr);
      if (enableValidationLayers) {
         DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
      }
      vkDestroySurfaceKHR(instance, surface, nullptr);
      vkDestroyInstance(instance, nullptr);
      glfwDestroyWindow(window);
      glfwTerminate();

   }

   void createInstance() {

      extensionCheck();
      if (enableValidationLayers && !checkValidationLayerSupport()) {
         throw std::runtime_error("validation layers requested, but not available!");
      }
      //Struct containing information for the application (optional)
      VkApplicationInfo appInfo = {};
      appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
      appInfo.pApplicationName = "Hello Triangle";
      appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
      appInfo.pEngineName = "No Engine";
      appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
      appInfo.apiVersion = VK_API_VERSION_1_0;
      //Struct for Vulkan driver to use which global extension and validation layers
      //Global in this context means to apply to the entire program, not a specific device.
      VkInstanceCreateInfo createInfo = {};
      createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
      createInfo.pApplicationInfo = &appInfo;
      //Specify desired global extensions
      //An extension is needed to interfaces with the window system (platform agnostic API == Vulkan)
      uint32_t glfwExtensionCount = 0;
      const char** glfwExtensions;
      glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
      //Specify what global validation layers to enable
      auto extensions = getRequiredExtensions();
      createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
      createInfo.ppEnabledExtensionNames = extensions.data();
      //Include validation layer names if enabled
      //Prevents returning VK_ERROR_LAYER_NOT_PRESENT error
      if (enableValidationLayers) {
         createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
         createInfo.ppEnabledLayerNames = validationLayers.data();
      }
      else {
         createInfo.enabledLayerCount = 0;  
      }
      //Vulkan generally follows 
      //ptr to creation info struct, 
      //ptr to custom allocator callbacks, 
      //ptr to variable to handle new object
      //Most Vulkan functions return VkResult which is either VK_SUCCESS or an error code
      if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
         throw std::runtime_error("failed to create instance!");
      }

   }

   void setupDebugMessenger() {

      if (!enableValidationLayers)
         return;
      VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
      createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
      //Specify which severities to be called
      createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT 
         | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT 
         | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      //Specify which messages to be called
      createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT 
         | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      //Specifies ptr to callback function
      createInfo.pfnUserCallback = debugCallback;
      createInfo.pUserData = nullptr;
      if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
         throw std::runtime_error("failed to set up debug messenger!");
      }

   }

   void createSurface() {

      //GLFW call to create a window surface
      //First parameter is the VkInstance
      //Second parameter is the GLFW window pointer
      //Third parameter is a custom allocator
      //Last parameter is the ptr to VkSurfaceKHR
      if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
         throw std::runtime_error("failed to create window surface!");
      }

   }

   void pickPhysicalDevice() {

      //List the amount of devices that supports Vulkan
      uint32_t deviceCount = 0;
      vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
      if (deviceCount == 0) {
         throw std::runtime_error("failed to find GPUs with Vulkan support!");
      }
      //Allocate VkPhysicalDevice handles
      std::vector<VkPhysicalDevice> devices(deviceCount);
      vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
      //Check if device can perform operations
      for (const auto& device : devices) {
         if (isDeviceSuitable(device)) {
            physicalDevice = device;
            break;
         }
      }
      if (physicalDevice == VK_NULL_HANDLE) {
         throw std::runtime_error("failed to find a suitable GPU!");
      }

   }

   void createLogicalDevice() {

      //Queue family creation
      QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
      //Queue creation for multiple queue families
      std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
      std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };
      float queuePriority = 1.0f;
      for (uint32_t queueFamily : uniqueQueueFamilies) {
         VkDeviceQueueCreateInfo queueCreateInfo = {};
         queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
         queueCreateInfo.queueFamilyIndex = queueFamily;
         queueCreateInfo.queueCount = 1;
         queueCreateInfo.pQueuePriorities = &queuePriority;
         queueCreateInfos.push_back(queueCreateInfo);
      }
      VkPhysicalDeviceFeatures deviceFeatures = {};
      //Logical device info creation
      VkDeviceCreateInfo createInfo = {};
      createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
      //Link logical device with queues
      createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
      createInfo.pQueueCreateInfos = queueCreateInfos.data();
      createInfo.pEnabledFeatures = &deviceFeatures;
      //Enable use for swap chain
      createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
      createInfo.ppEnabledExtensionNames = deviceExtensions.data();
      if (enableValidationLayers) {
         createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
         createInfo.ppEnabledLayerNames = validationLayers.data();
      }
      else {
         createInfo.enabledLayerCount = 0;
      }
      //Logical device link to physical device
      //First parameter is the physical device to link
      //Second parameter is the information of the queue family to use
      //Third parameter is the call back pointer (optional)
      //Last parameter is the variable for the logical device
      if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
         throw std::runtime_error("failed to create logical device!");
      }
      //Retrieve queue handles with the specified queue family
      //First parameter is the logical device
      //Second parameter is the queue family,
      //Third parameter is the queue index of the queue to retrieve
      //Last paramater is the variable for the queue
      vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
      vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);

   }

   void createSwapChain() {
      
      //Retrieve all necessary information to create swap chain
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
      VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
      VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
      VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
      //Decide on the amount of images to render
      uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
      if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
         imageCount = swapChainSupport.capabilities.maxImageCount;
      }
      //Set information for creating a swap cahin
      VkSwapchainCreateInfoKHR createInfo = {};
      createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
      createInfo.surface = surface;
      createInfo.minImageCount = imageCount;
      createInfo.imageFormat = surfaceFormat.format;
      createInfo.imageColorSpace = surfaceFormat.colorSpace;
      createInfo.imageExtent = extent;
      createInfo.imageArrayLayers = 1;
      createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
      QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
      uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };
      //Specify the type of sharing mode for the swap chain to use
      //Either to be exclusive in that the queue family takes ownership of that image
      //Or concurrent to share the images across queue families
      if (indices.graphicsFamily != indices.presentFamily) {
         createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
         createInfo.queueFamilyIndexCount = 2;
         createInfo.pQueueFamilyIndices = queueFamilyIndices;
      }
      else {
         createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
         createInfo.queueFamilyIndexCount = 0;
         createInfo.pQueueFamilyIndices = nullptr;
      }
      createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
      //Set the alpha/presentation mode/clipping
      createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
      createInfo.presentMode = presentMode;
      createInfo.clipped = VK_TRUE;
      createInfo.oldSwapchain = VK_NULL_HANDLE;
      //Create the swap chain
      if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
         throw std::runtime_error("failed to create swap chain!");
      }
      vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
      swapChainImages.resize(imageCount);
      vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
      //Store values for future usage
      swapChainImageFormat = surfaceFormat.format;
      swapChainExtent = extent;

   }

   void createImageViews() {
   
      //Resize list to fit all image views
      swapChainImageViews.resize(swapChainImages.size());
      //Iterate through each swap chain image
      for (size_t i = 0; i < swapChainImages.size(); i++) {

         //Create an image view
         VkImageViewCreateInfo createInfo = {};
         createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
         createInfo.image = swapChainImages[i];
         //Type of image view
         createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
         createInfo.format = swapChainImageFormat;
         //Color channels of image view
         createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
         createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
         createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
         createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
         //Purpose of image view
         createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
         createInfo.subresourceRange.baseMipLevel = 0;
         createInfo.subresourceRange.levelCount = 1;
         createInfo.subresourceRange.baseArrayLayer = 0;
         createInfo.subresourceRange.layerCount = 1;
         if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
         }

      }

   }

   void extensionCheck() {

      //Extension checking
      uint32_t extensionCount = 0;
      //Extension properties takes an 
      //optional first parameter for filtering by specific validation layers,
      //ptr that stores the number of extensions
      //array of VkExtensionProperties to store details of extensions
      vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
      std::vector<VkExtensionProperties> extensions(extensionCount);
      vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
      std::cout << "available extensions:" << std::endl;
      //VkExtensionProperties == struct which contains name and version of the extension
      for (const auto& extension : extensions) {
         std::cout << "\t" << extension.extensionName << std::endl;
      }

   }

   bool checkValidationLayerSupport() {

      //Retrieve available layers
      uint32_t layerCount;
      vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
      std::vector<VkLayerProperties> availableLayers(layerCount);
      vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
      //Check if layers in validationLayers are in availableLayers
      for (const char* layerName : validationLayers) {
         bool layerFound = false;
         for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
               layerFound = true;
               break;
            }
         }
         if (!layerFound) {
            return false;
         }
      }
      return true;

   }

   std::vector<const char*> getRequiredExtensions() {
      
      //Return the required list of extensions if validation layers is enabled
      uint32_t glfwExtensionCount = 0;
      const char** glfwExtensions;
      glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
      std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
      //Include VK_EXT_debug_utils extension for relaying debug messages
      if (enableValidationLayers) {
         extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
      }
      return extensions;

   }

   //First parameter is the severity of the message
   //Second parameter is the type of message
   //Third parameter is a struct detailing the message
   //Last parameter is a pointer allows passing user data to it
   //Returns boolean for Vulkan call to be aborted
   static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
      VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, 
      VkDebugUtilsMessageTypeFlagsEXT messageType, 
      const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, 
      void* pUserData) {

   std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
   return VK_FALSE;

   }

   bool isDeviceSuitable(VkPhysicalDevice device) {

      //Check for the device's properties and features
      /*VkPhysicalDeviceProperties deviceProperties;
      VkPhysicalDeviceFeatures deviceFeatures;
      vkGetPhysicalDeviceProperties(device, &deviceProperties);
      vkGetPhysicalDeviceFeatures(device, &deviceFeatures);*/
      //Find queue family that the device supports
      QueueFamilyIndices indices = findQueueFamilies(device);
      //Check for extensions
      bool extensionsSupported = checkDeviceExtensionSupport(device);
      //Check for swap chain is valid
      bool swapChainAdequate = false;
      if (extensionsSupported) {
         SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
         swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
      }
      return /*deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && deviceFeatures.geometryShader &&*/ 
         indices.isComplete() && extensionsSupported && swapChainAdequate;

   }

   QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {

      //Search for queue families that meet requirements for Vulkan operations
      QueueFamilyIndices indices;
      uint32_t queueFamilyCount = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
      std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
      vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
      //Check for queue family capabilities for Vulkan operations
      int i = 0;
      for (const auto& queueFamily : queueFamilies) {
         if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
         }
         //Check for a queue family that supports presenting window surfaces
         VkBool32 presentSupport = false;
         vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
         if (queueFamily.queueCount > 0 && presentSupport) {
            indices.presentFamily = i;
         }
         if (indices.isComplete()) {
            break;
         }
         i++;
      }
      return indices;

   }

   bool checkDeviceExtensionSupport(VkPhysicalDevice device) {

      uint32_t extensionCount;
      //Enumerate extensions
      vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
      std::vector<VkExtensionProperties> availableExtensions(extensionCount);
      vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
      std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
      //Cross reference required extensions with available
      //Track extensions to avoid reuse of extensions
      for (const auto& extension : availableExtensions) {
         requiredExtensions.erase(extension.extensionName);
      }
      return requiredExtensions.empty();

   }

   SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {

      //Retrieve details for creating a swap chain
      SwapChainSupportDetails details;
      vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
      //Query for support surface formats
      uint32_t formatCount;
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
      if (formatCount != 0) {
         details.formats.resize(formatCount);
         vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
      }
      //Query for suppported presentation modes
      uint32_t presentModeCount;
      vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
      if (presentModeCount != 0) {
         details.presentModes.resize(presentModeCount);
         vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
      }
      return details;

   }

   VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {

      //Pick surface format for swap chain
      //Surface format contains a format and a colorSpace
      //Pick a preferred format or a default from the available format that meets the criteria
      //Or return the first surface format
      if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED) {
         return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
      }
      for (const auto& availableFormat : availableFormats) {
         if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR){
            return availableFormat;
         }
      }
      return availableFormats[0];

   }

   VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
      
      //Pick the type of presentation mode
      //Pick the best desired presentation mode or check from available presentation modes
      //If criteria is met then use the presentation mode, or use a default
      VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;
      for (const auto& availablePresentMode : availablePresentModes) {
         if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
         }
         else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
            bestMode = availablePresentMode;
         }
      }
      return bestMode;

   }

   VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {

      //Pick the swap extent
      //Swap extent is the resolution for the swap chain images
      //Typically the resolution of the window to be drawn on
      if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
         return capabilities.currentExtent;
      }
      else {
         VkExtent2D actualExtent = { WIDTH, HEIGHT };
         actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
         actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
      }
   }

};

int main() {
   HelloTriangleApplication app;

   try {
      app.run();
   }
   catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}