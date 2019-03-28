#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <vector>
#include <optional>

//Constants for window size
const int WIDTH = 1280;
const int HEIGHT = 720;
//Constant vector for LunarG sstandard validation layers
const std::vector<const char*> validationLayers = {
   "VK_LAYER_LUNARG_standard_validation"
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

   bool isComplete() {
      
      return graphicsFamily.has_value();

   }

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
   VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
   VkDevice device;
   VkQueue graphicsQueue;

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
      pickPhysicalDevice();
      createLogicalDevice();

   }

   void mainLoop() {

      while (!glfwWindowShouldClose(window)) {
         glfwPollEvents();
      }

   }

   void cleanup() {

      if (enableValidationLayers) {
         DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
      }
      vkDestroyDevice(device, nullptr);
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
         if (isDevicesSuitable(device)) {
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
      VkDeviceQueueCreateInfo queueCreateInfo = {};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
      queueCreateInfo.queueCount = 1;
      float queuePriority = 1.0f;
      queueCreateInfo.pQueuePriorities = &queuePriority;
      VkPhysicalDeviceFeatures deviceFeatures = {};
      //Logical device info creation
      VkDeviceCreateInfo createInfo = {};
      createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
      createInfo.pQueueCreateInfos = &queueCreateInfo;
      createInfo.queueCreateInfoCount = 1;
      createInfo.pEnabledFeatures = &deviceFeatures;
      createInfo.enabledExtensionCount = 0;
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

   bool isDevicesSuitable(VkPhysicalDevice device) {

      //Check for the device's properties and features
      /*VkPhysicalDeviceProperties deviceProperties;
      VkPhysicalDeviceFeatures deviceFeatures;
      vkGetPhysicalDeviceProperties(device, &deviceProperties);
      vkGetPhysicalDeviceFeatures(device, &deviceFeatures);*/
      //Find queue family that the device supports
      QueueFamilyIndices indices = findQueueFamilies(device);
      return /*deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && deviceFeatures.geometryShader &&*/ indices.isComplete();

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
         if (indices.isComplete()) {
            break;
         }
         i++;
      }
      return indices;

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