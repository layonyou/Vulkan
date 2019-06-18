#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define STB_IMAGE_IMPLEMENTATION
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <vector>
#include <optional>
#include <set>
#include <algorithm>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>
#include <array>
#include <stb_image.h>

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
//Constant for how many frames to process concurrently
const int MAX_FRAMES_IN_FLIGHT = 2;

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
//Struct to contain vertex details
struct Vertex {

   glm::vec2 pos;
   glm::vec3 color;

   //How Vulkan passes in this data format to vertex shader after uploaded to GPU memory
   static VkVertexInputBindingDescription getBindingDescription() {

      VkVertexInputBindingDescription bindingDescription = {};
      bindingDescription.binding = 0;
      bindingDescription.stride = sizeof(Vertex);
      bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
      return bindingDescription;

   }

   //How to handle vertex input
   static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {

      std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = {};
      attributeDescriptions[0].binding = 0;
      attributeDescriptions[0].location = 0;
      attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
      attributeDescriptions[0].offset = offsetof(Vertex, pos);
      attributeDescriptions[1].binding = 0;
      attributeDescriptions[1].location = 1;
      attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
      attributeDescriptions[1].offset = offsetof(Vertex, color);
      return attributeDescriptions;

   }

};

//Constant vector of Vertex to store Vertex data
const std::vector<Vertex> vertices = {
   {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
   {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
   {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
   {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};
//Constant vector to represent indicies for Vertices
const std::vector<uint16_t> indices = {
   0, 1, 2, 2, 3, 0
};

//Struct containing the matrix transformations
struct UniformBufferObject {

   //Match memory alignment with uniform definition in shader
   alignas(16) glm::mat4 model;
   alignas(16) glm::mat4 view;
   alignas(16) glm::mat4 proj;

};

class HelloTriangleApplication {

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
   VkRenderPass renderPass;
   VkDescriptorSetLayout descriptorSetLayout;
   VkPipelineLayout pipelineLayout;
   VkPipeline graphicsPipeline;
   std::vector<VkFramebuffer> swapChainFramebuffers;
   VkCommandPool commandPool;
   VkImage textureImage;
   VkDeviceMemory textureImageMemory;
   VkBuffer vertexBuffer;
   VkDeviceMemory vertexBufferMemory;
   VkBuffer indexBuffer;
   VkDeviceMemory indexBufferMemory;
   std::vector<VkBuffer> uniformBuffers;
   std::vector<VkDeviceMemory> uniformBuffersMemory;
   VkDescriptorPool descriptorPool;
   std::vector<VkDescriptorSet> descriptorSets;
   std::vector<VkCommandBuffer> commandBuffers;
   std::vector<VkSemaphore> imageAvailableSemaphores;
   std::vector<VkSemaphore> renderFinishedSemaphore;
   std::vector<VkFence> inFlightFences;
   size_t currentFrame = 0;
   bool framebufferResized = false;

   void initWindow() {

      //Initialize glfw
      glfwInit();
      //Window hint for glfw to not use an OpenGL context
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
      //Window hint for glfw to not allow resizing of the created window
      //glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
      //Store created window
      window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
      glfwSetWindowUserPointer(window, this);
      glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

   }

   static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {

      //Reference window for resizing
      auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
      app->framebufferResized = true;

   }

   void initVulkan() {

      createInstance();
      setupDebugMessenger();
      createSurface();
      pickPhysicalDevice();
      createLogicalDevice();
      createSwapChain();
      createImageViews();
      createRenderPass();
      createDescriptorSetLayout();
      createGraphicsPipeline();
      createFramebuffers();
      createCommandPool();
      createTextureImage();
      createVertexBuffer();
      createIndexBuffer();
      createUniformBuffers();
      createDescriptorPool();
      createDescriptorSets();
      createCommandBuffers();
      createSyncObjects();

   }

   void mainLoop() {

      while (!glfwWindowShouldClose(window)) {
         glfwPollEvents();
         drawFrame();
      }
      vkDeviceWaitIdle(device);

   }

   void drawFrame() {

      vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());
      //Acquire image from swap chain and VkImage from imageIndex to refernce correct command buffer
      uint32_t imageIndex;
      VkResult result = vkAcquireNextImageKHR(device, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
      //Check if the swap chain is out of date
      if (result == VK_ERROR_OUT_OF_DATE_KHR) {
         recreateSwapChain();
         return;
      }
      else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
         throw std::runtime_error("failed to acquire swap chain image!");
      }
      //Update uniform buffer
      updateUniformBuffer(imageIndex);
      //Submit image for rendering
      VkSubmitInfo submitInfo = {};
      submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
      VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
      submitInfo.waitSemaphoreCount = 1;
      submitInfo.pWaitSemaphores = waitSemaphores;
      submitInfo.pWaitDstStageMask = waitStages;
      submitInfo.commandBufferCount = 1;
      submitInfo.pCommandBuffers = &commandBuffers[imageIndex];
      VkSemaphore signalSemaphores[] = { renderFinishedSemaphore[currentFrame] };
      submitInfo.signalSemaphoreCount = 1;
      submitInfo.pSignalSemaphores = signalSemaphores;
      vkResetFences(device, 1, &inFlightFences[currentFrame]);
      if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
         throw std::runtime_error("failed to submit draw command buffer!");
      }
      //Present submitted rendered image
      VkPresentInfoKHR presentInfo = {};
      presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
      presentInfo.waitSemaphoreCount = 1;
      presentInfo.pWaitSemaphores = signalSemaphores;
      VkSwapchainKHR swapChains[] = { swapChain };
      presentInfo.swapchainCount = 1;
      presentInfo.pSwapchains = swapChains;
      presentInfo.pImageIndices = &imageIndex;
      presentInfo.pResults = nullptr;
      //Present rendered image and update current frame
      result = vkQueuePresentKHR(presentQueue, &presentInfo);
      //Check if image was able to be presented
      if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
         framebufferResized = false;
         recreateSwapChain();
      }
      else if (result != VK_SUCCESS) {
         throw std::runtime_error("failed to present swap chain image!");
      }
      currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

   }

   void cleanup() {

      cleanupSwapChain();
      vkDestroyImage(device, textureImage, nullptr);
      vkFreeMemory(device, textureImageMemory, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++){
         vkDestroyFence(device, inFlightFences[i], nullptr);
         vkDestroySemaphore(device, renderFinishedSemaphore[i], nullptr);
         vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
      }
      vkDestroyBuffer(device, indexBuffer, nullptr);
      vkFreeMemory(device, indexBufferMemory, nullptr);
      vkDestroyBuffer(device, vertexBuffer, nullptr);
      vkFreeMemory(device, vertexBufferMemory, nullptr);
      vkDestroyCommandPool(device, commandPool, nullptr);
      vkDestroyDevice(device, nullptr);
      if (enableValidationLayers) {
         DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
      }
      vkDestroySurfaceKHR(instance, surface, nullptr);
      vkDestroyInstance(instance, nullptr);
      glfwDestroyWindow(window);
      glfwTerminate();

   }

   void cleanupSwapChain() {

      for (auto framebuffer : swapChainFramebuffers) {
         vkDestroyFramebuffer(device, framebuffer, nullptr);
      }
      vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
      vkDestroyPipeline(device, graphicsPipeline, nullptr);
      vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
      vkDestroyRenderPass(device, renderPass, nullptr);
      for (auto imageView : swapChainImageViews) {
         vkDestroyImageView(device, imageView, nullptr);
      }
      vkDestroySwapchainKHR(device, swapChain, nullptr);
      for (size_t i = 0; i < swapChainImages.size(); i++) {
         vkDestroyBuffer(device, uniformBuffers[i], nullptr);
         vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
      }
      vkDestroyDescriptorPool(device, descriptorPool, nullptr);

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

   void createRenderPass() {

      //Create a description defining what to do with computed images
      VkAttachmentDescription colorAttachment = {};
      colorAttachment.format = swapChainImageFormat;
      colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
      colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      //Create an attachment reference which will be used for subpasses
      VkAttachmentReference colorAttachmentRef = {};
      colorAttachmentRef.attachment = 0;
      colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      //Create a subpass which is used for subsequent rendering operations
      VkSubpassDescription subpass = {};
      subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
      subpass.colorAttachmentCount = 1;
      subpass.pColorAttachments = &colorAttachmentRef;
      //Create a render pass object
      VkRenderPassCreateInfo renderPassInfo = {};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
      renderPassInfo.attachmentCount = 1;
      renderPassInfo.pAttachments = &colorAttachment;
      renderPassInfo.subpassCount = 1;
      renderPassInfo.pSubpasses = &subpass;
      //Create a subpass dependency to make render pass wait for specified stage
      VkSubpassDependency dependency = {};
      dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
      dependency.dstSubpass = 0;
      dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependency.srcAccessMask = 0;
      dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      renderPassInfo.dependencyCount = 1;
      renderPassInfo.pDependencies = &dependency;
      if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
         throw std::runtime_error("failed to create render pass!");
      }

   }

   void createDescriptorSetLayout() {

      //Specify which binding from vertex shader to use
      VkDescriptorSetLayoutBinding uboLayoutBinding = {};
      uboLayoutBinding.binding = 0;
      uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      uboLayoutBinding.descriptorCount = 1;
      //Specify which stage should the descriptor execute
      uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
      uboLayoutBinding.pImmutableSamplers = nullptr;
      //Create decriptor set layout to bind to pipeline
      VkDescriptorSetLayoutCreateInfo layoutInfo = {};
      layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      layoutInfo.bindingCount = 1;
      layoutInfo.pBindings = &uboLayoutBinding;
      if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
         throw std::runtime_error("failed to create descriptor set layout!");
      }

   }

   void createGraphicsPipeline() {

      //Read file is a spv file compiled from fragment shader and vertex shader
      auto vertShaderCode = readFile("../Resources/shaders/vert.spv");
      auto fragShaderCode = readFile("../Resources/shaders/frag.spv");
      //Wrap vertex and fragment into VkShaderModules
      VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
      VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
      //Create staging info for vertex shader
      VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
      vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
      vertShaderStageInfo.module = vertShaderModule;
      vertShaderStageInfo.pName = "main";
      //Create staging info for fragment shader
      VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
      fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
      fragShaderStageInfo.module = fragShaderModule;
      fragShaderStageInfo.pName = "main";
      //Consolidate vertex shader and fragment shader info into an array
      VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };
      //Create info for vertex inputs which describes the format of the vertex data
      VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
      vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
      //Create binding and attribute descriptions to allow reading in of vertices
      auto bindingDescription = Vertex::getBindingDescription();
      auto attributeDescriptions = Vertex::getAttributeDescriptions();
      vertexInputInfo.vertexBindingDescriptionCount = 1;
      vertexInputInfo.pVertexBindingDescriptions = &bindingDescription; 
      vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
      vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
      //Create info for the input assembly which describes what kind of geometry and for enabling primitive restart
      VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
      inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
      inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      inputAssembly.primitiveRestartEnable = VK_FALSE;
      //Create a viewport which is the region of the framebuffer which the output will be rendered to
      VkViewport viewport = {};
      viewport.x = 0.0f;
      viewport.y = 0.0f;
      viewport.width = (float)swapChainExtent.width;
      viewport.height = (float)swapChainExtent.height;
      viewport.minDepth = 0.0f;
      viewport.maxDepth = 1.0f;
      //Create a scissor to define the area to discard in the rasterizer
      VkRect2D scissor = {};
      scissor.offset = { 0, 0 };
      scissor.extent = swapChainExtent;
      VkPipelineViewportStateCreateInfo viewportState = {};
      viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
      viewportState.viewportCount = 1;
      viewportState.pViewports = &viewport;
      viewportState.scissorCount = 1;
      viewportState.pScissors = &scissor;
      //Create a rasterizer which takes in the geometry from vertex shader and converts it into fragments for coloring by the fragment shader
      VkPipelineRasterizationStateCreateInfo rasterizer = {};
      rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
      rasterizer.depthClampEnable = VK_FALSE;
      rasterizer.rasterizerDiscardEnable = VK_FALSE;
      rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
      rasterizer.lineWidth = 1.0f;
      rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
      rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
      rasterizer.depthBiasEnable = VK_FALSE;
      rasterizer.depthBiasConstantFactor = 0.0f;
      rasterizer.depthBiasClamp = 0.0f;
      rasterizer.depthBiasSlopeFactor = 0.0f;
      //Create an info to configure multisampling
      VkPipelineMultisampleStateCreateInfo multisampling = {};
      multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
      multisampling.sampleShadingEnable = VK_FALSE;
      multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
      multisampling.minSampleShading = 1.0f;
      multisampling.pSampleMask = nullptr;
      multisampling.alphaToCoverageEnable = VK_FALSE;
      multisampling.alphaToOneEnable = VK_FALSE;
      //Create an color blend info which contains the configuration per attached framebuffer
      VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
      colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      colorBlendAttachment.blendEnable = VK_FALSE;
      colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
      colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
      colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
      colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
      colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
      //Create an info which references the array of structures for all framebuffers 
      //and provides the capability to set blend constants to affect blending
      VkPipelineColorBlendStateCreateInfo colorBlending = {};
      colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
      colorBlending.logicOpEnable = VK_FALSE;
      colorBlending.logicOp = VK_LOGIC_OP_COPY;
      colorBlending.attachmentCount = 1;
      colorBlending.pAttachments = &colorBlendAttachment;
      colorBlending.blendConstants[0] = 0.0f;
      colorBlending.blendConstants[1] = 0.0f;
      colorBlending.blendConstants[2] = 0.0f;
      colorBlending.blendConstants[3] = 0.0f;
      //Create a pipeline layout object
      VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
      pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipelineLayoutInfo.setLayoutCount = 1;
      pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
      pipelineLayoutInfo.pushConstantRangeCount = 0;
      pipelineLayoutInfo.pPushConstantRanges = nullptr;
      if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
         throw std::runtime_error("failed to create pipeline layout!");
      }
      VkGraphicsPipelineCreateInfo pipelineInfo = {};
      pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
      pipelineInfo.stageCount = 2;
      pipelineInfo.pStages = shaderStages;
      pipelineInfo.pVertexInputState = &vertexInputInfo;
      pipelineInfo.pInputAssemblyState = &inputAssembly;
      pipelineInfo.pViewportState = &viewportState;
      pipelineInfo.pRasterizationState = &rasterizer;
      pipelineInfo.pMultisampleState = &multisampling;
      pipelineInfo.pDepthStencilState = nullptr;
      pipelineInfo.pColorBlendState = &colorBlending;
      pipelineInfo.pDynamicState = nullptr;
      pipelineInfo.layout = pipelineLayout;
      pipelineInfo.renderPass = renderPass;
      pipelineInfo.subpass = 0;
      pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
      pipelineInfo.basePipelineIndex = -1;
      if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
         throw std::runtime_error("failed to create graphics pipeline!");
      }
      //Destory vertex and fragment shader modules
      vkDestroyShaderModule(device, fragShaderModule, nullptr);
      vkDestroyShaderModule(device, vertShaderModule, nullptr);

   }

   void createFramebuffers() {

      //Resize container to hold all framebuffers
      swapChainFramebuffers.resize(swapChainImageViews.size());
      //Iterate and crete framebuffers for each image view
      for (size_t i = 0; i < swapChainImageViews.size(); i++) {
         VkImageView attachments[] = { swapChainImageViews[i] };
         VkFramebufferCreateInfo framebufferInfo = {};
         framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
         framebufferInfo.renderPass = renderPass;
         framebufferInfo.attachmentCount = 1;
         framebufferInfo.pAttachments = attachments;
         framebufferInfo.width = swapChainExtent.width;
         framebufferInfo.height = swapChainExtent.height;
         framebufferInfo.layers = 1;
         if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
         }
      }
   }

   void createCommandPool() {

      //Create a command pool which manages the memory used to store buffers and command buffers are allocated from them
      QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
      VkCommandPoolCreateInfo poolInfo = {};
      poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
      poolInfo.flags = 0;
      if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
         throw std::runtime_error("failed to create command pool!");
      }

   }

   void createTextureImage() {

      //Load the texture
      int texWidth, texHeight, texChannels;
      stbi_uc* pixels = stbi_load("../Resources/Textures/test.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
      VkDeviceSize imageSize = texWidth * texHeight * 4;
      if (!pixels) {
         throw std::runtime_error("failed to load texture image!");
      }
      //Create a staging buffer and memory
      VkBuffer stagingBuffer;
      VkDeviceMemory stagingBufferMemory;
      createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
      void* data;
      //Copy texture into staging buffer
      vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
      memcpy(data, pixels, static_cast<size_t>(imageSize));
      vkUnmapMemory(device, stagingBufferMemory);
      //Free the pixel values and create the image from the texture
      stbi_image_free(pixels);
      createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);
      //Copy staging buffer to texture image
      transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
      transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      //Free staging buffer and memory
      vkDestroyBuffer(device, stagingBuffer, nullptr);
      vkFreeMemory(device, stagingBufferMemory, nullptr);

   }

   void createImage(uint32_t width, uint32_t height, 
      VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, 
      VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
      //Create an image
      VkImageCreateInfo imageInfo = {};
      imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      imageInfo.imageType = VK_IMAGE_TYPE_2D;
      imageInfo.extent.width = width;
      imageInfo.extent.height = height;
      imageInfo.extent.depth = 1;
      imageInfo.mipLevels = 1;
      imageInfo.arrayLayers = 1;
      imageInfo.format = format;
      imageInfo.tiling = tiling;
      imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      imageInfo.usage = usage;
      imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
      imageInfo.flags = 0;
      if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
         throw std::runtime_error("failed to create image!");
      }
      //Allocate the required memory for the image
      VkMemoryRequirements memRequirements;
      vkGetImageMemoryRequirements(device, image, &memRequirements);
      VkMemoryAllocateInfo allocInfo = {};
      allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocInfo.allocationSize = memRequirements.size;
      allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
      if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
         throw std::runtime_error("failed to allocate image memory!");
      }
      vkBindImageMemory(device, image, imageMemory, 0);


   }

   void createVertexBuffer() {

      //Create a buffer to store data for operations by the GPU
      VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
      //Staging buffer to copy data onto CPU
      VkBuffer stagingBuffer;
      VkDeviceMemory stagingBufferMemory;
      createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
      //Copy vertex data to the buffer from CPU to GPU
      void* data;
      vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
      memcpy(data, vertices.data(), (size_t) bufferSize);
      vkUnmapMemory(device, stagingBufferMemory);
      //Final buffer visible only to device
      createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
      //Copy staging buffer to final buffer
      copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
      //Destory staging buffer and free staging buffer memory
      vkDestroyBuffer(device, stagingBuffer, nullptr);
      vkFreeMemory(device, stagingBufferMemory, nullptr);

   }

   void createIndexBuffer() {

      //Create a buffer to store data for operations by the GPU
      VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
      //Staging buffer to copy data onto CPU
      VkBuffer stagingBuffer;
      VkDeviceMemory stagingBufferMemory;
      createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
      //Copy index data to the buffer from CPU to GPU
      void* data;
      vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
      memcpy(data, indices.data(), (size_t)bufferSize);
      vkUnmapMemory(device, stagingBufferMemory);
      //Final buffer visible only to device
      createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
      //Copy staging buffer to final buffer
      copyBuffer(stagingBuffer, indexBuffer, bufferSize);
      //Destory staging buffer and free staging buffer memory
      vkDestroyBuffer(device, stagingBuffer, nullptr);
      vkFreeMemory(device, stagingBufferMemory, nullptr);


   }

   void createUniformBuffers() {

      //Create a uniform buffer to the amount of swap chain images
      VkDeviceSize bufferSize = sizeof(UniformBufferObject);
      uniformBuffers.resize(swapChainImages.size());
      uniformBuffersMemory.resize(swapChainImages.size());
      for (size_t i = 0; i < swapChainImages.size(); i++) {
         createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
      }

   }

   void createDescriptorPool() {

      //Describe descriptor types the descriptor sets will contain and the amount
      VkDescriptorPoolSize poolSize = {};
      poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      poolSize.descriptorCount = static_cast<uint32_t>(swapChainImages.size());
      //Allocate a descriptor for each frame
      VkDescriptorPoolCreateInfo poolInfo = {};
      poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      poolInfo.poolSizeCount = 1;
      poolInfo.pPoolSizes = &poolSize;
      poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());
      //Create a descriptor pool for descriptor sets
      if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
         throw std::runtime_error("failed to create descriptor pool!");
      }

   }

   void createDescriptorSets() {

      //Allocate a descriptor set
      std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);
      VkDescriptorSetAllocateInfo allocInfo = {};
      allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      allocInfo.descriptorPool = descriptorPool;
      allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
      allocInfo.pSetLayouts = layouts.data();
      descriptorSets.resize(swapChainImages.size());
      //Create descriptor sets
      if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
         throw std::runtime_error("failed to allocate descriptor sets!");
      }
      for (size_t i = 0; i < swapChainImages.size(); i++) {
         //Buffer and region that contains the descriptor's data
         VkDescriptorBufferInfo bufferInfo = {};
         bufferInfo.buffer = uniformBuffers[i];
         bufferInfo.offset = 0;
         bufferInfo.range = sizeof(UniformBufferObject);
         //Configure descriptors using VkWriteDescriptorSet
         VkWriteDescriptorSet descriptorWrite = {};
         descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
         //Specify the uniform buffer binding index
         descriptorWrite.dstSet = descriptorSets[i];
         descriptorWrite.dstBinding = 0;
         descriptorWrite.dstArrayElement = 0;
         descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
         descriptorWrite.descriptorCount = 1;
         descriptorWrite.pBufferInfo = &bufferInfo;
         descriptorWrite.pImageInfo = nullptr;
         descriptorWrite.pTexelBufferView = nullptr;
         //Update descriptor sets
         vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
      }

   }

   void updateUniformBuffer(uint32_t currentImage) {

      //Determine the time since last render to determine transformation for next render
      static auto startTime = std::chrono::high_resolution_clock::now();
      auto currentTime = std::chrono::high_resolution_clock::now();
      float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
      UniformBufferObject ubo = {};
      ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
      ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
      ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
      //GLM was designed for OpenGl, Vulkan needs to invert the Y coordinate
      ubo.proj[1][1] *= -1;
      //Copy updated matrix transformations to uniform buffer memory
      void* data;
      vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
      memcpy(data, &ubo, sizeof(ubo));
      vkUnmapMemory(device, uniformBuffersMemory[currentImage]);

   }

   void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
   
      //Create a buffer to store data for operations by the GPU
      VkBufferCreateInfo bufferInfo = {};
      bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bufferInfo.size = size;
      bufferInfo.usage = usage;
      bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
         throw std::runtime_error("failed to create vertex buffer!");
      }
      VkMemoryRequirements memRequirements;
      vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
      //Determine the correct memory type to allocate memory to fill with vertex data
      VkMemoryAllocateInfo allocInfo = {};
      allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocInfo.allocationSize = memRequirements.size;
      allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
      if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
         throw std::runtime_error("failed to allocate vertex buffer memory!");
      }
      vkBindBufferMemory(device, buffer, bufferMemory, 0);

   }

   void createCommandBuffers() {

      //Create a command buffer for each image in the swap chain
      commandBuffers.resize(swapChainFramebuffers.size());
      //Allocate command buffers
      VkCommandBufferAllocateInfo allocInfo = {};
      allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      allocInfo.commandPool = commandPool;
      allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();
      if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
         throw std::runtime_error("failed to allocate command buffers!");
      }
      //Create a command buffer detailing about its usage for each command buffer
      for (size_t i = 0; i < commandBuffers.size(); i++) {
         VkCommandBufferBeginInfo beginInfo = {};
         beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
         beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
         beginInfo.pInheritanceInfo = nullptr;
         if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
         }
         //Start a render pass to begin drawing
         VkRenderPassBeginInfo renderPassInfo = {};
         renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
         renderPassInfo.renderPass = renderPass;
         renderPassInfo.framebuffer = swapChainFramebuffers[i];
         renderPassInfo.renderArea.offset = { 0, 0 };
         renderPassInfo.renderArea.extent = swapChainExtent;
         VkClearValue clearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
         renderPassInfo.clearValueCount = 1;
         renderPassInfo.pClearValues = &clearColor;
         vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
         //Bind the command buffers to the graphics pipeline
         vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
         //Bind vertex buffers to the graphics pipeline
         VkBuffer vertexBuffers[] = { vertexBuffer };
         VkDeviceSize offsets[] = { 0 };
         vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);
         //Bind index buffers to the graphics pipeline
         vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT16);
         //Bind descriptor sets to a pipeline
         vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);
         //Draw with indices
         vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
         vkCmdEndRenderPass(commandBuffers[i]);
         if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
         }
      }

   }

   //TODO: Layout transitions

   void createSyncObjects() {

      //Create semaphores and fences for sychronization of rendering frames
      imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
      renderFinishedSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
      inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
      //Create semaphore info
      VkSemaphoreCreateInfo semaphoreInfo = {};
      semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
      //Create fence info and initialize in signaled state
      VkFenceCreateInfo fenceInfo = {};
      fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
         if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS
            || vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore[i]) != VK_SUCCESS
            || vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
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
         if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
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
         //Retrieve width and height of the window
         int width, height;
         glfwGetFramebufferSize(window, &width, &height);
         VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
         actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
         actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
         return actualExtent;
      }
   }

   uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {

      //Find the correct memory requirements to use for memory allocation
      VkPhysicalDeviceMemoryProperties memProperties;
      vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
      //Iterate through each memory type and return a memory type based on properties
      for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
         if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
         }
      }
      throw std::runtime_error("failed to find suitable memory type!");

   }

   void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {

      //Create a one time use command to copy srcBuffer to dstBuffer
      VkCommandBuffer commandBuffer = beginSingleTimeCommands();
      VkBufferCopy copyRegion = {};
      copyRegion.size = size;
      vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
      //Free command buffer
      endSingleTimeCommands(commandBuffer);

   }

   VkCommandBuffer beginSingleTimeCommands() {

      //Create a one time use command to copy srcBuffer to dstBuffer
      VkCommandBufferAllocateInfo allocInfo = {};
      allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      allocInfo.commandPool = commandPool;
      allocInfo.commandBufferCount = 1;
      VkCommandBuffer commandBuffer;
      vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
      VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(commandBuffer, &beginInfo);
      return commandBuffer;

   }

   void endSingleTimeCommands(VkCommandBuffer commandBuffer) {

      //Submit copy command and wait for completion
      vkEndCommandBuffer(commandBuffer);
      VkSubmitInfo submitInfo = {};
      submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submitInfo.commandBufferCount = 1;
      submitInfo.pCommandBuffers = &commandBuffer;
      vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
      vkQueueWaitIdle(graphicsQueue);
      vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

   }

   void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {

      VkCommandBuffer commandBuffer = beginSingleTimeCommands();
      //Specify parts of the buffer to copied to certain parts of the image
      VkBufferImageCopy region = {};
      region.bufferOffset = 0;
      region.bufferRowLength = 0;
      region.bufferImageHeight = 0;
      region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      region.imageSubresource.mipLevel = 0;
      region.imageSubresource.baseArrayLayer = 0;
      region.imageSubresource.layerCount = 1;
      region.imageOffset = { 0,0,0 };
      region.imageExtent = { width, height, 1 };
      vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
      endSingleTimeCommands(commandBuffer);

   }

   void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {

      VkCommandBuffer commandBuffer = beginSingleTimeCommands();
      //Create an image barrier for synchronizing transitions between oldLayout and newLayout
      VkImageMemoryBarrier barrier = {};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.oldLayout = oldLayout;
      barrier.newLayout = newLayout;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = image;
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      barrier.subresourceRange.baseMipLevel = 0;
      barrier.subresourceRange.levelCount = 1;
      barrier.subresourceRange.baseArrayLayer = 0;
      barrier.subresourceRange.layerCount = 1;
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = 0;
      //Check for transitions of oldLayout and newLayout to set the correct access masks
      VkPipelineStageFlags sourceStage;
      VkPipelineStageFlags destinationStage;
      if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
         barrier.srcAccessMask = 0;
         barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
         sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
         destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      }
      else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
         barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
         barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
         sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
         destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      }
      else {
         throw std::invalid_argument("unsupported layout transition!");
      }
      //Create the pipeline barrier
      vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
      endSingleTimeCommands(commandBuffer);

   }

   void recreateSwapChain() {

      //Pause window until window is back in foreground
      int width = 0, height = 0;
      while (width == 0 || height == 0) {
         glfwGetFramebufferSize(window, &width, &height);
         glfwWaitEvents();
      }
      //Any resizing of the window will recreate the swap chain
      vkDeviceWaitIdle(device);
      cleanupSwapChain();
      createSwapChain();
      createImageViews();
      createRenderPass();
      createGraphicsPipeline();
      createFramebuffers();
      createUniformBuffers();
      createDescriptorPool();
      createDescriptorSets();
      createCommandBuffers();

   }

   static std::vector<char> readFile(const std::string& filename) {

      //Read in a binary file starting from the end of the file
      std::ifstream file(filename, std::ios::ate | std::ios::binary);
      if (!file.is_open()) {
         throw std::runtime_error("failed to open file!");
      }
      size_t fileSize = (size_t)file.tellg();
      std::vector<char> buffer(fileSize);
      file.seekg(0);
      file.read(buffer.data(), fileSize);
      file.close();
      std::cout << fileSize << std::endl;
      return buffer;

   }

   VkShaderModule createShaderModule(const std::vector<char>& code) {

      //Wrap shader code into a VkShaderModule
      VkShaderModuleCreateInfo createInfo = {};
      createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      createInfo.codeSize = code.size();
      createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
      //Create a VkShaderModule from byte code passed in
      VkShaderModule shaderModule;
      if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
         throw std::runtime_error("failed to create shader module!");
      }
      return shaderModule;

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