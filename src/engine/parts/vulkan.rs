use std::sync::Arc;
use sdl2::sys::VkSurfaceKHR;
use vulkano::instance::Instance;
use vulkano::swapchain::Surface;

pub struct VulkanParts {
    pub instance: Arc<Instance>,
    pub surface: Arc<Surface>,
    pub surface_handle: VkSurfaceKHR,
}