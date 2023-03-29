use std::sync::Arc;
use sdl2::sys::VkSurfaceKHR;
use vulkano::instance::Instance;
use vulkano::swapchain::Surface;

pub struct VulkanPart {
    pub instance: Arc<Instance>,
    pub surface_handle: VkSurfaceKHR,
    pub surface: Surface,
}