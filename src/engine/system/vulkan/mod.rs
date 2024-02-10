use vulkano::buffer::AllocateBufferError;
use vulkano::image::AllocateImageError;
use vulkano::pipeline::layout::IntoPipelineLayoutCreateInfoError;
use vulkano::{Validated, ValidationError, VulkanError};

pub mod desc;
pub mod utils;

pub mod beautiful_lines;
pub mod buffers;
#[cfg(feature = "ui-egui")]
pub mod egui;
pub mod fps;
pub mod glowing_balls;
pub mod lines;
pub mod pipelines;
pub mod system;
pub mod textured;
pub mod textures;
pub mod triangles;
pub mod wds;
#[cfg(feature = "world2d")]
pub mod world2d;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Failed to retrieve vulkan extensions for the surface: {0}")]
    MissingVulkanExtensionsForSurface(String),
    #[error("Unable to enumerate physical devices of the system: {0:?}")]
    FailedToEnumeratePhysicalDevices(VulkanError),
    #[error("Unable to find physical devices that satisfies all needs")]
    NoSatisfyingPhysicalDevicePresent,
    #[error("Failed to initialize device instance {0:?}")]
    DeviceInitializationFailed(Validated<VulkanError>),
    #[error("Failed to initialize swapchain instance {0:?}")]
    SwapchainInitializationFailed(Validated<VulkanError>),
    #[error("Failed to retrieve surface capabilities: {0:?}")]
    FailedToRetrieveSurfaceCapabilities(Validated<VulkanError>),
    #[error("Failed to retrieve surface formats: {0:?}")]
    FailedToRetrieveSurfaceFormats(Validated<VulkanError>),
    #[error("Failed to create framebuffers: {0:?}")]
    FailedToCreateFramebuffers(Validated<VulkanError>),
    #[error("Failed to create render pass: {0:?}")]
    RenderPassCreationError(#[from] PipelineCreateError),
    #[error("Failed to allocate Buffer of WriteDescriptor for binding {1}: {0:?}")]
    FailedToAllocateWriteDescriptorBuffer(Validated<AllocateBufferError>, u32),
    #[error("Failed to update Buffer of WriteDescriptor for binding {1}: {0:?}")]
    FailedToUpdateWriteDescriptorBuffer(Box<ValidationError>, u32),
    #[error("Failed to create a (secondary) command buffer: {0:?} ")]
    FailedToCreateCommandBuffer(Validated<VulkanError>),
}

#[derive(thiserror::Error, Debug)]
pub enum DrawError {
    // #[error("Vulkan Error: {0}")]
    // VulkanError(#[from] Validated<VulkanError>),
    #[error("Validation Error: {0}")]
    ValidationError(#[from] Box<ValidationError>),
    #[error("Failed to allocate buffer: {0}")]
    BufferAllocateError(#[from] Validated<AllocateBufferError>),
    #[error("Failed to re-create the framebuffers: {0}")]
    FailedToRecreateTheFramebuffers(Validated<VulkanError>),
    // #[error("Failed to execute the pipeline: {0}")]
    // PipelineExecutionError(#[from] Validated<VulkanError>),
    #[error("Failed to build command buffer: {0}")]
    FailedToBuildCommandBuffer(Validated<VulkanError>),
    #[error("Failed to acquire the next swapchain image: {0}")]
    FailedToAcquireSwapchainImage(VulkanError),
}

#[derive(thiserror::Error, Debug)]
pub enum UploadError {
    #[error("Vulkan Error: {0}")]
    VulkanError(#[from] Validated<VulkanError>),
    #[error("Failed to upload the image: {0}")]
    ImageError(#[from] Validated<AllocateImageError>),
    #[error("Failed to allocate buffer: {0}")]
    BufferAllocateError(#[from] Validated<AllocateBufferError>),
}

#[derive(thiserror::Error, Debug)]
pub enum PipelineCreateError {
    #[error("Vulkan Error: {0}")]
    VulkanError(#[from] Validated<VulkanError>),
    #[error("Validation Error: {0}")]
    ValidationError(#[from] Box<ValidationError>),
    #[error("Failed to create pipeline: {0}")]
    IntoPipelineLayoutCreateInfoError(#[from] IntoPipelineLayoutCreateInfoError),
    #[error("Failed to load at least one shader: {0}")]
    ShaderLoadError(#[from] ShaderLoadError),
    #[error("Failed to init pipeline because of allocation error: {0}")]
    PipelineInitErrorOnAllocation(#[from] Validated<AllocateBufferError>),
}

#[derive(thiserror::Error, Debug)]
pub enum ShaderLoadError {
    #[error("Vulkan Error: {0}")]
    VulkanError(#[from] Validated<VulkanError>),
    #[error("The shader '{0}' is missing the entry point (function) '{1}")]
    MissingEntryPoint(&'static str, &'static str),
}
