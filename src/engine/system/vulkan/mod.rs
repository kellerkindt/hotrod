use crate::engine::system::vulkan::utils::pipeline::single_pass_render_pass_from_image_format;
use std::sync::Arc;
use std::time::Duration;
use vulkano::buffer::BufferAllocateError;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassEndInfo,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageAllocateError, ImageUsage};
use vulkano::pipeline::cache::PipelineCache;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::layout::IntoPipelineLayoutCreateInfoError;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::swapchain::{
    acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::{Validated, ValidationError, Version, VulkanError};

pub mod utils;

pub mod beautiful_lines;
#[cfg(feature = "ui-egui")]
pub mod egui;
pub mod lines;
pub mod textures;

pub struct VulkanSystem {
    device: Arc<Device>,
    queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<Image>>,
    swapchain_framebuffers: Vec<Arc<Framebuffer>>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl VulkanSystem {
    pub fn new(
        surface: Arc<Surface>,
        width: u32,
        height: u32,
        features: Features,
    ) -> Result<Self, Error> {
        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_dynamic_rendering: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            choose_physical_device(&surface, &mut device_extensions)?;

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: Features {
                    dynamic_rendering: true,
                    ..Features::empty()
                } | features,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .map_err(Error::DeviceInitializationFailed)?;

        let (swapchain, swapchain_images) = create_swapchain(&device, &surface, [width, height])?;
        let render_pass = single_pass_render_pass_from_image_format(
            Arc::clone(&device),
            swapchain.image_format(),
        )
        .map_err(Error::FailedToCreateFramebuffers)?;

        Ok(Self {
            queue: queues.next().expect("Promised queue is not present"),
            recreate_swapchain: false,
            previous_frame_end: Some(vulkano::sync::now(Arc::clone(&device)).boxed()),
            device,
            swapchain_framebuffers: create_framebuffers(&swapchain_images, &render_pass)
                .map_err(Error::FailedToCreateFramebuffers)?,
            swapchain,
            swapchain_images,
            render_pass,
        })
    }

    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    #[inline]
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    #[inline]
    pub fn image_format(&self) -> Format {
        self.swapchain.image_format()
    }

    #[inline]
    pub fn render_pass(&self) -> &Arc<RenderPass> {
        &self.render_pass
    }

    #[inline]
    pub fn pipeline_cache(&self) -> Option<&Arc<PipelineCache>> {
        eprintln!("NO PipelineCache configured!");
        None
    }

    #[inline]
    pub fn recreatee_swapchain(&mut self) {
        self.recreate_swapchain = true;
    }

    // TODO just for demo
    pub fn render<F1, F2>(
        &mut self,
        width: u32,
        height: u32,
        before_render: F1,
    ) -> Result<(), DrawError>
    where
        F1: FnOnce(&mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) -> F2,
        F2: FnOnce(&mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>),
    {
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(Arc::clone(&self.device), Default::default());

        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if core::mem::take(&mut self.recreate_swapchain) {
            match self.swapchain.recreate(SwapchainCreateInfo {
                image_extent: [width, height],
                ..self.swapchain.create_info()
            }) {
                Ok((new_swapchain, new_image)) => {
                    eprintln!("Swapchain re-recreated");
                    self.swapchain = new_swapchain;
                    self.swapchain_images = new_image;
                    self.swapchain_framebuffers =
                        create_framebuffers(&self.swapchain_images, &self.render_pass)
                            .map_err(DrawError::FailedToRecreateTheFramebuffers)?;
                }
                Err(e) => {
                    eprintln!("{e}");
                    eprintln!("{e:?}");
                    // try again
                    self.recreate_swapchain = true;
                    return Ok(());
                    // panic!()
                }
            }
        }

        let (image_index, suboptimal, acquire_future) =
            acquire_next_image(Arc::clone(&self.swapchain), Some(Duration::from_secs(1))).unwrap();

        if suboptimal {
            self.recreate_swapchain = true;
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let inside_render = before_render(&mut builder);
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.5, 1.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(Arc::clone(
                        &self.swapchain_framebuffers[image_index as usize],
                    ))
                },
                SubpassBeginInfo::default(),
            )?
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [
                        self.swapchain_images[0].extent()[0] as f32,
                        self.swapchain_images[0].extent()[1] as f32,
                    ],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )?;

        inside_render(&mut builder);
        builder.end_render_pass(SubpassEndInfo::default())?;

        let command_buffer = builder
            .build()
            .map_err(DrawError::FailedToBuildCommandBuffer)?;

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(Arc::clone(&self.queue), command_buffer)
            .unwrap()
            .then_swapchain_present(
                Arc::clone(&self.queue),
                SwapchainPresentInfo::swapchain_image_index(
                    Arc::clone(&self.swapchain),
                    image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end =
                    Some(vulkano::sync::now(Arc::clone(&self.device)).boxed());
            }
            Err(e) => {
                eprintln!("Failed to flush future: {e:?}");
                self.previous_frame_end =
                    Some(vulkano::sync::now(Arc::clone(&self.device)).boxed());
            }
        }

        Ok(())
    }
}

fn choose_physical_device(
    surface: &Surface,
    device_extensions: &mut DeviceExtensions,
) -> Result<(Arc<PhysicalDevice>, u32), Error> {
    surface
        .instance()
        .enumerate_physical_devices()
        .map_err(Error::FailedToEnumeratePhysicalDevices)?
        .filter(|p| {
            let dynamic =
                p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering;
            if dynamic {
                eprintln!(
                    "Dynamic rendering supported on {}",
                    p.properties().device_name
                );
            } else {
                eprintln!(
                    "Dynamic rendering not supported on {}",
                    p.properties().device_name
                );
            }
            dynamic
        })
        .filter(|p| {
            let satisfies_req_device_extensions =
                p.supported_extensions().contains(&device_extensions);
            if !satisfies_req_device_extensions {
                eprintln!(
                    "Device is missing required device extensions {}",
                    p.properties().device_name
                );
            }
            satisfies_req_device_extensions
        })
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .map(|(p, i)| {
            eprintln!(
                "Chosen physical device {} and with queue family index {i} and v{:?}",
                p.properties().device_name,
                p.api_version()
            );

            // // https://github.com/vulkano-rs/vulkano/blob/master/examples/src/bin/triangle-v1_3.rs#L181
            // if p.api_version() < Version::V1_3 {
            device_extensions.khr_dynamic_rendering = true;
            // }

            (p, i)
        })
        .ok_or(Error::NoSatisfyingPhysicalDevicePresent)
}

fn create_swapchain(
    device: &Arc<Device>,
    surface: &Arc<Surface>,
    image_extent: [u32; 2],
) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>), Error> {
    let surface_capabilities = device
        .physical_device()
        .surface_capabilities(&surface, Default::default())
        .map_err(Error::FailedToRetrieveSurfaceCapabilities)?;

    let image_format = device
        .physical_device()
        .surface_formats(&surface, Default::default())
        .map_err(Error::FailedToRetrieveSurfaceFormats)?[0]
        .0;

    Swapchain::new(
        Arc::clone(&device),
        Arc::clone(&surface),
        SwapchainCreateInfo {
            min_image_count: surface_capabilities.min_image_count,
            image_format,
            image_extent,
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha: surface_capabilities
                .supported_composite_alpha
                .into_iter()
                .next()
                .unwrap(),
            ..Default::default()
        },
    )
    .map_err(Error::SwapchainInitializationFailed)
}

fn create_framebuffers(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
) -> Result<Vec<Arc<Framebuffer>>, Validated<VulkanError>> {
    images
        .iter()
        .map(|image| {
            Framebuffer::new(
                Arc::clone(&render_pass),
                FramebufferCreateInfo {
                    attachments: vec![ImageView::new_default(Arc::clone(image))?],
                    ..FramebufferCreateInfo::default()
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()
}

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
}

#[derive(thiserror::Error, Debug)]
pub enum DrawError {
    // #[error("Vulkan Error: {0}")]
    // VulkanError(#[from] Validated<VulkanError>),
    #[error("Validation Error: {0}")]
    ValidationError(#[from] Box<ValidationError>),
    #[error("Failed to allocate buffer: {0}")]
    BufferAllocateError(#[from] Validated<BufferAllocateError>),
    #[error("Failed to re-create the framebuffers: {0}")]
    FailedToRecreateTheFramebuffers(Validated<VulkanError>),
    // #[error("Failed to execute the pipeline: {0}")]
    // PipelineExecutionError(#[from] Validated<VulkanError>),
    #[error("Failed to build command buffer: {0}")]
    FailedToBuildCommandBuffer(Validated<VulkanError>),
}

#[derive(thiserror::Error, Debug)]
pub enum UploadError {
    #[error("Vulkan Error: {0}")]
    VulkanError(#[from] Validated<VulkanError>),
    #[error("Failed to upload the image: {0}")]
    ImageError(#[from] Validated<ImageAllocateError>),
    #[error("Failed to allocate buffer: {0}")]
    BufferAllocateError(#[from] Validated<BufferAllocateError>),
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
}

#[derive(thiserror::Error, Debug)]
pub enum ShaderLoadError {
    #[error("Vulkan Error: {0}")]
    VulkanError(#[from] Validated<VulkanError>),
    #[error("The shader '{0}' is missing the entry point (function) '{1}")]
    MissingEntryPoint(&'static str, &'static str),
}
