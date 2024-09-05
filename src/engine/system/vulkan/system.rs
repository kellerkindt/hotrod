use crate::engine::system::vulkan::buffers::BasicBuffersManager;
use crate::engine::system::vulkan::desc::binding_101_window_size::WindowSize;
use crate::engine::system::vulkan::desc::binding_201_world_2d_view::World2dView;
use crate::engine::system::vulkan::desc::WriteDescriptorSetOrigin;
use crate::engine::system::vulkan::textures::ImageSystem;
use crate::engine::system::vulkan::utils::pipeline::single_pass_render_pass_from_image_format;
use crate::engine::system::vulkan::wds::WriteDescriptorSetManager;
use crate::engine::system::vulkan::{DrawError, Error};
use std::borrow::Borrow;
use std::sync::Arc;
use std::time::Duration;
use vulkano::command_buffer::allocator::{
    CommandBufferAllocator, StandardCommandBufferAllocator,
    StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferInheritanceRenderPassInfo,
    CommandBufferInheritanceRenderPassType, CommandBufferUsage, RenderPassBeginInfo,
    SecondaryAutoCommandBuffer, SecondaryCommandBufferAbstract, SubpassBeginInfo, SubpassContents,
    SubpassEndInfo,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageUsage};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::cache::PipelineCache;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{
    acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::{Validated, Version, VulkanError};

pub struct VulkanSystem {
    device: Arc<Device>,
    queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<Image>>,
    swapchain_framebuffers: Vec<Arc<Framebuffer>>,
    recreate_swapchain: bool,
    swapchain_is_new: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    write_descriptors: Arc<WriteDescriptorSetManager>,
    cmd_allocator: StandardCommandBufferAllocator,
    image_system: Arc<ImageSystem>,
    basic_buffers_manager: Arc<BasicBuffersManager>,
    clear_value_rgba: [f32; 4],
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

        Self {
            image_system: Arc::new(ImageSystem::new(StandardMemoryAllocator::new_default(
                Arc::clone(&device),
            ))?),
            basic_buffers_manager: Arc::new(BasicBuffersManager::new(
                StandardMemoryAllocator::new_default(Arc::clone(&device)),
            )),
            cmd_allocator: StandardCommandBufferAllocator::new(
                Arc::clone(&device),
                StandardCommandBufferAllocatorCreateInfo {
                    primary_buffer_count: 32,
                    secondary_buffer_count: 32,
                    ..StandardCommandBufferAllocatorCreateInfo::default()
                },
            ),
            queue: queues.next().expect("Promised queue is not present"),
            recreate_swapchain: false,
            swapchain_is_new: false,
            previous_frame_end: Some(vulkano::sync::now(Arc::clone(&device)).boxed()),
            swapchain_framebuffers: create_framebuffers(&swapchain_images, &render_pass)
                .map_err(Error::FailedToCreateFramebuffers)?,
            swapchain,
            swapchain_images,
            render_pass,
            write_descriptors: Arc::new(WriteDescriptorSetManager::new(
                Arc::new(StandardDescriptorSetAllocator::new(
                    Arc::clone(&device),
                    StandardDescriptorSetAllocatorCreateInfo::default(),
                )),
                Arc::new(StandardMemoryAllocator::new_default(Arc::clone(&device))),
            )),
            device,
            clear_value_rgba: [0.0, 0.5, 1.0, 1.0], // blue-ish value
        }
        .with_write_descriptors_initialized()
    }

    #[inline]
    fn with_write_descriptors_initialized(mut self) -> Result<Self, Error> {
        self.init_write_descriptors()?;
        Ok(self)
    }

    fn init_write_descriptors(&mut self) -> Result<(), Error> {
        // clone to not re-create allocators
        let mut write_descriptor = WriteDescriptorSetManager::new(
            Arc::clone(self.write_descriptors.descriptor_set_allocator()),
            Arc::clone(self.write_descriptors.memory_allocator()),
        );

        write_descriptor.insert(WindowSize::from(&*self))?;
        write_descriptor.insert(World2dView::from(&*self))?;

        self.write_descriptors = Arc::new(write_descriptor);
        Ok(())
    }

    fn update_write_descriptor_sets<T, A: CommandBufferAllocator>(
        &self,
        cmds: &mut AutoCommandBufferBuilder<T, A>,
    ) -> Result<(), Error> {
        self.write_descriptors
            .update(cmds, WindowSize::from(self))?;

        Ok(())
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
    pub fn swapchain(&self) -> &Swapchain {
        &self.swapchain
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
    pub fn image_system(&self) -> &Arc<ImageSystem> {
        &self.image_system
    }

    #[inline]
    pub fn write_descriptor_set_manager(&self) -> &Arc<WriteDescriptorSetManager> {
        &self.write_descriptors
    }

    #[inline]
    pub fn basic_buffers_manager(&self) -> &Arc<BasicBuffersManager> {
        &self.basic_buffers_manager
    }

    #[inline]
    pub fn recreate_swapchain(&mut self) {
        self.recreate_swapchain = true;
    }

    #[inline]
    pub fn clear_value(&self) -> [f32; 4] {
        self.clear_value_rgba
    }

    #[inline]
    pub fn set_clear_value(&mut self, rgba: [f32; 4]) {
        self.clear_value_rgba = rgba;
    }

    // TODO just for demo
    pub fn render<F1>(
        &mut self,
        width: u32,
        height: u32,
        render_callback: F1,
    ) -> Result<(), DrawError>
    where
        F1: FnOnce(&RenderContext) -> Vec<Arc<SecondaryAutoCommandBuffer>>,
    {
        if core::mem::take(&mut self.recreate_swapchain) {
            match self.swapchain.recreate(SwapchainCreateInfo {
                image_extent: [width, height],
                ..self.swapchain.create_info()
            }) {
                Ok((new_swapchain, new_image)) => {
                    self.swapchain = new_swapchain;
                    self.swapchain_images = new_image;
                    self.swapchain_framebuffers =
                        create_framebuffers(&self.swapchain_images, &self.render_pass)
                            .map_err(DrawError::FailedToRecreateTheFramebuffers)?;
                    self.swapchain_is_new = true;
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

        let (swapchain_image_index, suboptimal, acquire_future) =
            match acquire_next_image(Arc::clone(&self.swapchain), Some(Duration::from_secs(1))) {
                Ok(ok) => Ok(ok),
                Err(Validated::Error(VulkanError::Timeout)) => {
                    return Err(DrawError::AcquiringSwapchainImageReachedTimeout)
                }
                e => e,
            }
            .unwrap();

        if suboptimal {
            self.recreate_swapchain = true;
        }

        let mut primary = AutoCommandBufferBuilder::primary(
            &self.cmd_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let context = RenderContext {
            queue_family_index: self.queue.queue_family_index(),
            renderpass: &self.render_pass,
            swapchain_framebuffer: &self.swapchain_framebuffers[swapchain_image_index as usize],
            command_buffer_allocator: &self.cmd_allocator,
            write_descriptor_set_manager: &self.write_descriptors,
            image_system: &self.image_system,
        };

        let mut prepare_commands: Vec<Arc<dyn SecondaryCommandBufferAbstract>> = Vec::new();
        let mut render_commands: Vec<Arc<dyn SecondaryCommandBufferAbstract>> = Vec::new();

        acquire_future
            .wait(Some(Duration::from_secs(10)))
            .map_err(DrawError::FailedToAcquireSwapchainImage)?;
        if let Some(previous) = self.previous_frame_end.as_mut() {
            previous.cleanup_finished();
        }

        if core::mem::take(&mut self.swapchain_is_new) {
            let mut buffer = context
                .create_preparation_buffer_builder()
                .expect("Failed to create preparation command buffer for descriptor updates");
            self.update_write_descriptor_sets(&mut buffer)
                .expect("Failed to update write descriptor sets");
            prepare_commands.push(
                buffer
                    .build()
                    .expect("Failed to build command buffer for descriptor updates"),
            );
        }

        for command in render_callback(&context) {
            if command.inheritance_info().render_pass.is_none() {
                prepare_commands.push(command);
            } else {
                render_commands.push(command);
            }
        }

        if let Err(e) = primary.execute_commands_from_vec(prepare_commands) {
            eprintln!("Failed to execute preparation commands: {e:?}");
        }

        primary
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some(self.clear_value_rgba.into())],
                    // clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(Arc::clone(
                        &self.swapchain_framebuffers[swapchain_image_index as usize],
                    ))
                },
                SubpassBeginInfo {
                    contents: SubpassContents::SecondaryCommandBuffers,
                    ..SubpassBeginInfo::default()
                },
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

        if let Err(e) = primary.execute_commands_from_vec(render_commands) {
            eprintln!("Failed to execute rendering commands: {e:?}");
        }

        primary.end_render_pass(SubpassEndInfo::default())?;
        let command_buffer = primary
            .build()
            .map_err(DrawError::FailedToBuildCommandBuffer)?;

        let future = self
            .previous_frame_end
            .take()
            .unwrap_or_else(|| vulkano::sync::now(Arc::clone(&self.device)).boxed())
            .join(acquire_future)
            .then_execute(Arc::clone(&self.queue), command_buffer)
            .unwrap()
            .then_swapchain_present(
                Arc::clone(&self.queue),
                SwapchainPresentInfo::swapchain_image_index(
                    Arc::clone(&self.swapchain),
                    swapchain_image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(e) => {
                match e {
                    Validated::Error(VulkanError::OutOfDate) => {}
                    Validated::Error(e) => eprintln!("Error: {e}"),
                    Validated::ValidationError(e) => eprintln!("Validation Error: {e}"),
                }
                self.recreate_swapchain = true;
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
        .map_err(Error::FailedToRetrieveSurfaceFormats)?
        .iter()
        .find(|(format, _color_space)| {
            [
                Format::R8G8B8_SRGB,
                Format::R8G8B8A8_SRGB,
                Format::B8G8R8_SRGB,
                Format::B8G8R8A8_SRGB,
            ]
            .contains(format)
        })
        .expect("Did not find a suitable color space")
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

pub struct RenderContext<'a> {
    queue_family_index: u32,
    renderpass: &'a Arc<RenderPass>,
    swapchain_framebuffer: &'a Arc<Framebuffer>,
    command_buffer_allocator: &'a StandardCommandBufferAllocator,
    write_descriptor_set_manager: &'a WriteDescriptorSetManager,
    image_system: &'a ImageSystem,
}

impl<'a> RenderContext<'a> {
    pub fn create_preparation_buffer_builder(
        &self,
    ) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, Error> {
        AutoCommandBufferBuilder::secondary(
            self.command_buffer_allocator,
            self.queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
            CommandBufferInheritanceInfo {
                render_pass: None,
                occlusion_query: None,
                query_statistics_flags: Default::default(),
                ..CommandBufferInheritanceInfo::default()
            },
        )
        .map_err(Error::FailedToCreateCommandBuffer)
    }

    pub fn create_render_buffer_builder(
        &self,
    ) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, Error> {
        let mut secondary = AutoCommandBufferBuilder::secondary(
            self.command_buffer_allocator,
            self.queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(CommandBufferInheritanceRenderPassType::BeginRenderPass(
                    CommandBufferInheritanceRenderPassInfo {
                        subpass: Subpass::from(Arc::clone(&self.renderpass), 0).unwrap(),
                        framebuffer: Some(Arc::clone(&self.swapchain_framebuffer)),
                    },
                )),
                occlusion_query: None,
                query_statistics_flags: Default::default(),
                ..CommandBufferInheritanceInfo::default()
            },
        )
        .map_err(Error::FailedToCreateCommandBuffer)?;
        secondary
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [
                        self.swapchain_framebuffer.extent()[0] as f32,
                        self.swapchain_framebuffer.extent()[1] as f32,
                    ],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .expect("Using the Swapchain extents should never fail");
        Ok(secondary)
    }

    #[inline]
    pub fn update_write_descriptor_set<
        T,
        A: CommandBufferAllocator,
        W: WriteDescriptorSetOrigin,
    >(
        &self,
        cmds: &mut AutoCommandBufferBuilder<T, A>,
        origin: impl Borrow<W>,
    ) -> Result<Option<&WriteDescriptorSet>, Error> {
        self.write_descriptor_set_manager.update(cmds, origin)
    }

    #[inline]
    pub fn image_system(&self) -> &ImageSystem {
        self.image_system
    }
}
