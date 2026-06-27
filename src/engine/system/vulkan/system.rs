use crate::engine::system::vulkan::buffers::BasicBuffersManager;
use crate::engine::system::vulkan::desc::binding_101_window_size::WindowSize;
use crate::engine::system::vulkan::desc::binding_201_world_2d_view::World2dView;
use crate::engine::system::vulkan::desc::WriteDescriptorSetOrigin;
use crate::engine::system::vulkan::textures::{CopyInfo, ImageSystem};
use crate::engine::system::vulkan::utils::pipeline::single_pass_render_pass_from_image_format;
use crate::engine::system::vulkan::wds::WriteDescriptorSetManager;
use crate::engine::system::vulkan::{DrawError, Error};
use std::borrow::Borrow;
use std::iter::repeat_n;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::{Duration, Instant};
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
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage, SampleCount};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator, StandardMemoryAllocator};
use vulkano::pipeline::cache::PipelineCache;
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{
    acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::{Validated, Version, VulkanError};

pub struct VulkanSystem {
    device: Arc<Device>,
    render_queue: Arc<Queue>,
    transfer_queue: Vec<Arc<Queue>>,
    render_pass: Arc<RenderPass>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<Image>>,
    swapchain_framebuffers: Vec<Arc<Framebuffer>>,
    recreate_swapchain: bool,
    swapchain_is_new: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    write_descriptors: Arc<WriteDescriptorSetManager>,
    cmd_allocator: Arc<StandardCommandBufferAllocator>,
    image_system: Arc<ImageSystem>,
    basic_buffers_manager: Arc<BasicBuffersManager>,
    clear_value_rgba: [f32; 4],
    samples: SampleCount,
}

impl VulkanSystem {
    pub fn new(
        surface: Arc<Surface>,
        width: u32,
        height: u32,
        features: DeviceFeatures,
        samples: SampleCount,
    ) -> Result<Self, Error> {
        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_dynamic_rendering: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, index_graphics_queue, index_transfer_queues) =
            choose_physical_device(&surface, &mut device_extensions)?;

        info!(
            "Graphics Queue: {index_graphics_queue:?}, Transfer Queue(s): {index_transfer_queues:?}"
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures {
                    dynamic_rendering: true,
                    ..DeviceFeatures::empty()
                } | features,
                queue_create_infos: [(index_graphics_queue, NonZeroU32::new(1).unwrap())]
                    .into_iter()
                    .chain(index_transfer_queues.into_iter())
                    .enumerate()
                    .map(|(index, (queue_family_index, count))| QueueCreateInfo {
                        queue_family_index,
                        // the first queue (which is the render queue) gets the highest priority
                        queues: repeat_n(if index == 0 { 1.0 } else { 0.0 }, count.get() as usize)
                            .collect(),
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            },
        )
        .map_err(Error::DeviceInitializationFailed)?;

        let (swapchain, swapchain_images) =
            create_swapchain(&device, &surface, [width, height], samples)?;
        let render_pass = single_pass_render_pass_from_image_format(
            Arc::clone(&device),
            swapchain.image_format(),
            samples,
        )
        .map_err(Error::FailedToCreateFramebuffers)?;

        let basic_buffers_manager = Arc::new(BasicBuffersManager::new(
            StandardMemoryAllocator::new_default(Arc::clone(&device)),
        ));

        Self {
            image_system: Arc::new(ImageSystem::new(StandardMemoryAllocator::new_default(
                Arc::clone(&device),
            ))?),
            cmd_allocator: Arc::new(StandardCommandBufferAllocator::new(
                Arc::clone(&device),
                StandardCommandBufferAllocatorCreateInfo {
                    primary_buffer_count: 32,
                    secondary_buffer_count: 32,
                    ..StandardCommandBufferAllocatorCreateInfo::default()
                },
            )),
            render_queue: queues.next().expect("Promised queue is not present"),
            transfer_queue: queues
                .inspect(|queue| {
                    debug!(
                        "Found dedicated transfer_queue, family_index={}",
                        queue.queue_family_index()
                    );
                })
                .collect(),
            recreate_swapchain: false,
            swapchain_is_new: false,
            previous_frame_end: Some(vulkano::sync::now(Arc::clone(&device)).boxed()),
            swapchain_framebuffers: create_framebuffers(
                &basic_buffers_manager.memo_allocator,
                &swapchain_images,
                &render_pass,
                samples,
            )
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
            basic_buffers_manager,
            samples,
        }
        .with_write_descriptors_initialized()?
        .spawn_transfer_threads()
    }

    #[inline]
    fn with_write_descriptors_initialized(mut self) -> Result<Self, Error> {
        self.init_write_descriptors()?;
        Ok(self)
    }

    fn spawn_transfer_threads(self) -> Result<Self, Error> {
        let queues = if self.transfer_queue.is_empty() {
            info!("Using render_queue for transfer");
            vec![Arc::clone(&self.render_queue)]
        } else {
            info!(
                "Using {} dedicated transfer_queue threads(s)",
                self.transfer_queue.len()
            );
            self.transfer_queue.clone()
        };

        let render_queue_family_index = self.render_queue.queue_family_index();

        for (index, queue) in queues.into_iter().enumerate() {
            let device = Arc::clone(&self.device);
            let allocator = Arc::clone(&self.cmd_allocator) as Arc<dyn CommandBufferAllocator>;
            let image_system = Arc::clone(&self.image_system);

            let _ = std::thread::Builder::new()
                .name(format!("transfer_queue_{index}"))
                .spawn(move || {
                    info!(
                        "Started transfer_queue[{index}] thread on family_index={} (render_queue.family_index={render_queue_family_index})",
                        queue.queue_family_index(),
                    );

                    const MAX_BYTES: usize = 1024 * 1024 * 32;
                    const MAX_REQUESTS: usize = 1024;
                    const MAX_WAIT_IDLE: Duration = Duration::from_secs(1);
                    const MAX_WAIT_BUSY: Duration = Duration::from_millis(1);

                    let mut requests = Vec::with_capacity(MAX_REQUESTS);
                    let mut waiters = Vec::with_capacity(MAX_REQUESTS);

                    let mut stash = None;

                    loop {
                        let max_wait = if requests.is_empty() { MAX_WAIT_IDLE } else { MAX_WAIT_BUSY };
                        let (started, mut bytes) = stash.unwrap_or_else(|| (Instant::now(), 0));
                        let mut skip = 0;


                        'outer: while requests.len() < MAX_REQUESTS
                            && started.elapsed() < max_wait
                            && bytes < MAX_BYTES
                        {
                            while let Some(upload_request) =
                                image_system.wait_for_next_upload_info_until(started + max_wait)
                            {
                                let estimated_bytes = upload_request.estimated_bytes;
                                let required_before_render = upload_request.required_before_render;

                                if required_before_render {
                                    skip = requests.len();
                                    stash = Some((started, core::mem::take(&mut bytes)));
                                    requests.push(upload_request);
                                    bytes += estimated_bytes;
                                    break 'outer;
                                } else {
                                    requests.push(upload_request);
                                    bytes += estimated_bytes;
                                }

                                if requests.len() >= MAX_REQUESTS || (bytes + estimated_bytes) >= MAX_BYTES {
                                    break;
                                }
                            }
                        }

                        if requests.is_empty() {
                            continue;
                        } else {
                            info!("Handling {} transfer requests...", requests.len());
                        }

                        let mut buffer = AutoCommandBufferBuilder::primary(
                            Arc::clone(&allocator),
                            queue.queue_family_index(),
                            CommandBufferUsage::OneTimeSubmit,
                        )
                            .unwrap();

                        for request in requests.drain(skip..) {
                            let copy_info = match request.info {
                                CopyInfo::Deferred(eval) => {
                                    match eval(&image_system) {
                                        Ok(copy_info) =>copy_info,
                                        Err(e) => {
                                            error!("Failed to eval CopyInfo: {e:?}");
                                            continue;
                                        }
                                    }
                                },
                                CopyInfo::Immediate(copy_info) => copy_info,
                            };
                            if let Err(e) = buffer.copy_buffer_to_image(copy_info) {
                                error!("Failed to enqueue copy_buffer_to_image-cmd: {e}");
                            } else {
                                waiters.push(request.response);
                            }
                        }

                        // No successful copy request?
                        if waiters.is_empty() {
                            continue;
                        }


                        debug!("Submitting upload requests");
                        let commands = buffer.build().unwrap();
                        let future = vulkano::sync::now(Arc::clone(&device))
                            .then_execute(Arc::clone(&queue), commands)
                            .unwrap()
                            .then_signal_fence_and_flush()
                            .unwrap();

                        debug!("Awaiting upload completion");
                        future.wait(None).unwrap();

                        debug!("Notifying upload waiters...");
                        for waiter in waiters.drain(..) {
                            waiter.notify_completion();
                        }

                        debug!("Notifying upload waiters... DONE");
                    }
                });
        }

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

    fn update_write_descriptor_sets<T>(
        &self,
        cmds: &mut AutoCommandBufferBuilder<T>,
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
        &self.render_queue
    }

    #[inline]
    pub fn swapchain(&self) -> &Swapchain {
        &self.swapchain
    }

    #[inline]
    pub fn render_pass_(&self) -> &Arc<RenderPass> {
        &self.render_pass
    }

    pub fn graphics_pipeline_render_pass_info(&self) -> GraphicsPipelineRenderPassInfo {
        GraphicsPipelineRenderPassInfo(Arc::clone(&self.render_pass))
    }

    #[inline]
    pub fn pipeline_cache(&self) -> Option<&Arc<PipelineCache>> {
        info!("NO PipelineCache configured!");
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
                    self.swapchain_framebuffers = create_framebuffers(
                        &self.basic_buffers_manager.memo_allocator,
                        &self.swapchain_images,
                        &self.render_pass,
                        self.samples,
                    )
                    .map_err(DrawError::FailedToRecreateTheFramebuffers)?;
                    self.swapchain_is_new = true;
                }
                Err(e) => {
                    error!("{e}");
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
            Arc::clone(&self.cmd_allocator) as Arc<_>,
            self.render_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let context = RenderContext {
            render_queue_family_index: self.render_queue.queue_family_index(),
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

        let callback_commands = render_callback(&context);

        while let Some(waiter) = self.image_system.next_required_waiter() {
            waiter.wait_for_completion();
        }

        for command in callback_commands {
            if command.inheritance_info().render_pass.is_none() {
                prepare_commands.push(command);
            } else {
                render_commands.push(command);
            }
        }

        if let Err(e) = primary.execute_commands_from_vec(prepare_commands) {
            error!("Failed to execute preparation commands: {e:?}");
        }

        primary
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: if self.samples == SampleCount::Sample1 {
                        vec![Some(self.clear_value_rgba.into())]
                    } else {
                        vec![Some(self.clear_value_rgba.into()), None]
                    },
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
            error!("Failed to execute rendering commands: {e:?}");
        }

        primary.end_render_pass(SubpassEndInfo::default())?;
        let command_buffer = primary
            .build()
            .map_err(DrawError::FailedToBuildCommandBuffer)?;

        let future = match self.previous_frame_end.take() {
            None => vulkano::sync::now(Arc::clone(&self.device)).boxed(),
            Some(prev) => prev
                .join(vulkano::sync::now(Arc::clone(&self.device)))
                .boxed(),
        }
        .join(acquire_future)
        .then_execute(Arc::clone(&self.render_queue), command_buffer)
        .unwrap()
        .then_swapchain_present(
            Arc::clone(&self.render_queue),
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
                    Validated::Error(e) => error!("Error: {e}"),
                    Validated::ValidationError(e) => error!("Validation Error: {e}"),
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
) -> Result<(Arc<PhysicalDevice>, u32, Vec<(u32, NonZeroU32)>), Error> {
    surface
        .instance()
        .enumerate_physical_devices()
        .map_err(Error::FailedToEnumeratePhysicalDevices)?
        .filter(|p| {
            let dynamic =
                p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering;
            if dynamic {
                info!(
                    "Dynamic rendering supported on {}",
                    p.properties().device_name
                );
            } else {
                warn!(
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
                warn!(
                    "Device is missing required device extensions {}",
                    p.properties().device_name
                );
            }
            satisfies_req_device_extensions
        })
        .filter_map(|device| {
            let index_graphics_queue =
                device
                    .queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        info!("Queue({i}) = {q:?}");
                        q.queue_flags.contains(QueueFlags::GRAPHICS)
                            && device.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|index| index as u32)?;

            let index_transfer_queues =
                device
                    .queue_family_properties()
                    .iter()
                    .enumerate()
                    .filter(|(i, q)| {
                        info!("Queue({i}) = {q:?}");
                        q.queue_flags.contains(QueueFlags::TRANSFER) && *i as u32 != index_graphics_queue
                        && q.queue_count > 0
                    })
                    .filter_map(|(index, q)| {
                        Some((index as u32, NonZeroU32::new(q.queue_count)?))
                    })
                    .collect::<Vec<_>>();

            Some((device, index_graphics_queue, index_transfer_queues))
        })
        .min_by_key(|(p, ..)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .map(|(p, index_graphics_queue, index_transfer_queue)| {
            info!(
                "Chosen physical device {} and with queue family index {index_graphics_queue} and v{:?}",
                p.properties().device_name,
                p.api_version()
            );

            // // https://github.com/vulkano-rs/vulkano/blob/master/examples/src/bin/triangle-v1_3.rs#L181
            // if p.api_version() < Version::V1_3 {
            device_extensions.khr_dynamic_rendering = true;
            // }

            (p, index_graphics_queue, index_transfer_queue)
        })
        .ok_or(Error::NoSatisfyingPhysicalDevicePresent)
}

fn create_swapchain(
    device: &Arc<Device>,
    surface: &Arc<Surface>,
    image_extent: [u32; 2],
    samples: SampleCount,
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
            image_usage: if samples == SampleCount::Sample1 {
                ImageUsage::COLOR_ATTACHMENT
            } else {
                ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST
            },
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
    allocator: &Arc<dyn MemoryAllocator>,
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
    sample_count: SampleCount,
) -> Result<Vec<Arc<Framebuffer>>, Validated<VulkanError>> {
    images
        .iter()
        .map(|image| {
            Framebuffer::new(
                Arc::clone(&render_pass),
                if sample_count == SampleCount::Sample1 {
                    FramebufferCreateInfo {
                        attachments: vec![ImageView::new_default(Arc::clone(image))?],
                        ..FramebufferCreateInfo::default()
                    }
                } else {
                    FramebufferCreateInfo {
                        attachments: vec![
                            ImageView::new_default(
                                Image::new(
                                    Arc::clone(&allocator),
                                    ImageCreateInfo {
                                        image_type: ImageType::Dim2d,
                                        format: image.format(),
                                        extent: image.extent(),
                                        usage: ImageUsage::COLOR_ATTACHMENT
                                            | ImageUsage::TRANSIENT_ATTACHMENT,
                                        samples: sample_count,
                                        ..Default::default()
                                    },
                                    AllocationCreateInfo::default(),
                                )
                                .unwrap(),
                            )?,
                            ImageView::new_default(Arc::clone(image))?,
                        ],
                        ..FramebufferCreateInfo::default()
                    }
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()
}

pub struct RenderContext<'a> {
    render_queue_family_index: u32,
    renderpass: &'a Arc<RenderPass>,
    swapchain_framebuffer: &'a Arc<Framebuffer>,
    command_buffer_allocator: &'a Arc<StandardCommandBufferAllocator>,
    write_descriptor_set_manager: &'a WriteDescriptorSetManager,
    image_system: &'a Arc<ImageSystem>,
}

impl<'a> RenderContext<'a> {
    pub fn create_preparation_buffer_builder(
        &self,
    ) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, Error> {
        AutoCommandBufferBuilder::secondary(
            Arc::clone(self.command_buffer_allocator) as Arc<_>,
            self.render_queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
            CommandBufferInheritanceInfo {
                render_pass: None,
                occlusion_query: None,
                pipeline_statistics: Default::default(),
                ..CommandBufferInheritanceInfo::default()
            },
        )
        .map_err(Error::FailedToCreateCommandBuffer)
    }

    pub fn create_render_buffer_builder(
        &self,
    ) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, Error> {
        let mut secondary = AutoCommandBufferBuilder::secondary(
            Arc::clone(self.command_buffer_allocator) as Arc<_>,
            self.render_queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(CommandBufferInheritanceRenderPassType::BeginRenderPass(
                    CommandBufferInheritanceRenderPassInfo {
                        subpass: Subpass::from(Arc::clone(&self.renderpass), 0).unwrap(),
                        framebuffer: Some(Arc::clone(&self.swapchain_framebuffer)),
                    },
                )),
                occlusion_query: None,
                pipeline_statistics: Default::default(),
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
    pub fn update_write_descriptor_set<T, W: WriteDescriptorSetOrigin>(
        &self,
        cmds: &mut AutoCommandBufferBuilder<T>,
        origin: impl Borrow<W>,
    ) -> Result<Option<&WriteDescriptorSet>, Error> {
        self.write_descriptor_set_manager.update(cmds, origin)
    }

    #[inline]
    pub fn image_system(&self) -> &Arc<ImageSystem> {
        self.image_system
    }
}

#[derive(Clone)]
pub struct GraphicsPipelineRenderPassInfo(Arc<RenderPass>);

impl GraphicsPipelineRenderPassInfo {
    #[inline]
    pub fn render_pass(&self) -> &Arc<RenderPass> {
        &self.0
    }

    #[inline]
    pub fn into_subpass(self) -> Subpass {
        Subpass::from(self.0, 0).expect("There must always be at least one subpass")
    }

    #[inline]
    pub fn into_subpass_type(self) -> PipelineSubpassType {
        self.into_subpass().into()
    }

    #[inline]
    fn subpass(&self) -> Subpass {
        Subpass::from(Arc::clone(&self.0), 0).expect("There must always be at least one subpass")
    }

    #[inline]
    pub fn rasterization_samples(&self) -> SampleCount {
        self.subpass().num_samples().unwrap_or(SampleCount::Sample1)
    }

    #[inline]
    pub fn num_color_attachments(&self) -> u32 {
        self.subpass().num_color_attachments()
    }
}
