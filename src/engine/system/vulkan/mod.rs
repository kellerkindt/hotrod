use std::sync::Arc;
use std::time::Duration;
use vulkano::buffer::{BufferAllocateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
    RenderingAttachmentInfo, RenderingInfo,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceError, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceCreationError, DeviceExtensions, Features, Queue,
    QueueCreateInfo, QueueFlags,
};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageAccess, ImageUsage, SwapchainImage};
use vulkano::memory::allocator::{
    FreeListAllocator, GenericMemoryAllocator, StandardMemoryAllocator,
};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::render_pass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{LoadOp, RenderPass, RenderPassCreationError, StoreOp, Subpass};
use vulkano::swapchain::{
    acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    SwapchainPresentInfo,
};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::{Version, VulkanError};

#[cfg(feature = "ui-egui")]
pub mod egui;

pub struct VulkanSystem {
    surface: Arc<Surface>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<SwapchainImage>>,
    allocator: GenericMemoryAllocator<Arc<FreeListAllocator>>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    render_pass: Arc<RenderPass>,
    sub_pass_counter: u32,
}

impl VulkanSystem {
    pub fn new(surface: Arc<Surface>, image_extent: [u32; 2]) -> Result<Self, Error> {
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
                },
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )?;

        let (swapchain, swapchain_images) = create_swapchain(&device, &surface, image_extent)?;

        Ok(Self {
            render_pass: vulkano::single_pass_renderpass!(
                    Arc::clone(&device),
                     attachments: {
                // `color` is a custom name we give to the first and only attachment.
                color: {
                    // `load: Clear` means that we ask the GPU to clear the content of this attachment
                    // at the start of the drawing.
                    load: Clear,
                    // `store: Store` means that we ask the GPU to store the output of the draw in the
                    // actual image. We could also ask it to discard the result.
                    store: Store,
                    // `format: <ty>` indicates the type of the format of the image. This has to be one
                    // of the types of the `vulkano::format` module (or alternatively one of your
                    // structs that implements the `FormatDesc` trait). Here we use the same format as
                    // the swapchain.
                    format: swapchain.image_format(),
                    // `samples: 1` means that we ask the GPU to use one sample to determine the value
                    // of each pixel in the color attachment. We could use a larger value
                    // (multisampling) for antialiasing. An example of this can be found in
                    // msaa-renderpass.rs.
                    samples: 1,
                },
            },
            pass: {
                // We use the attachment named `color` as the one and only color attachment.
                color: [color],
                // No depth-stencil attachment is indicated with empty brackets.
                depth_stencil: {},
            },
                )?,
            sub_pass_counter: 0,
            queue: queues.next().expect("Promised queue is not present"),
            allocator: StandardMemoryAllocator::new_default(Arc::clone(&device)),
            recreate_swapchain: false,
            previous_frame_end: Some(vulkano::sync::now(Arc::clone(&device)).boxed()),
            surface,
            device,
            swapchain,
            swapchain_images,
        })
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    pub fn create_subpass(&mut self) -> Subpass {
        let subpass = Subpass::from(Arc::clone(&self.render_pass), self.sub_pass_counter);
        self.sub_pass_counter += 1;
        subpass.unwrap()
    }

    // TODO just for demo
    pub fn render<F1, F2>(&mut self, width: u32, height: u32, before_render: F1)
    where
        F1: FnOnce(&mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) -> F2,
        F2: FnOnce(&mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>),
    {
        // We now create a buffer that will store the shape of our triangle.
        // We use #[repr(C)] here to force rustc to not do anything funky with our data, although for this
        // particular example, it doesn't actually change the in-memory representation.
        use bytemuck::{Pod, Zeroable};
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, Zeroable, Pod, Vertex)]
        struct Vertex {
            #[format(R32G32_SFLOAT)]
            position: [f32; 2],
        }

        let vertices = [
            Vertex {
                position: [-0.5, -0.25],
            },
            Vertex {
                position: [0.0, 0.5],
            },
            Vertex {
                position: [0.25, -0.1],
            },
        ];
        let vertex_buffer = vulkano::buffer::Buffer::from_iter(
            &self.allocator,
            BufferAllocateInfo {
                buffer_usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: "
				#version 450
				layout(location = 0) in vec2 position;
				void main() {
					gl_Position = vec4(position, 0.0, 1.0);
				}
			"
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: "
				#version 450
				layout(location = 0) out vec4 f_color;
				void main() {
					f_color = vec4(1.0, 0.0, 0.0, 1.0);
				}
			"
            }
        }

        let vs = vs::load(self.device.clone()).unwrap();
        let fs = fs::load(self.device.clone()).unwrap();

        let pipeline = GraphicsPipeline::start()
            .render_pass(PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(self.swapchain.image_format())],
                ..Default::default()
            })
            .vertex_input_state(Vertex::per_vertex())
            .input_assembly_state(InputAssemblyState::new())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .build(Arc::clone(&self.device))
            .unwrap();

        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [0.0, 0.0],
            depth_range: 0.0..1.0,
        };

        let attachment_image_views =
            window_size_dependent_setup(&self.swapchain_images, &mut viewport);

        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(Arc::clone(&self.device), Default::default());

        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if core::mem::take(&mut self.recreate_swapchain) {
            let (new_swapchain, new_image) = match self.swapchain.recreate(SwapchainCreateInfo {
                image_extent: dbg!([width, height]),
                ..self.swapchain.create_info()
            }) {
                Ok(ok) => ok,
                Err(e) => {
                    eprintln!("{e}");
                    eprintln!("{e:?}");
                    panic!()
                }
            };
            self.swapchain = new_swapchain;
            self.swapchain_images = new_image;
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
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    // `Clear` means that we ask the GPU to clear the content of this
                    // attachment at the start of rendering.
                    load_op: LoadOp::Clear,
                    // `Store` means that we ask the GPU to store the rendered output
                    // in the attachment image. We could also ask it to discard the result.
                    store_op: StoreOp::Store,
                    // The value to clear the attachment with. Here we clear it with a
                    // blue color.
                    //
                    // Only attachments that have `LoadOp::Clear` are provided with
                    // clear values, any others should use `None` as the clear value.
                    clear_value: Some([0.0, 0.5, 1.0, 1.0].into()),
                    ..RenderingAttachmentInfo::image_view(
                        // We specify image view corresponding to the currently acquired
                        // swapchain image, to use for this attachment.
                        attachment_image_views[image_index as usize].clone(),
                    )
                })],
                ..Default::default()
            })
            .unwrap()
            .set_viewport(0, [viewport.clone()])
            .bind_pipeline_graphics(pipeline)
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();

        inside_render(&mut builder);

        builder.end_rendering().unwrap();

        let command_buffer = builder.build().unwrap();
        eprintln!("5");

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

        eprintln!("6");

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
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

        fn window_size_dependent_setup(
            images: &[Arc<SwapchainImage>],
            viewport: &mut Viewport,
        ) -> Vec<Arc<ImageView<SwapchainImage>>> {
            let dimensions = images[0].dimensions().width_height();
            viewport.dimensions = dbg!([dimensions[0] as f32, dimensions[1] as f32]);

            images
                .iter()
                .map(|image| ImageView::new_default(image.clone()).unwrap())
                .collect::<Vec<_>>()
        }
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
) -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), Error> {
    let surface_capabilities = device
        .physical_device()
        .surface_capabilities(&surface, Default::default())
        .map_err(Error::FailedToRetrieveSurfaceCapabilities)?;

    let image_format = Some(
        device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .map_err(Error::FailedToRetrieveSurfaceFormats)?[0]
            .0,
    );

    Ok(Swapchain::new(
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
    )?)
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
    DeviceInitializationFailed(#[from] DeviceCreationError),
    #[error("Failed to initialize swapchain instance {0:?}")]
    SwapchainInitializationFailed(#[from] SwapchainCreationError),
    #[error("Failed to retrieve surface capabilities: {0:?}")]
    FailedToRetrieveSurfaceCapabilities(PhysicalDeviceError),
    #[error("Failed to retrieve surface formats: {0:?}")]
    FailedToRetrieveSurfaceFormats(PhysicalDeviceError),
    #[error("Failed to create render pass: {0:?}")]
    RenderPassCreationError(#[from] RenderPassCreationError),
}
