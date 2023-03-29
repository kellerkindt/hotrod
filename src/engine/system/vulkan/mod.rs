use std::sync::Arc;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceError, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceCreationError, DeviceExtensions, Features, Queue,
    QueueCreateInfo, QueueFlags,
};
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::Instance;
use vulkano::memory::allocator::{
    FreeListAllocator, GenericMemoryAllocator, StandardMemoryAllocator,
};
use vulkano::swapchain::{
    CompositeAlpha, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
};
use vulkano::sync::GpuFuture;
use vulkano::{Version, VulkanError};

pub struct VulkanSystem {
    surface: Arc<Surface>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<SwapchainImage>>,
    allocator: GenericMemoryAllocator<Arc<FreeListAllocator>>,
    recreate_swapchain: bool,
    previous_frame_end: Box<dyn GpuFuture>,
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
            queue: queues.next().expect("Promised queue is not present"),
            allocator: StandardMemoryAllocator::new_default(Arc::clone(&device)),
            recreate_swapchain: false,
            previous_frame_end: vulkano::sync::now(Arc::clone(&device)).boxed(),
            surface,
            device,
            swapchain,
            swapchain_images,
        })
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
}
