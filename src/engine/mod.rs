use crate::engine::builder::EngineBuilder;
use sdl2::video::WindowBuildError;
use std::sync::Arc;
use vulkano::instance::{Instance, InstanceCreationError, InstanceExtensions};
use vulkano::swapchain::{Surface, SurfaceApi};
use vulkano::{Handle, LoadingError, VulkanLibrary, VulkanObject};
use crate::engine::part::sdl::SdlPart;
use crate::engine::part::vulkan::VulkanPart;

pub mod builder;
pub mod part;

pub struct Engine {
    sdl: SdlPart,
    vulkan: VulkanPart,
}

impl Engine {
    pub fn new(builder: EngineBuilder) -> Result<Self, Error> {
        let context = sdl2::init().map_err(Error::SdlError)?;
        let video_subsystem = context.video().map_err(Error::SdlError)?;
        let event_pump = context.event_pump().map_err(Error::SdlError)?;

        let window = video_subsystem
            .window(
                builder.window_title.as_ref(),
                builder.window_width,
                builder.window_height,
            )
            .vulkan()
            .build()
            .map_err(Error::SdlWindowBuildError)?;

        let instance_extensions = InstanceExtensions::from_iter(
            window
                .vulkan_instance_extensions()
                .map_err(Error::SdlError)?,
        );

        let instance = Instance::new(VulkanLibrary::new()?, {
            let mut instance_info = builder.instance_info;
            instance_info.enabled_extensions = instance_extensions;
            instance_info
        })?;

        let surface_handle = window
            .vulkan_create_surface(instance.handle().as_raw() as _)
            .map_err(Error::SdlCreateVulkanSurfaceError)?;

        // SAFETY: that's the way it is
        let surface = unsafe {
            Surface::from_handle(
                Arc::clone(&instance),
                <_ as Handle>::from_raw(surface_handle),
                SurfaceApi::Xlib,
                None,
            )
        };


        Ok(Self {
            sdl: SdlPart {
                context,
                video_subsystem,
                event_pump,
                window,
            },
            vulkan: VulkanPart {
                instance,
                surface_handle,
                surface,
            },
        })
    }
}

impl Default for Engine {
    #[inline]
    fn default() -> Self {
        EngineBuilder::default()
            .build()
            .expect("Failed to build with default configuration")
    }
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("SDL error: {0}")]
    SdlError(String),
    #[error("SDL failed to create the window: {0}")]
    SdlWindowBuildError(#[from] WindowBuildError),
    #[error("SDL failed to create a vulkan surface: {0}")]
    SdlCreateVulkanSurfaceError(String),
    #[error("Failed to create a Vulkan instance: {0}")]
    VulkanInstanceCreationError(#[from] InstanceCreationError),
    #[error("Failed to load the vulkan library {0}")]
    VulkanLibraryLoadingError(#[from] LoadingError),
}
