use crate::engine::builder::EngineBuilder;
use crate::engine::parts::sdl::SdlParts;
use crate::engine::parts::vulkan::VulkanParts;
use crate::engine::system::vulkan::VulkanSystem;
use sdl2::video::WindowBuildError;
use std::sync::Arc;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use vulkano::instance::{Instance, InstanceCreationError, InstanceExtensions};
use vulkano::swapchain::{Surface, SurfaceApi};
use vulkano::{Handle, LoadingError, VulkanLibrary, VulkanObject};

pub mod builder;
pub mod parts;
pub mod system;

pub struct Engine {
    sdl: SdlParts,
    vulkan: VulkanParts,
    vulkan_system: VulkanSystem,
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
        let surface = Arc::new(unsafe {
            Surface::from_handle(
                Arc::clone(&instance),
                <_ as Handle>::from_raw(surface_handle),
                SurfaceApi::Xlib,
                None,
            )
        });

        Ok(Self {
            sdl: SdlParts {
                context,
                video_subsystem,
                event_pump,
                window,
            },
            vulkan: VulkanParts {
                instance,
                surface: Arc::clone(&surface),
                surface_handle,
            },
            vulkan_system: VulkanSystem::new(
                surface,
                [builder.window_width, builder.window_height],
            )?,
        })
    }

    pub fn run(mut self, mut f: impl FnMut()) -> Self {
        'running: loop {
            f();
            for event in self.sdl.event_pump.poll_iter() {
                match event {
                    Event::Quit { .. } | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                        break 'running;
                    }
                    _ => {}
                }
            }
            let (widht, height) = self.sdl.window.drawable_size();
            self.vulkan_system.render(widht, height);
            ::std::thread::sleep(::std::time::Duration::new(0, 1_000_000_000u32 / 60));
        }
        self
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
    #[error("Error in vulkan subsystem: {0}")]
    VulkanSubsystemError(#[from] system::vulkan::Error),
}
