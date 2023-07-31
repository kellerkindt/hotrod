use crate::engine::builder::EngineBuilder;
use crate::engine::parts::sdl::SdlParts;
use crate::engine::parts::vulkan::VulkanParts;
use crate::engine::system::vulkan::beautiful_lines::{Line, Vertex2d, VulkanBeautifulLineSystem};
use crate::engine::system::vulkan::VulkanSystem;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::video::WindowBuildError;
use std::ops::Mul;
use std::sync::Arc;
use std::time::{Instant, UNIX_EPOCH};
use vulkano::instance::{Instance, InstanceCreationError, InstanceExtensions};
use vulkano::pipeline::graphics::GraphicsPipelineCreationError;
use vulkano::swapchain::{Surface, SurfaceApi};
use vulkano::{Handle, LoadingError, VulkanLibrary, VulkanObject};

pub mod builder;
pub mod parts;
pub mod system;

pub struct Engine {
    sdl: SdlParts,
    vulkan: VulkanParts,
    vulkan_system: VulkanSystem,
    vulkan_lines: VulkanBeautifulLineSystem,
    #[cfg(feature = "ui-egui")]
    egui_system: system::vulkan::egui::EguiSystem,
    #[cfg(feature = "ui-egui")]
    egui_parts: parts::egui::EguiParts,
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
            .resizable()
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

        let vulkan = VulkanParts {
            instance,
            surface: Arc::clone(&surface),
            surface_handle,
        };
        let vulkan_system = VulkanSystem::new(
            surface,
            builder.window_width,
            builder.window_height,
            VulkanBeautifulLineSystem::REQUIRED_FEATURES,
        )?;

        Ok(Self {
            sdl: SdlParts {
                context,
                video_subsystem,
                event_pump,
                window,
            },
            #[cfg(feature = "ui-egui")]
            egui_system: crate::engine::system::vulkan::egui::EguiSystem::new(
                Arc::clone(&vulkan_system.device()),
                Arc::clone(&vulkan_system.queue()),
                vulkan_system.image_format(),
                builder.window_width as f32,
                builder.window_height as f32,
            )
            .unwrap(),
            #[cfg(feature = "ui-egui")]
            egui_parts: parts::egui::EguiParts::default(),
            vulkan,
            vulkan_lines: VulkanBeautifulLineSystem::try_from(&vulkan_system)?,
            vulkan_system,
        })
    }

    #[cfg(feature = "ui-egui")]
    pub fn with_egui_context_callback(
        self,
        callback: impl FnMut(&egui::Context) + 'static,
    ) -> Self {
        self.with_egui_context_callback_dyn(Box::new(callback))
    }

    #[cfg(feature = "ui-egui")]
    pub fn with_egui_context_callback_dyn(
        mut self,
        callback: Box<dyn FnMut(&egui::Context) + 'static>,
    ) -> Self {
        self.egui_parts.content_callback = Some(callback.into());
        self
    }

    pub fn run(mut self) -> Self {
        let mut maximized = false;
        'running: loop {
            let mut allow_maximize_change = true;
            let time_start = Instant::now();
            for event in self.sdl.event_pump.poll_iter() {
                #[cfg(feature = "ui-egui")]
                self.egui_system.on_sdl2_event(&event);

                match event {
                    Event::Quit { .. }
                    | Event::KeyDown {
                        keycode: Some(Keycode::Escape),
                        ..
                    } => {
                        break 'running;
                    }
                    Event::Window {
                        win_event: WindowEvent::Resized(..) | WindowEvent::SizeChanged(..),
                        ..
                    } => {
                        self.vulkan_system.recreatee_swapchain();
                    }
                    Event::KeyUp {
                        keycode: Some(Keycode::F11),
                        repeat: false,
                        ..
                    } if allow_maximize_change => {
                        maximized = !maximized;
                        if maximized {
                            self.sdl.window.restore();
                        } else {
                            self.sdl.window.maximize();
                        }
                        self.sdl.window.set_bordered(maximized);
                        allow_maximize_change = false;
                    }
                    _ => {}
                }
            }

            let (width, height) = self.sdl.window.vulkan_drawable_size();

            #[cfg(feature = "ui-egui")]
            if let Some(callback) = &mut self.egui_parts.content_callback {
                self.egui_system.update_egui(width, height, |ctx| {
                    callback(ctx);
                });
            }

            self.vulkan_system.render(width, height, |builder| {
                #[cfg(feature = "ui-egui")]
                self.egui_system.prepare_render(builder).unwrap();
                |builder| {
                    #[cfg(feature = "ui-egui")]
                    self.egui_system.render(builder).unwrap();

                    let time = UNIX_EPOCH.elapsed().unwrap_or_default().subsec_millis() as f32
                        * std::f32::consts::PI.mul(2.0)
                        / 10.0;
                    self.vulkan_lines
                        .draw(
                            builder,
                            width as f32,
                            height as f32,
                            &[
                                Line {
                                    vertices: (0..200)
                                        .map(|x| {
                                            [
                                                100.0_f32 + (x as f32 * 2.5),
                                                150.0_f32
                                                    + (x as f32 / 2.0 + (time / 333.0))
                                                        .sin()
                                                        .mul(60.0),
                                            ]
                                        })
                                        .map(|pos| Vertex2d {
                                            pos,
                                            color: [0.25, 0.75, 0.45, 0.5],
                                        })
                                        .collect(),
                                    width: 1.0, // ((time / 666.0).sin().mul(3.0) + 4.0),
                                },
                                Line {
                                    vertices: vec![
                                        Vertex2d {
                                            pos: [400.0, 300.0],
                                            color: [0.0, 1.0, 1.0, 1.0],
                                        },
                                        Vertex2d {
                                            pos: [400.0, 450.0],
                                            color: [1.0, 1.0, 0.0, 1.0],
                                        },
                                        Vertex2d {
                                            pos: [550.0, 300.0],
                                            color: [1.0, 0.0, 1.0, 1.0],
                                        },
                                    ],
                                    width: 117.9,
                                },
                            ],
                        )
                        .unwrap();
                }
            });

            let duration_to_end = time_start.elapsed();
            dbg!(duration_to_end);
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
    #[error("Failed to load the vulkan library: {0}")]
    VulkanLibraryLoadingError(#[from] LoadingError),
    #[error("Error in vulkan system: {0}")]
    VulkanSystemError(#[from] system::vulkan::Error),
    #[error("Failed to load at least one subsystem: {0}")]
    VulkanSubsystemError(#[from] GraphicsPipelineCreationError),
}
