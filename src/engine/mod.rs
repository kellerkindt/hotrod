use crate::engine::builder::EngineBuilder;
use crate::engine::parts::sdl::SdlParts;
use crate::engine::system::vulkan::beautiful_lines::BeautifulLinePipeline;
use crate::engine::system::vulkan::pipelines::VulkanPipelines;
use crate::engine::system::vulkan::DrawError;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::video::WindowBuildError;
use std::sync::Arc;
use std::time::{Duration, Instant};
use system::vulkan::system::VulkanSystem;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, SecondaryAutoCommandBuffer,
};
use vulkano::instance::{Instance, InstanceExtensions};
use vulkano::swapchain::{Surface, SurfaceApi};
use vulkano::{Handle, LoadingError, Validated, VulkanError, VulkanLibrary, VulkanObject};

pub mod builder;
pub mod parts;
pub mod system;
pub mod types;

pub struct Engine {
    vulkan_system: VulkanSystem,
    vulkan_pipelines: Arc<VulkanPipelines>,
    #[cfg(feature = "ui-egui")]
    egui_system: system::egui::EguiSystem,
    #[cfg(feature = "ui-egui")]
    // drop after the vulkan system! (last is fine, too)
    sdl: SdlParts,
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

        // ============================ WARNING ============================
        //   Because of this unsafe linkage the `window` object *must not*
        //   be dropped before the vulkan stuff / swapchain or the program
        //   will experience a SIGSEGV!
        // =================================================================
        let surface_handle = window
            .vulkan_create_surface(instance.handle().as_raw() as _)
            .map_err(Error::SdlCreateVulkanSurfaceError)?;

        // SAFETY: that's the way it is
        // SAFETY: Be sure not to drop the `window` before the `Surface` or vulkan `Swapchain`! (SIGSEGV otherwise)
        let surface = Arc::new(unsafe {
            Surface::from_handle(
                Arc::clone(&instance),
                <_ as Handle>::from_raw(surface_handle),
                SurfaceApi::Xlib,
                None,
            )
        });

        let vulkan_system = VulkanSystem::new(
            surface,
            builder.window_width,
            builder.window_height,
            BeautifulLinePipeline::REQUIRED_FEATURES,
        )?;

        Ok(Self {
            vulkan_pipelines: Arc::new(VulkanPipelines::try_from(&vulkan_system)?),
            #[cfg(feature = "ui-egui")]
            egui_system: system::egui::EguiSystem::default(),
            vulkan_system,
            sdl: SdlParts {
                video_subsystem,
                event_pump,
                // drop after the vulkan system!
                window,
                window_maximized: false,
                context,
            },
        })
    }

    pub fn update<T>(&mut self, f: impl FnOnce(BeforeRenderContext) -> T) -> RenderResponse<T> {
        let start = Instant::now();
        let events = self.poll_events();
        let (width, height) = self.sdl.window.vulkan_drawable_size();

        let data = f(BeforeRenderContext {
            engine: self,
            events,
            width,
            height,
            start,
        });

        RenderResponse {
            data,
            start,
            duration: start.elapsed(),
        }
    }

    fn poll_events(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        let mut allow_maximize_change = true;
        for event in self.sdl.event_pump.poll_iter() {
            #[cfg(feature = "ui-egui")]
            self.egui_system.on_sdl2_event(&event);

            match &event {
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
                    self.sdl.window_maximized = !self.sdl.window_maximized;
                    if self.sdl.window_maximized {
                        self.sdl.window.restore();
                    } else {
                        self.sdl.window.maximize();
                    }
                    self.sdl.window.set_bordered(self.sdl.window_maximized);
                    allow_maximize_change = false;
                }
                _ => {}
            }
            events.push(event);
        }
        events
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
    // #[error("Failed to create a Vulkan instance: {0}")]
    // VulkanInstanceCreateInfo(#[from] InstanceCreateInfo),
    #[error("Failed to load the vulkan library: {0}")]
    VulkanLibraryLoadingError(#[from] LoadingError),
    #[error("Validated Vulkan Error: {0}")]
    ValidatedVulkanError(#[from] Validated<VulkanError>),
    #[error("Vulkan System Error: {0}")]
    VulkanSystemError(#[from] system::vulkan::Error),
    #[error("Failed to create a Vulkan System Pipeline: {0}")]
    PipelineSystemCreateError(#[from] system::vulkan::PipelineCreateError),
}

pub struct BeforeRenderContext<'a> {
    engine: &'a mut Engine,
    pub events: Vec<Event>,
    pub width: u32,
    pub height: u32,
    pub start: Instant,
}

impl<'a> BeforeRenderContext<'a> {
    #[cfg(feature = "ui-egui")]
    pub fn update_egui(&mut self, f: impl FnOnce(&egui::Context)) {
        self.engine.egui_system.update(self.width, self.height, f)
    }

    pub fn render<F1, F2>(self, f1: F1) -> Result<(), DrawError>
    where
        F1: FnOnce(PrepareRenderContext) -> F2,
        F2: FnOnce(RenderContext) -> Vec<Arc<SecondaryAutoCommandBuffer>>,
    {
        self.engine
            .vulkan_system
            .render(self.width, self.height, |commands| {
                #[cfg(feature = "ui-egui")]
                if let Err(e) = self
                    .engine
                    .vulkan_pipelines
                    .egui
                    .prepare(commands, &self.engine.egui_system)
                {
                    eprintln!("Failed to prepare rendering for egui: {e}");
                    eprintln!("{e:?}");
                }

                let f2 = f1(PrepareRenderContext {
                    commands,
                    pipelines: &self.engine.vulkan_pipelines,
                    width: self.width,
                    height: self.height,
                });

                |render_context| {
                    let mut commands = Vec::default();

                    std::thread::scope(|scope| {
                        #[cfg(feature = "ui-egui")]
                        let egui = {
                            scope.spawn(|| {
                                let mut commands =
                                    render_context.create_command_buffer_builder().unwrap();
                                if let Err(e) = self
                                    .engine
                                    .vulkan_pipelines
                                    .egui
                                    .draw(&mut commands, &self.engine.egui_system)
                                {
                                    eprintln!("Failed to render egui: {e}");
                                    eprintln!("{e:?}");
                                }

                                commands.build().unwrap()
                            })
                        };

                        commands.extend(f2(RenderContext {
                            inner: render_context,
                            pipelines: &self.engine.vulkan_pipelines,
                            width: self.width,
                            height: self.height,
                        }));

                        #[cfg(feature = "ui-egui")]
                        match egui.join() {
                            Ok(egui) => {
                                commands.push(egui);
                            }
                            Err(e) => {
                                eprintln!("Failed to collect egui commands: {e:?}")
                            }
                        }
                    });

                    commands
                }
            })
    }
}

pub struct PrepareRenderContext<'a> {
    pub commands: &'a mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    pub pipelines: &'a Arc<VulkanPipelines>,
    pub width: u32,
    pub height: u32,
}

pub struct RenderContext<'a, 'b> {
    pub inner: &'a system::vulkan::system::RenderContext<'b>,
    pub pipelines: &'a Arc<VulkanPipelines>,
    pub width: u32,
    pub height: u32,
}

pub struct RenderResponse<T> {
    pub data: T,
    pub start: Instant,
    pub duration: Duration,
}
