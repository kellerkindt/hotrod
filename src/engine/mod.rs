use crate::engine::builder::EngineBuilder;
use crate::engine::parts::sdl::SdlParts;
use crate::engine::system::fps::FpsManager;
use crate::engine::system::ttf::FontRenderer;
use crate::engine::system::vulkan::beautiful_lines::BeautifulLinePipeline;
use crate::engine::system::vulkan::pipelines::VulkanPipelines;
use crate::engine::system::vulkan::DrawError;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::video::{FullscreenType, WindowBuildError};
use std::sync::Arc;
use std::time::{Duration, Instant};
use system::vulkan::system::VulkanSystem;
use vulkano::command_buffer::SecondaryAutoCommandBuffer;
use vulkano::image::SampleCount;
use vulkano::instance::{Instance, InstanceExtensions};
use vulkano::swapchain::Surface;
use vulkano::{LoadingError, Validated, VulkanError, VulkanLibrary};

pub mod builder;
pub mod parts;
pub mod system;
pub mod types;

pub struct Engine {
    vulkan_system: VulkanSystem,
    vulkan_pipelines: Arc<VulkanPipelines>,
    #[cfg(feature = "ui-egui")]
    egui_system: system::egui::EguiSystem,
    #[cfg(feature = "ttf-font-renderer")]
    font_renderer: FontRenderer,
    #[cfg(feature = "ui-egui")]
    // drop after the vulkan system! (last is fine, too)
    sdl: SdlParts,
    framerate_manager: FpsManager,
}

impl Engine {
    pub fn new(builder: EngineBuilder) -> Result<Self, Error> {
        info!("SDL2 Version {}", sdl2::version::version());
        info!(
            "SDL2 Video Drivers: {:?}",
            sdl2::video::drivers().collect::<Vec<_>>()
        );

        let context = sdl2::init().map_err(Error::SdlError)?;
        let video_subsystem = context.video().map_err(Error::SdlError)?;
        let event_pump = context.event_pump().map_err(Error::SdlError)?;

        info!(
            "SDL2 Chosen Video Driver: {}",
            video_subsystem.current_video_driver()
        );

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

        // SAFETY: Be sure not to drop the `window` before the `Surface` or vulkan `Swapchain`! (SIGSEGV otherwise)
        let surface = unsafe { Surface::from_window_ref(Arc::clone(&instance), &window) }
            .expect("Failed to create surface from window ref");

        info!("Window Surface API: {:?}", surface.api());

        let mut vulkan_system = VulkanSystem::new(
            surface,
            builder.window_width,
            builder.window_height,
            BeautifulLinePipeline::REQUIRED_FEATURES,
            builder.msaa.unwrap_or(SampleCount::Sample1),
        )?;

        if let Some(clear_color) = builder.background_clear_color {
            vulkan_system.set_clear_value(clear_color);
        }

        let mut this = Self {
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
                #[cfg(feature = "ttf-sdl2")]
                ttf: sdl2::ttf::init()
                    .map_err(|e| Error::SdlError(format!("Failed to init TTF module: {e}")))?,
                context,
                window_icon: None,
            }
            .maybe_with_window_icon(builder.window_icon),
            framerate_manager: FpsManager::new(builder.target_frame_rate),
            #[cfg(feature = "ttf-font-renderer")]
            font_renderer: FontRenderer::new(
                builder.font_renderer_ttf.expect("Missing TrueType Font"),
            ),
        };

        this.set_fullscreen(builder.fullscreen);

        Ok(this)
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

        #[cfg(feature = "ttf-font-renderer")]
        self.font_renderer.on_frame_completed();

        RenderResponse {
            data,
            start,
            duration: start.elapsed(),
        }
    }

    fn poll_events(&mut self) -> Vec<Event> {
        let mut allow_maximize_change = true;
        let events = self.sdl.event_pump.poll_iter().collect();

        for event in &events {
            #[cfg(feature = "ui-egui")]
            self.egui_system.on_sdl2_event(event);

            match event {
                Event::Window {
                    win_event: WindowEvent::Resized(..) | WindowEvent::SizeChanged(..),
                    ..
                } => {
                    self.vulkan_system.recreate_swapchain();
                }
                Event::KeyUp {
                    keycode: Some(Keycode::F11),
                    repeat: false,
                    ..
                } if allow_maximize_change => {
                    self.set_fullscreen(!self.sdl.window_maximized);
                    allow_maximize_change = false;
                }
                _ => {}
            }
        }

        events
    }

    #[inline]
    pub fn set_fps(&mut self, fps: u16) {
        self.framerate_manager.set_target_frame_rate(fps);
        #[cfg(feature = "egui")]
        self.egui_system.set_target_frame_rate(fps);
    }

    pub fn set_fullscreen(&mut self, fullscreen: bool) {
        self.sdl.window_maximized = fullscreen;
        if self.sdl.window_maximized {
            self.sdl.window.maximize();
            if let Err(e) = self.sdl.window.set_fullscreen(FullscreenType::True) {
                error!("Enabling fullscreen failed: {e}");
            }
        } else {
            if let Err(e) = self.sdl.window.set_fullscreen(FullscreenType::Off) {
                error!("Disabling fullscreen failed: {e}");
            }
            self.sdl.window.restore();
        }
        self.sdl.window.set_bordered(!self.sdl.window_maximized);
        #[cfg(feature = "egui")]
        self.egui_system.set_fullscreen(fullscreen);
    }

    #[inline]
    pub fn delay(&mut self) -> Duration {
        self.framerate_manager.delay()
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
        self.engine
            .egui_system
            .update(self.width, self.height, &mut self.engine.sdl, f)
    }

    pub fn render<F1>(self, f1: F1) -> Result<(), DrawError>
    where
        F1: FnOnce(RenderContext) -> Vec<Arc<SecondaryAutoCommandBuffer>>,
    {
        self.engine
            .vulkan_system
            .render(self.width, self.height, |render_context| {
                let mut commands = Vec::default();

                #[cfg(feature = "ui-egui")]
                if let Err(e) = self
                    .engine
                    .vulkan_pipelines
                    .egui
                    .prepare(&self.engine.egui_system)
                {
                    error!("Failed to prepare rendering for egui: {e}");
                }

                commands.extend(f1(RenderContext {
                    inner: render_context,
                    pipelines: &self.engine.vulkan_pipelines,
                    width: self.width,
                    height: self.height,
                    #[cfg(feature = "ttf-font-renderer")]
                    font_renderer: &mut self.engine.font_renderer,
                }));

                #[cfg(feature = "ui-egui")]
                {
                    let mut builder = render_context.create_render_buffer_builder().unwrap();
                    if let Err(e) = self
                        .engine
                        .vulkan_pipelines
                        .egui
                        .draw(&mut builder, &self.engine.egui_system)
                    {
                        error!("Failed to render egui: {e}");
                    }

                    commands.push(builder.build().unwrap());
                }

                commands
            })
    }
}

pub struct RenderContext<'a, 'b> {
    pub inner: &'a system::vulkan::system::RenderContext<'b>,
    pub pipelines: &'a Arc<VulkanPipelines>,
    pub width: u32,
    pub height: u32,
    #[cfg(feature = "ttf-font-renderer")]
    pub font_renderer: &'a mut FontRenderer,
}

pub struct RenderResponse<T> {
    pub data: T,
    pub start: Instant,
    pub duration: Duration,
}
