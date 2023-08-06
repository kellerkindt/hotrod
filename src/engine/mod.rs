use crate::engine::builder::EngineBuilder;
use crate::engine::parts::sdl::SdlParts;
use crate::engine::system::canvas::buffered_layer::BufferedCanvasLayer;
use crate::engine::system::vulkan::beautiful_lines::{
    BeautifulLine, BeautifulLinePipeline, Vertex2d,
};
use crate::engine::system::vulkan::pipelines::VulkanPipelines;
use crate::engine::system::vulkan::textures::{Textured, Vertex2dUv};
use crate::engine::system::vulkan::DrawError;
use crate::engine::types::world2d::{Dim, Pos};
use image::GenericImageView;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::video::WindowBuildError;
use std::io::Cursor;
use std::ops::{Div, Mul};
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};
use system::vulkan::system::VulkanSystem;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::instance::{Instance, InstanceExtensions};
use vulkano::swapchain::{Surface, SurfaceApi};
use vulkano::{Handle, LoadingError, Validated, VulkanError, VulkanLibrary, VulkanObject};

pub mod builder;
pub mod parts;
pub mod system;
pub mod types;

pub struct Engine {
    vulkan_system: VulkanSystem,
    vulkan_pipelines: VulkanPipelines,
    #[cfg(feature = "ui-egui")]
    egui_system: system::egui::EguiSystem,
    #[cfg(feature = "ui-egui")]
    egui: parts::egui::EguiParts,
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
            vulkan_pipelines: VulkanPipelines::try_from(&vulkan_system)?,
            #[cfg(feature = "ui-egui")]
            egui_system: system::egui::EguiSystem::default(),
            #[cfg(feature = "ui-egui")]
            egui: parts::egui::EguiParts::default(),
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
        self.egui.content_callback = Some(callback.into());
        self
    }

    #[cfg(feature = "ui-egui")]
    pub fn update_egui(&mut self, f: impl FnOnce(&egui::Context)) {
        let (width, height) = self.sdl.window.vulkan_drawable_size();
        self.egui_system.update(width, height, f)
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

    pub fn run(mut self) -> Self {
        const IMAGE_DATA: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/res/rust-logo-256x256.png",
        ));

        let mut texture = None;

        'running: loop {
            let time_start = Instant::now();

            if self.poll_events().iter().any(|e| match e {
                Event::Quit { .. } => true,
                Event::KeyDown { keycode, .. } => {
                    matches!(keycode, Some(Keycode::Escape))
                }
                _ => false,
            }) {
                break 'running;
            }

            let (width, height) = self.sdl.window.vulkan_drawable_size();

            #[cfg(feature = "ui-egui")]
            if let Some(callback) = &mut self.egui.content_callback {
                self.egui_system.update(width, height, |ctx| {
                    callback(ctx);
                });
            }

            let render_result = self.vulkan_system.render(width, height, |builder| {
                #[cfg(feature = "ui-egui")]
                self.vulkan_pipelines
                    .egui
                    .prepare(builder, &self.egui_system)
                    .unwrap();

                if texture.is_none() {
                    let image = image::io::Reader::new(Cursor::new(IMAGE_DATA))
                        .with_guessed_format()
                        .unwrap()
                        .decode()
                        .unwrap();

                    texture = Some(
                        self.vulkan_pipelines
                            .texture
                            .create_texture(
                                builder,
                                image
                                    .pixels()
                                    .flat_map(|(_x, _y, rgba)| rgba.0)
                                    .collect::<Vec<u8>>(),
                                image.width(),
                                image.height(),
                            )
                            .unwrap(),
                    );
                }

                |builder| {
                    #[cfg(feature = "ui-egui")]
                    self.vulkan_pipelines
                        .egui
                        .draw(builder, &self.egui_system)
                        .unwrap();

                    let time = UNIX_EPOCH.elapsed().unwrap_or_default().subsec_millis() as f32
                        * std::f32::consts::PI.mul(2.0)
                        / 10.0;

                    self.vulkan_pipelines
                        .beautiful_line
                        .draw(
                            builder,
                            width as f32,
                            height as f32,
                            &[
                                BeautifulLine {
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
                                BeautifulLine {
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

                    self.vulkan_pipelines
                        .line
                        .draw(
                            builder,
                            width as f32,
                            height as f32,
                            &[system::vulkan::lines::Line {
                                vertices: (0..200)
                                    .map(|x| {
                                        [
                                            100.0_f32 + (x as f32 * 2.5),
                                            150.0_f32
                                                + (x as f32 / 2.0 + (time / 333.0)).sin().mul(60.0),
                                        ]
                                    })
                                    .map(|pos| system::vulkan::lines::Vertex2d { pos })
                                    .collect(),
                                // color: [0.25, 0.75, 0.45, 0.5],
                                color: [(time / 1000.0).fract(), 0.0, 0.0, 1.0],
                            }],
                        )
                        .unwrap();

                    let mut layer = BufferedCanvasLayer::from([width, height]);
                    layer.draw_line([10.0, 10.0], [100.0, 100.0]);
                    layer.set_draw_color([1.0, 0.0, 0.0, 1.0]);
                    layer.draw_path(&[[10.0, 10.0], [100.0, 10.0], [100.0, 100.0]]);
                    layer.set_draw_color([0.0, 1.0, 0.0, 1.0]);
                    layer.draw_path(&[[100.0, 100.0], [10.0, 100.0], [10.0, 10.0]]);
                    layer.set_draw_color([1.0, 0.0, 1.0, 1.0]);
                    layer.draw_rect(Pos::new(200.0, 200.0), Dim::new(25.0, 25.0));

                    if let Some(texture) = texture {
                        layer.draw_textured_rect(
                            Pos::new(100.0, 500.0),
                            {
                                let d = 50.0 * (2.0 + time.div(100.0).cos());
                                Dim::new(d, d)
                            },
                            texture,
                        );
                    }

                    layer.submit_to_render_pass(builder, &mut self.vulkan_pipelines);

                    if let Some(texture) = texture {
                        self.vulkan_pipelines
                            .texture
                            .draw(
                                builder,
                                width as f32,
                                height as f32,
                                &[Textured {
                                    vertices: vec![
                                        Vertex2dUv {
                                            pos: [500.0, 100.0],
                                            uv: [0.0, 0.0],
                                        },
                                        Vertex2dUv {
                                            pos: [600.0, 100.0],
                                            uv: [1.0, 0.0],
                                        },
                                        Vertex2dUv {
                                            pos: [600.0, 200.0],
                                            uv: [1.0, 1.0],
                                        },
                                        Vertex2dUv {
                                            pos: [600.0, 200.0],
                                            uv: [1.0, 1.0],
                                        },
                                        Vertex2dUv {
                                            pos: [500.0, 200.0],
                                            uv: [0.0, 1.0],
                                        },
                                        Vertex2dUv {
                                            pos: [500.0, 100.0],
                                            uv: [0.0, 0.0],
                                        },
                                    ],
                                    texture,
                                }],
                            )
                            .unwrap();
                    }
                }
            });

            if let Err(e) = render_result {
                eprintln!("RENDER ERROR: {e:?}");
            }

            let duration_to_end = time_start.elapsed();
            let expected_fps = 1.0 / duration_to_end.as_secs_f32();
            eprintln!("duration_to_end={duration_to_end:?}, ~fps={expected_fps:.2}");
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
        F2: FnOnce(RenderContext),
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

                let mut canvas = BufferedCanvasLayer::from([self.width, self.height]);
                let f2 = f1(PrepareRenderContext { commands });

                |commands| {
                    f2(RenderContext {
                        commands,
                        canvas: &mut canvas,
                        pipelines: &mut self.engine.vulkan_pipelines,
                    });

                    canvas.submit_to_render_pass(commands, &mut self.engine.vulkan_pipelines);

                    #[cfg(feature = "ui-egui")]
                    if let Err(e) = self
                        .engine
                        .vulkan_pipelines
                        .egui
                        .draw(commands, &self.engine.egui_system)
                    {
                        eprintln!("Failed to render egui: {e}");
                        eprintln!("{e:?}");
                    }
                }
            })
    }
}

pub struct PrepareRenderContext<'a> {
    pub commands: &'a mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
}

pub struct RenderContext<'a> {
    pub commands: &'a mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    pub canvas: &'a mut BufferedCanvasLayer,
    pub pipelines: &'a mut VulkanPipelines,
}

pub struct RenderResponse<T> {
    pub data: T,
    pub start: Instant,
    pub duration: Duration,
}
