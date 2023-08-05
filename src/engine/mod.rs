use crate::engine::builder::EngineBuilder;
use crate::engine::parts::sdl::SdlParts;
use crate::engine::parts::vulkan::VulkanParts;
use crate::engine::system::canvas::buffered_layer::BufferedCanvasLayer;
use crate::engine::system::vulkan::beautiful_lines::{
    BeautifulLine, Vertex2d, VulkanBeautifulLineSystem,
};
use crate::engine::system::vulkan::lines::VulkanLineSystem;
use crate::engine::system::vulkan::textures::{Textured, Vertex2dUv, VulkanTextureSystem};
use crate::engine::system::vulkan::VulkanSystem;
use crate::engine::types::world2d::{Dim, Pos};
use image::GenericImageView;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::video::WindowBuildError;
use std::io::Cursor;
use std::ops::{Div, Mul};
use std::sync::Arc;
use std::time::{Instant, UNIX_EPOCH};
use vulkano::instance::{Instance, InstanceCreationError, InstanceExtensions};
use vulkano::pipeline::graphics::GraphicsPipelineCreationError;
use vulkano::sampler::SamplerCreationError;
use vulkano::swapchain::{Surface, SurfaceApi};
use vulkano::{Handle, LoadingError, VulkanLibrary, VulkanObject};

pub mod builder;
pub mod parts;
pub mod system;
pub mod types;

pub struct Engine {
    sdl: SdlParts,
    vulkan: VulkanParts,
    vulkan_system: VulkanSystem,
    vulkan_lines: VulkanLineSystem,
    vulkan_textures: VulkanTextureSystem,
    vulkan_beautiful_lines: VulkanBeautifulLineSystem,
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
            vulkan_lines: VulkanLineSystem::try_from(&vulkan_system)?,
            vulkan_textures: VulkanTextureSystem::try_from(&vulkan_system)?,
            vulkan_beautiful_lines: VulkanBeautifulLineSystem::try_from(&vulkan_system)?,
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
        const IMAGE_DATA: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/res/rust-logo-256x256.png",
        ));

        let mut texture = None;
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

                if texture.is_none() {
                    let image = image::io::Reader::new(Cursor::new(IMAGE_DATA))
                        .with_guessed_format()
                        .unwrap()
                        .decode()
                        .unwrap();

                    texture = Some(
                        self.vulkan_textures
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
                    self.egui_system.render(builder).unwrap();

                    let time = UNIX_EPOCH.elapsed().unwrap_or_default().subsec_millis() as f32
                        * std::f32::consts::PI.mul(2.0)
                        / 10.0;
                    self.vulkan_beautiful_lines
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

                    self.vulkan_lines
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

                    layer.submit_to_render_pass(
                        builder,
                        &mut self.vulkan_lines,
                        &mut self.vulkan_textures,
                    );

                    if let Some(texture) = texture {
                        self.vulkan_textures
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
    #[error("Failed to create a Vulkan instance: {0}")]
    VulkanInstanceCreationError(#[from] InstanceCreationError),
    #[error("Failed to load the vulkan library: {0}")]
    VulkanLibraryLoadingError(#[from] LoadingError),
    #[error("Error in vulkan system: {0}")]
    VulkanSystemError(#[from] system::vulkan::Error),
    #[error("Failed to load at least one subsystem: {0}")]
    VulkanSubsystemError(#[from] GraphicsPipelineCreationError),
    #[error("Failed to load at least one subsystem because of an sampler creation error: {0}")]
    VulkanSubsystemSamplerError(#[from] SamplerCreationError),
}

impl From<system::vulkan::textures::CreationError> for Error {
    #[inline]
    fn from(error: system::vulkan::textures::CreationError) -> Self {
        match dbg!(error) {
            system::vulkan::textures::CreationError::GraphicsPipelineCreationError(e) => {
                Self::VulkanSubsystemError(e)
            }
            system::vulkan::textures::CreationError::SamplerCreationError(e) => {
                Self::VulkanSubsystemSamplerError(e)
            }
        }
    }
}
