use egui::Context;
use hotrod::engine::builder::EngineBuilder;
use hotrod::engine::system::canvas::buffered_layer::BufferedCanvasLayer;
use hotrod::engine::system::vulkan::beautiful_lines::{BeautifulLine, Vertex2d};
use hotrod::engine::system::vulkan::textured::{
    Textured, TexturedIndexed, TexturedPipeline, Vertex2dUv,
};
use hotrod::engine::system::vulkan::textures::TextureId;
use hotrod::engine::system::vulkan::triangles::{Triangles, TrianglesIndexed};
use hotrod::engine::types::world2d::{Dim, Pos};
use hotrod::engine::RenderContext;
use hotrod::sdl2;
use hotrod::sdl2::event::Event;
use hotrod::sdl2::keyboard::Keycode;
use hotrod::sdl2::pixels::PixelFormatEnum;
use hotrod::sdl2::ttf::{FontStyle, Hinting};
use image::GenericImageView;
use std::io::Cursor;
use std::ops::{Div, Mul};
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};
use vulkano::image::SampleCount;

const IMAGE_DATA: &[u8] = include_bytes!(concat!("rust-logo-256x256.png"));

fn main() {
    // #############################################################################################
    // #                                                                                           #
    // #    Initialization                                                                         #
    // #                                                                                           #
    // #############################################################################################
    hotrod::hint::video::prefer_wayland();
    hotrod::logging::init_logger_with_customization(|builder| {
        builder
            .with_env_filter(format!(
                "{}=trace",
                env!("CARGO_PKG_NAME").replace('-', "_")
            ))
            .with_max_level(tracing::metadata::LevelFilter::INFO)
    })
    .expect("Unable to init logger");

    let mut engine = {
        let engine = EngineBuilder::default()
            .with_window_title("Silly Example - Forged by the HotRod Engine")
            .with_target_frame_rate(144)
            .with_fullscreen(!cfg!(debug_assertions))
            .with_msaa(SampleCount::Sample1);

        // ttf example: use the FontRenderer
        #[cfg(feature = "ttf-font-renderer")]
        let engine = engine.with_ttf_font_renderer(include_bytes!(
            "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf"
        ));

        engine.build().expect("Failed to build the engine")
    };

    let mut duration_engine = Duration::from_secs(1);
    let mut duration_loop = Duration::from_secs(1);
    let mut texture = None;
    let mut ttf = None;

    // #############################################################################################
    // #                                                                                           #
    // #    Game Loop                                                                              #
    // #                                                                                           #
    // #############################################################################################
    loop {
        let loop_start = Instant::now();
        let response = engine.update(|mut ctx| {
            // handle all new inputs
            let quit = ctx.events.drain(..).any(|e| wants_to_quit(e));

            // update the UI
            ctx.update_egui(|ctx| {
                show_stats_windows(ctx, duration_engine, duration_loop);
            });

            // render custom stuff
            ctx.render(|context| {
                let mut buffers = Vec::default();

                if texture.is_none() {
                    texture = Some(load_image_texture(&context));
                }

                // ttf example: handle all the font rendering yourself
                #[cfg(feature = "ttf-font-renderer")]
                if ttf.is_none() {
                    ttf = Some(load_ttf(&context));
                }

                let mut commands = context.inner.create_render_buffer_builder().unwrap();
                let mut canvas = BufferedCanvasLayer::new(
                    context.inner.create_render_buffer_builder().unwrap(),
                    Arc::clone(context.pipelines),
                );

                canvas.draw_rect(Pos::new(20.0, 20.0), Dim::new(50.0, 50.0));

                let time = UNIX_EPOCH.elapsed().unwrap_or_default().subsec_millis() as f32
                    * std::f32::consts::PI.mul(2.0)
                    / 10.0;

                context
                    .pipelines
                    .beautiful_line
                    .draw(
                        &mut commands,
                        &[
                            BeautifulLine {
                                vertices: (0..200)
                                    .map(|x| {
                                        [
                                            100.0_f32 + (x as f32 * 2.5),
                                            150.0_f32
                                                + (x as f32 / 2.0 + (time / 333.0)).sin().mul(60.0),
                                        ]
                                    })
                                    .map(|pos| Vertex2d {
                                        pos,
                                        color: [0.25, 0.75, 0.45, 0.5],
                                    })
                                    .collect(),
                                width: 1.0, // ((time / 666.0).sin().mul(3.0) + 4.0),
                            },
                            // can you spot it in the final image?
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

                context
                    .pipelines
                    .line
                    .draw(
                        &mut commands,
                        &[hotrod::engine::system::vulkan::lines::Line {
                            vertices: (0..200)
                                .map(|x| {
                                    [
                                        100.0_f32 + (x as f32 * 2.5),
                                        150.0_f32
                                            + (x as f32 / 2.0 + (time / 333.0)).sin().mul(60.0),
                                    ]
                                })
                                .map(|pos| hotrod::engine::system::vulkan::lines::Vertex2d { pos })
                                .collect(),
                            // color: [0.25, 0.75, 0.45, 0.5],
                            color: [(time / 1000.0).fract(), 0.0, 0.0, 1.0],
                        }],
                    )
                    .unwrap();

                // use the BufferedCanvasLayer as for easier drawing if 'primitives'
                let mut layer = BufferedCanvasLayer::default();
                layer.draw_line([10.0, 10.0], [100.0, 100.0]);
                layer.set_draw_color([1.0, 0.0, 0.0, 1.0]);
                layer.draw_path(&[[10.0, 10.0], [100.0, 10.0], [100.0, 100.0]]);
                layer.set_draw_color([0.0, 1.0, 0.0, 1.0]);
                layer.draw_path(&[[100.0, 100.0], [10.0, 100.0], [10.0, 10.0]]);
                layer.set_draw_color([1.0, 0.0, 1.0, 1.0]);
                layer.draw_rect(Pos::new(200.0, 200.0), Dim::new(25.0, 25.0));
                layer.set_draw_color([1.0, 1.0, 0.0, 0.5]);
                layer.fill_rect(Pos::new(250.0, 550.0), Dim::new(25.0, 25.0));

                if let Some(texture) = &texture {
                    layer.draw_textured_rect(
                        Pos::new(100.0, 500.0),
                        {
                            let d = 50.0 * (2.0 + time.div(100.0).cos());
                            Dim::new(d, d)
                        },
                        texture.clone(),
                    );
                }

                // ttf example: handle all the font rendering yourself
                #[cfg(feature = "ttf-font-renderer")]
                if let Some((ttf, ratio)) = &ttf {
                    let width = 500.0;
                    layer.draw_textured_rect(
                        Pos::new(500.0, 500.0),
                        Dim::new(width, width * ratio),
                        ttf.clone(),
                    );
                }

                // ttf example: use the FontRenderer
                #[cfg(feature = "ttf-font-renderer")]
                {
                    let prepared = context.font_renderer.prepare_render(
                        &context.pipelines.texture,
                        context.inner.image_system(),
                        "The FontRenderer Text",
                        24,
                        [255, 255, 0, 255],
                        50.0,
                        400.0,
                    );
                    context
                        .pipelines
                        .texture
                        .draw(&mut commands, &[prepared])
                        .unwrap();
                }

                buffers.push(layer.flush(context.inner, context.pipelines));

                // draw some funny pictures without the BufferedCanvasLayer
                if let Some(texture) = &texture {
                    context
                        .pipelines
                        .texture
                        .draw(
                            &mut commands,
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
                                texture: texture.clone(),
                            }],
                        )
                        .unwrap();

                    context
                        .pipelines
                        .texture
                        .draw_indexed(
                            &mut commands,
                            &[
                                TexturedIndexed {
                                    vertices: vec![
                                        Vertex2dUv {
                                            pos: [550.0, 200.0],
                                            uv: [0.0, 0.0],
                                        },
                                        Vertex2dUv {
                                            pos: [650.0, 200.0],
                                            uv: [1.0, 0.0],
                                        },
                                        Vertex2dUv {
                                            pos: [650.0, 300.0],
                                            uv: [1.0, 1.0],
                                        },
                                        Vertex2dUv {
                                            pos: [550.0, 300.0],
                                            uv: [0.0, 1.0],
                                        },
                                    ],
                                    indices: vec![[0, 1, 2], [2, 3, 0]],
                                    texture: texture.clone(),
                                },
                                TexturedIndexed {
                                    vertices: vec![
                                        Vertex2dUv {
                                            pos: [550.0, 500.0],
                                            uv: [0.0, 0.0],
                                        },
                                        Vertex2dUv {
                                            pos: [650.0, 500.0],
                                            uv: [1.0, 0.0],
                                        },
                                        Vertex2dUv {
                                            pos: [650.0, 600.0],
                                            uv: [1.0, 1.0],
                                        },
                                        Vertex2dUv {
                                            pos: [550.0, 600.0],
                                            uv: [0.0, 1.0],
                                        },
                                    ],
                                    indices: vec![[0, 1, 2], [2, 3, 0]],
                                    texture: texture.clone(),
                                },
                            ],
                        )
                        .unwrap();

                    context
                        .pipelines
                        .triangles
                        .draw(
                            &mut commands,
                            &[Triangles {
                                vertices: vec![
                                    hotrod::engine::system::vulkan::triangles::Vertex2d {
                                        pos: [800.0, 500.0],
                                    },
                                    hotrod::engine::system::vulkan::triangles::Vertex2d {
                                        pos: [900.0, 500.0],
                                    },
                                    hotrod::engine::system::vulkan::triangles::Vertex2d {
                                        pos: [900.0, 600.0],
                                    },
                                ],
                                color: [1.0, 1.0, 0.0, 1.0],
                            }],
                        )
                        .unwrap();

                    context
                        .pipelines
                        .triangles
                        .draw_indexed(
                            &mut commands,
                            &[TrianglesIndexed {
                                vertices: vec![
                                    hotrod::engine::system::vulkan::triangles::Vertex2d {
                                        pos: [850.0, 500.0],
                                    },
                                    hotrod::engine::system::vulkan::triangles::Vertex2d {
                                        pos: [950.0, 500.0],
                                    },
                                    hotrod::engine::system::vulkan::triangles::Vertex2d {
                                        pos: [950.0, 600.0],
                                    },
                                ],
                                indices: vec![[0, 1, 2]],
                                color: [0.0, 0.0, 1.0, 0.5],
                            }],
                        )
                        .unwrap();
                }

                buffers.push(canvas.flush(context.inner, &context.pipelines));
                buffers.push(commands.build().unwrap());
                buffers
            })
            .map(|_| !quit)
        });

        match response.data {
            Ok(true) => {
                duration_engine = response.duration;
            }
            Ok(false) => break,
            Err(e) => {
                eprintln!("RENDER ERROR: {e}");
                eprintln!("RENDER ERROR: {e:?}");
                break;
            }
        }

        engine.delay();
        duration_loop = loop_start.elapsed();
    }
}

fn load_ttf(context: &RenderContext) -> (TextureId<TexturedPipeline>, f32) {
    let ttf_ctxt = sdl2::ttf::init().expect("Failed to init TTF context");

    let mut font = ttf_ctxt
        .load_font("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", 100)
        .unwrap();

    font.set_style(FontStyle::BOLD);
    font.set_hinting(Hinting::Normal);

    let surface = font
        .render("Bernd das Brot")
        .blended(sdl2::pixels::Color::GREEN)
        .unwrap();

    // dbg!(surface.pixel_format());
    dbg!(surface.pixel_format_enum());
    dbg!(surface.alpha_mod());

    let surface = surface.convert_format(PixelFormatEnum::RGBA32).unwrap();

    dbg!(surface.pixel_format_enum());
    dbg!(surface.alpha_mod());

    let data = surface.without_lock().unwrap().to_vec();

    let image = context
        .inner
        .image_system()
        .create_image_and_enqueue_upload(data, surface.width(), surface.height())
        .expect("Failed to upload image");

    let texture = context.pipelines.texture.prepare_texture(image).unwrap();

    (texture, surface.height() as f32 / surface.width() as f32)
}

fn load_image_texture(context: &RenderContext) -> TextureId<TexturedPipeline> {
    let image = image::ImageReader::new(Cursor::new(IMAGE_DATA))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap();

    let image = context
        .inner
        .image_system()
        .create_image_and_enqueue_upload(
            image
                .pixels()
                .flat_map(|(_x, _y, rgba)| rgba.0)
                .collect::<Vec<u8>>(),
            image.width(),
            image.height(),
        )
        .expect("Failed to upload image");

    context.pipelines.texture.prepare_texture(image).unwrap()
}

fn wants_to_quit(e: Event) -> bool {
    match e {
        Event::Quit { .. } => true,
        Event::KeyDown { keycode, .. } => {
            matches!(keycode, Some(Keycode::Escape))
        }
        _ => false,
    }
}

fn show_stats_windows(ctx: &Context, duration_engine: Duration, duration_loop: Duration) {
    use hotrod::ui::egui::Window;
    Window::new("HotRod - Engine Time")
        .resizable(true)
        .show(ctx, |ui| {
            ui.label(format!("{duration_engine:?}"));
            ui.label(format!("~{:.2}fps", (1.0 / duration_engine.as_secs_f32())));
            if ui.button("CLICK ME").clicked() {
                eprintln!("I WAS CLICKED!");
            }
        });
    Window::new("HotRod - Present Time")
        .resizable(true)
        .show(ctx, |ui| {
            ui.label(format!("{duration_loop:?}"));
            ui.label(format!("~{:.2}fps", (1.0 / duration_loop.as_secs_f32())));
            if ui.button("CLICK ME").clicked() {
                eprintln!("I WAS CLICKED!");
            }
        });
}
