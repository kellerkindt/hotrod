use hotrod::engine::system::canvas::buffered_layer::BufferedCanvasLayer;
use hotrod::engine::system::vulkan::beautiful_lines::{BeautifulLine, Vertex2d};
use hotrod::engine::system::vulkan::textured::{Textured, TexturedIndexed, Vertex2dUv};
use hotrod::engine::system::vulkan::triangles::{Triangles, TrianglesIndexed};
use hotrod::engine::types::world2d::{Dim, Pos};
use hotrod::engine::Engine;
use hotrod::logging::LevelFilter;
use hotrod::sdl2;
use hotrod::sdl2::event::Event;
use hotrod::sdl2::keyboard::Keycode;
use hotrod::sdl2::pixels::PixelFormatEnum;
use hotrod::sdl2::ttf::{FontStyle, Hinting, Sdl2TtfContext};
use image::GenericImageView;
use std::io::Cursor;
use std::ops::{Div, Mul};
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const IMAGE_DATA: &[u8] = include_bytes!(concat!("rust-logo-256x256.png"));

fn main() {
    hotrod::logging::init_logger(Some(LevelFilter::Info)).expect("Unable to init logger");
    let mut engine = Engine::default().with_fps(144);

    let mut duration_engine = Duration::from_secs(1);
    let mut duration_loop = Duration::from_secs(1);
    let mut texture = None;
    let mut ttf = None;

    loop {
        let loop_start = Instant::now();
        let response = engine.update(|mut ctx| {
            let abort = ctx.events.iter().any(|e| match e {
                Event::Quit { .. } => true,
                Event::KeyDown { keycode, .. } => {
                    matches!(keycode, Some(Keycode::Escape))
                }
                _ => false,
            });

            ctx.update_egui(|ctx| {
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
            });

            ctx.render(|context| {
                let mut buffers = Vec::default();

                if texture.is_none() {
                    let mut commands = context.inner.create_preparation_buffer_builder().unwrap();

                    let image = image::io::Reader::new(Cursor::new(IMAGE_DATA))
                        .with_guessed_format()
                        .unwrap()
                        .decode()
                        .unwrap();

                    texture = Some(
                        context
                            .pipelines
                            .texture
                            .create_texture(
                                &mut commands,
                                image
                                    .pixels()
                                    .flat_map(|(_x, _y, rgba)| rgba.0)
                                    .collect::<Vec<u8>>(),
                                image.width(),
                                image.height(),
                            )
                            .unwrap(),
                    );

                    buffers.push(commands.build().unwrap());
                }

                if ttf.is_none() {
                    let mut commands = context.inner.create_preparation_buffer_builder().unwrap();
                    let ttf_ctxt = Sdl2TtfContext;

                    //
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

                    let texture = context
                        .pipelines
                        .texture
                        .create_texture(&mut commands, data, surface.width(), surface.height())
                        .unwrap();

                    ttf = Some((texture, surface.height() as f32 / surface.width() as f32));
                    buffers.push(commands.build().unwrap());
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

                // #[cfg(feature = "ttf-sdl2")]
                if let Some((ttf, ratio)) = &ttf {
                    let width = 500.0;
                    layer.draw_textured_rect(
                        Pos::new(500.0, 500.0),
                        Dim::new(width, width * ratio),
                        ttf.clone(),
                    );
                }

                buffers.push(layer.flush(context.inner, context.pipelines));

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
            .map(|_| !abort)
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

    // Engine::default()
    //     .with_egui_context_callback(|ctx| {
    //         use hotrod::ui::egui::Window;
    //         Window::new("HotRod - Testing")
    //             .resizable(true)
    //             .show(ctx, |ui| {
    //                 if ui.button("CLICK ME").clicked() {
    //                     eprintln!("I WAS CLICKED!");
    //                 }
    //             });
    //     })
    //     .run();
}
