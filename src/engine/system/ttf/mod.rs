use crate::engine::system::vulkan::textured::{Textured, TexturedPipeline, Vertex2dUv};
use crate::engine::system::vulkan::textures::{ImageSystem, TextureId};
use crossbeam::channel::Receiver;
use crossbeam::channel::Sender;
use crossbeam::queue::SegQueue;
use fnv::FnvHashMap;
use sdl2::pixels::{Color, PixelFormatEnum};
use sdl2::rwops::RWops;
use sdl2::ttf::{Font, Sdl2TtfContext};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

type CacheUpdate = (String, Vec<u8>, u32, u32);

pub struct FontRenderer {
    dummy_image: Option<TextureId<TexturedPipeline>>,
    cache: FnvHashMap<String, (TextureId<TexturedPipeline>, f32, f32, u8)>,
    sender: Sender<FontRenderRequest>,
    update_queue: Arc<SegQueue<CacheUpdate>>,
}

impl FontRenderer {
    const DUMMY_TEXTURE_WIDTH: u32 = 1;
    const DUMMY_TEXTURE_HEIGHT: u32 = 1;
    const DUMMY_TEXTURE_RGBA: [u8; 4] = [0, 0, 0, 0];
    const DEFAULT_LAST_USED_COUNTER: u8 = 0;

    pub fn new(ttf: Cow<'static, [u8]>) -> Self {
        let update_queue = Arc::default();
        let sender = FontRendererThread::spawn(ttf, Arc::clone(&update_queue));

        Self {
            dummy_image: None,
            cache: FnvHashMap::default(),
            sender,
            update_queue,
        }
    }

    pub fn on_frame_completed(&mut self) {
        let mut remove = Vec::default();
        for (key, (_, _, _, counter)) in self.cache.iter_mut() {
            if *counter > 254 {
                remove.push(key.clone());
            } else {
                *counter += 1;
            }
        }
        for key in remove {
            self.cache.remove(&key);
        }
    }

    #[must_use]
    #[instrument(level = "trace", skip(self, textured_pipeline, image_system))]
    pub fn prepare_render(
        &mut self,
        textured_pipeline: &TexturedPipeline,
        image_system: &ImageSystem,
        text: &str,
        size: u16,
        color: [u8; 4],
        x: f32,
        y: f32,
    ) -> Textured {
        self.retrieve_threaded_updates(textured_pipeline, image_system);

        let (texture, w, h) = match self.cache.get_mut(text) {
            // Fine, it already exists, just reset the counter
            Some((texture_id, w, h, counter)) => {
                *counter = Self::DEFAULT_LAST_USED_COUNTER;
                (texture_id.clone(), *w, *h)
            }
            // In this scenario, the text is submitted for rendering to the separate thread while
            // this context continues on returning a `Textured` instance with a dummy texture.
            None => {
                if let Err(e) = self.sender.send(FontRenderRequest {
                    size,
                    color,
                    text: text.to_string(),
                }) {
                    error!("Failed to send FontRenderRequest: {e}");
                }

                let dummy_texture =
                    self.get_or_create_dummy_texture(textured_pipeline, image_system);

                self.cache.insert(
                    text.to_string(),
                    (
                        dummy_texture.clone(),
                        Self::DUMMY_TEXTURE_WIDTH as f32,
                        Self::DUMMY_TEXTURE_HEIGHT as f32,
                        Self::DEFAULT_LAST_USED_COUNTER,
                    ),
                );

                (
                    dummy_texture,
                    Self::DUMMY_TEXTURE_WIDTH as f32,
                    Self::DUMMY_TEXTURE_HEIGHT as f32,
                )
            }
        };

        Textured {
            vertices: vec![
                Vertex2dUv {
                    pos: [x, y],
                    uv: [0.0, 0.0],
                },
                Vertex2dUv {
                    pos: [x + w, y],
                    uv: [1.0, 0.0],
                },
                Vertex2dUv {
                    pos: [x + w, y + h],
                    uv: [1.0, 1.0],
                },
                Vertex2dUv {
                    pos: [x + w, y + h],
                    uv: [1.0, 1.0],
                },
                Vertex2dUv {
                    pos: [x, y + h],
                    uv: [0.0, 1.0],
                },
                Vertex2dUv {
                    pos: [x, y],
                    uv: [0.0, 0.0],
                },
            ],
            texture,
        }
    }

    fn get_or_create_dummy_texture(
        &mut self,
        textured_pipeline: &TexturedPipeline,
        image_system: &ImageSystem,
    ) -> TextureId<TexturedPipeline> {
        self.dummy_image.clone().unwrap_or_else(|| {
            let image = image_system
                .create_image_and_enqueue_upload(
                    Self::DUMMY_TEXTURE_RGBA,
                    Self::DUMMY_TEXTURE_WIDTH,
                    Self::DUMMY_TEXTURE_HEIGHT,
                )
                .unwrap();

            let texture = textured_pipeline.prepare_texture(image).unwrap();

            self.dummy_image = Some(texture.clone());
            texture
        })
    }

    fn retrieve_threaded_updates(
        &mut self,
        textured_pipeline: &TexturedPipeline,
        image_system: &ImageSystem,
    ) {
        while let Some((text, image_data, w, h)) = self.update_queue.pop() {
            let image = image_system
                .create_image_and_enqueue_upload(image_data, w, h)
                .unwrap();
            let texture = textured_pipeline.prepare_texture(image).unwrap();
            self.cache.insert(text, (texture, w as f32, h as f32, 0));
        }
    }
}

struct FontRenderRequest {
    size: u16,
    color: [u8; 4],
    text: String,
}

struct FontRendererThread<'a> {
    ctx: &'a Sdl2TtfContext,
    ttf: &'a [u8],
    fonts: FnvHashMap<u16, Font<'a, 'a>>,
    receiver: Receiver<FontRenderRequest>,
    result_queue: Arc<SegQueue<CacheUpdate>>,
}

impl<'a> FontRendererThread<'a> {
    pub fn spawn(
        ttf: Cow<'static, [u8]>,
        result_queue: Arc<SegQueue<CacheUpdate>>,
    ) -> Sender<FontRenderRequest> {
        let (sender, receiver) = crossbeam::channel::unbounded();
        if let Err(e) = std::thread::Builder::new()
            .name("FontRendererThread".to_string())
            .spawn(move || {
                let ctx = Sdl2TtfContext;
                FontRendererThread {
                    ctx: &ctx,
                    ttf: ttf.as_ref(),
                    fonts: HashMap::default(),
                    receiver,
                    result_queue,
                }
                .run()
            })
        {
            error!("Failed to start FontRenderer Thread: {e}");
        }
        sender
    }

    fn run(mut self) {
        while let Ok(request) = self.receiver.recv() {
            self.process_request(request.text, request.size, request.color);
        }
    }

    #[instrument(level = "info", skip(self))]
    fn process_request(&mut self, text: String, size: u16, [r, g, b, a]: [u8; 4]) {
        let font = self
            .fonts
            .entry(size)
            .or_insert_with(|| Self::load_font_for_size(self.ctx, self.ttf, size));

        let surface = font.render(&text).blended(Color::RGBA(r, g, b, a)).unwrap();

        let surface = surface.convert_format(PixelFormatEnum::RGBA32).unwrap();
        let data = surface.without_lock().unwrap().to_vec();

        let w = surface.width();
        let h = surface.height();

        self.result_queue.push((text, data, w, h));
    }

    #[instrument(level = "info", skip(ctx, data))]
    fn load_font_for_size<'ctx, 'data>(
        ctx: &'ctx Sdl2TtfContext,
        data: &'data [u8],
        size: u16,
    ) -> Font<'ctx, 'data> {
        ctx.load_font_from_rwops(RWops::from_bytes(data).unwrap(), size)
            .unwrap()
    }
}
