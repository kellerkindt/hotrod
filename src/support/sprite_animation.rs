use crate::engine::system::texture::TextureLoaderExt;
use crate::engine::system::texture::{Error as TextureLoaderError, TextureView};
use crate::engine::system::vulkan::textures::{ImageSystem, TextureId};
use crate::engine::system::vulkan::PipelineTextureLoader;
use crate::engine::types::world2d::Pos;
use std::io::{BufRead, Cursor, Seek};

pub struct SpriteAnimationLoader<'i> {
    image_system: &'i ImageSystem,
    padding: [f32; 4],
    sprite_size: Option<(f32, f32)>,
}

impl<'i> SpriteAnimationLoader<'i> {
    pub fn new(image_system: &'i ImageSystem) -> Self {
        Self {
            image_system,
            padding: [0.0; 4],
            sprite_size: None,
        }
    }

    pub fn with_padding(mut self, padding: f32) -> Self {
        self.padding = [padding; 4];
        self
    }

    pub fn with_sprite_size(mut self, width: f32, height: f32) -> Self {
        self.sprite_size = Some((width, height));
        self
    }

    pub fn load_sprites<'a, P: PipelineTextureLoader + 'static, C: 'a>(
        &self,
        loader: &P,
        image: C,
    ) -> Result<Vec<Sprite<P>>, TextureLoaderError>
    where
        Cursor<C>: 'a + BufRead + Seek,
    {
        let mut texture = self
            .image_system
            .load_texture_from_raw_image(Cursor::new(image))?;

        texture
            .load_and_register_for(loader)
            .map_err(TextureLoaderError::VulkanError)?;

        Ok(self
            .load_sprites_from_texture::<P>(&texture.finalize())
            .unwrap())
    }

    pub fn load_sprites_from_texture<P: PipelineTextureLoader + 'static>(
        &self,
        view: &TextureView,
    ) -> Option<Vec<Sprite<P>>> {
        let texture_id = view.texture().get_texture_id::<P>()?;
        let mem_image = view.texture().memory_image();

        let image_width = mem_image.width() as f32;
        let image_height = mem_image.height() as f32;
        let (sprite_width, sprite_height) = self.sprite_size.unwrap_or_else(|| {
            let size = image_width.min(image_height);
            (size, size)
        });
        let sprite_size_padded_w = sprite_width - self.padding[1] - self.padding[3];
        let sprite_size_padded_h = sprite_height - self.padding[0] - self.padding[2];

        let origin_x = self.padding[3];
        let origin_y = self.padding[0];

        let elements =
            (mem_image.width() / mem_image.height()).max(mem_image.height() / mem_image.width());

        let stride_x = (mem_image.width() / mem_image.height()).min(1) as f32;
        let stride_y = (mem_image.height() / mem_image.width()).min(1) as f32;

        Some(
            (0..elements)
                .map(|i| {
                    let i = i as f32;

                    let subview = view.subview(
                        [
                            (origin_x + (i * stride_x * sprite_width)) / image_width,
                            (origin_y + (i * stride_y * sprite_height)) / image_height,
                        ],
                        [
                            (origin_x + (i * stride_x * sprite_width) + sprite_size_padded_w)
                                / image_width,
                            (origin_y + (i * stride_y * sprite_height) + sprite_size_padded_h)
                                / image_height,
                        ],
                    );

                    Sprite {
                        texture: texture_id.clone(),
                        uv0: subview.uv0_or_default().into(),
                        uv1: subview.uv1_or_default().into(),
                        view: subview,
                    }
                })
                .collect::<Vec<_>>(),
        )
    }
}

pub struct Sprite<P> {
    pub texture: TextureId<P>,
    pub view: TextureView,
    pub uv0: Pos<f32>,
    pub uv1: Pos<f32>,
}
