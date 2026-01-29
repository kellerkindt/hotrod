use crate::engine::system::texture::TextureView;
use crate::engine::system::vulkan::textures::TextureId;
use crate::engine::system::vulkan::PipelineTextureLoader;
use crate::engine::types::world2d::Pos;

#[derive(Default)]
pub struct SpriteAnimationLoader {
    padding: [f32; 4],
    sprite_size: Option<(f32, f32)>,
}

impl SpriteAnimationLoader {
    pub fn with_padding(mut self, padding: f32) -> Self {
        self.padding = [padding; 4];
        self
    }

    pub fn with_sprite_size(mut self, width: f32, height: f32) -> Self {
        self.sprite_size = Some((width, height));
        self
    }

    pub fn load_sprites_from_texture<P: PipelineTextureLoader + 'static>(
        &self,
        view: &TextureView,
    ) -> Option<Vec<Sprite<P>>> {
        let texture_id = view.texture().get_texture_id::<P>()?;
        let image_width = view.width() as u32;
        let image_height = view.height() as u32;

        let (sprite_width, sprite_height) = self.sprite_size.unwrap_or_else(|| {
            let size = image_width.min(image_height) as f32;
            (size, size)
        });
        let sprite_size_padded_w = sprite_width - self.padding[1] - self.padding[3];
        let sprite_size_padded_h = sprite_height - self.padding[0] - self.padding[2];

        let origin_x = self.padding[3];
        let origin_y = self.padding[0];

        let elements = (image_width / image_height).max(image_height / image_width);

        let stride_x = (image_width / image_height).min(1) as f32;
        let stride_y = (image_height / image_width).min(1) as f32;

        Some(
            (0..elements)
                .map(|i| {
                    let i = i as f32;

                    let subview = view.subview(
                        [
                            (origin_x + (i * stride_x * sprite_width)) / image_width as f32,
                            (origin_y + (i * stride_y * sprite_height)) / image_height as f32,
                        ],
                        [
                            (origin_x + (i * stride_x * sprite_width) + sprite_size_padded_w)
                                / image_width as f32,
                            (origin_y + (i * stride_y * sprite_height) + sprite_size_padded_h)
                                / image_height as f32,
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

#[cfg(feature = "image")]
mod image_ext {
    use crate::engine::system::texture::{Error, TextureLoaderExt};
    use crate::engine::system::vulkan::textures::ImageSystem;
    use crate::engine::system::vulkan::PipelineTextureLoader;
    use crate::support::sprite_animation::{Sprite, SpriteAnimationLoader};
    use std::io::{BufRead, Cursor, Seek};

    impl SpriteAnimationLoader {
        pub fn load_sprites<'a, P: PipelineTextureLoader + 'static, C: 'a>(
            &self,
            image_system: &ImageSystem,
            loader: &P,
            image: C,
        ) -> Result<Vec<Sprite<P>>, Error>
        where
            Cursor<C>: 'a + BufRead + Seek,
        {
            let mut texture = image_system.load_texture_from_raw_image(Cursor::new(image))?;

            texture
                .load_and_register_for(loader)
                .map_err(Error::VulkanError)?;

            Ok(self
                .load_sprites_from_texture::<P>(&texture.finalize())
                .unwrap())
        }
    }
}

pub struct Sprite<P> {
    pub texture: TextureId<P>,
    pub view: TextureView,
    pub uv0: Pos<f32>,
    pub uv1: Pos<f32>,
}
