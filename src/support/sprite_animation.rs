use crate::engine::system::vulkan::textures::{ImageSystem, TextureId};
use crate::engine::system::vulkan::{PipelineTextureLoader, UploadError};
use crate::engine::types::world2d::Pos;
use image::{DynamicImage, GenericImageView, ImageReader};
use std::io::{BufRead, Cursor, Seek};
use std::sync::Arc;
use vulkano::image::Image;
use vulkano::{Validated, VulkanError};

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

    pub fn load_sprites<'a, P: PipelineTextureLoader, C: 'a>(
        &self,
        loader: &P,
        image: C,
    ) -> Result<Vec<Sprite<P>>, Error>
    where
        Cursor<C>: 'a + BufRead + Seek,
    {
        let image = Cursor::new(image);
        let mem_image = self.read_image(image)?;
        let gpu_image = self.upload_image(
            mem_image
                .pixels()
                .flat_map(|(_x, _y, rgba)| rgba.0)
                .collect::<Vec<u8>>(),
            mem_image.width(),
            mem_image.height(),
        )?;

        let texture = loader
            .prepare_texture(gpu_image)
            .map_err(Error::VulkanError)?;

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

        Ok((0..elements)
            .map(|i| {
                let i = i as f32;

                Sprite {
                    texture: texture.clone(),
                    uv0: Pos::new(
                        (origin_x + (i * stride_x * sprite_width)) / image_width,
                        (origin_y + (i * stride_y * sprite_height)) / image_height,
                    ),
                    uv1: Pos::new(
                        (origin_x + (i * stride_x * sprite_width) + sprite_size_padded_w)
                            / image_width,
                        (origin_y + (i * stride_y * sprite_height) + sprite_size_padded_h)
                            / image_height,
                    ),
                }
            })
            .collect::<Vec<_>>())
    }

    fn read_image<'a, R: 'a + BufRead + Seek>(&self, bin: R) -> Result<DynamicImage, Error> {
        Ok(ImageReader::new(bin)
            .with_guessed_format()
            .map_err(Error::UnableToLoad)?
            .decode()
            .map_err(Error::UnableToDecode)?)
    }

    fn upload_image(&self, rgba: Vec<u8>, width: u32, height: u32) -> Result<Arc<Image>, Error> {
        Ok(self
            .image_system
            .create_image_and_enqueue_upload(rgba, width, height)
            .map_err(Error::FailedToUpload)?)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Unable to load the image: {0}")]
    UnableToLoad(std::io::Error),
    #[error("Unable to decode the image: {0}")]
    UnableToDecode(image::ImageError),
    #[error("Unable to upload the image: {0}")]
    FailedToUpload(UploadError),
    #[error("Vulkan error: {0}")]
    VulkanError(Validated<VulkanError>),
}

pub struct Sprite<P> {
    pub texture: TextureId<P>,
    pub uv0: Pos<f32>,
    pub uv1: Pos<f32>,
}
