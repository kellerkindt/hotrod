use crate::engine::system::texture::Texture;
use crate::engine::system::vulkan::textures::ImageSystem;
use crate::engine::system::vulkan::UploadError;
use std::sync::Arc;
use vulkano::image::Image;
use vulkano::{Validated, VulkanError};

pub trait TextureLoaderExt {
    #[cfg(feature = "image")]
    fn load_texture_from_raw_image<'a, R: 'a + std::io::BufRead + std::io::Seek>(
        &self,
        bin: R,
    ) -> Result<Texture, Error>;
}

impl TextureLoaderExt for &ImageSystem {
    #[cfg_attr(feature = "image", inline)]
    #[cfg(feature = "image")]
    fn load_texture_from_raw_image<'a, R: 'a + std::io::BufRead + std::io::Seek>(
        &self,
        bin: R,
    ) -> Result<Texture, Error> {
        TextureLoader::load_from_binary(self, bin)
    }
}

pub struct TextureLoader;

impl TextureLoader {
    #[cfg(feature = "image")]
    pub fn load_from_binary<'a, R: 'a + std::io::BufRead + std::io::Seek>(
        image_system: &ImageSystem,
        bin: R,
    ) -> Result<Texture, Error> {
        use image::GenericImageView;
        let mem_image = Self::read_image(bin)?;
        let gpu_image = Self::upload_image(
            image_system,
            mem_image
                .pixels()
                .flat_map(|(_x, _y, rgba)| rgba.0)
                .collect::<Vec<u8>>(),
            mem_image.width(),
            mem_image.height(),
        )?;

        Ok(Texture {
            vulkan_image: gpu_image,
            memory_image: mem_image,
            egui_texture: None,
            texture_ids: vec![],
        })
    }

    #[cfg(feature = "image")]
    pub fn read_image<'a, R: 'a + std::io::BufRead + std::io::Seek>(
        bin: R,
    ) -> Result<image::DynamicImage, Error> {
        Ok(image::ImageReader::new(bin)
            .with_guessed_format()
            .map_err(Error::UnableToLoad)?
            .decode()
            .map_err(Error::UnableToDecode)?)
    }

    pub fn upload_image(
        image_system: &ImageSystem,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    ) -> Result<Arc<Image>, Error> {
        Ok(image_system
            .create_image_and_enqueue_upload(rgba, width, height)
            .map_err(Error::FailedToUpload)?)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Unable to load the image: {0}")]
    UnableToLoad(std::io::Error),
    #[cfg_attr(feature = "image", error("Unable to decode the image: {0}"))]
    #[cfg(feature = "image")]
    UnableToDecode(image::ImageError),
    #[error("Unable to upload the image: {0}")]
    FailedToUpload(UploadError),
    #[error("Vulkan error: {0}")]
    VulkanError(Validated<VulkanError>),
}
