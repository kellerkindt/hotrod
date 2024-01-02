use std::sync::Arc;
use vulkano::device::Device;
use vulkano::image::sampler::{Filter, Sampler, SamplerCreateInfo, SamplerMipmapMode};
use vulkano::{Validated, VulkanError};

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum ImageSamplerMode {
    PixelPerfect,
    Linear,
}

impl ImageSamplerMode {
    #[inline]
    pub fn into_filter(self) -> Filter {
        match self {
            ImageSamplerMode::PixelPerfect => Filter::Nearest,
            ImageSamplerMode::Linear => Filter::Linear,
        }
    }

    #[inline]
    pub fn into_mipmap_mode(self) -> SamplerMipmapMode {
        match self {
            ImageSamplerMode::PixelPerfect => SamplerMipmapMode::Nearest,
            ImageSamplerMode::Linear => SamplerMipmapMode::Linear,
        }
    }

    #[inline]
    pub fn create_texture_sampler(
        self,
        device: Arc<Device>,
    ) -> Result<Arc<Sampler>, Validated<VulkanError>> {
        Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: self.into_filter(),
                min_filter: self.into_filter(),
                mipmap_mode: self.into_mipmap_mode(),
                ..Default::default()
            },
        )
    }
}
