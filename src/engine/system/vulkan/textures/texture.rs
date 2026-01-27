use crate::engine::system::vulkan::textures::ImageSamplerMode;
use crate::engine::system::vulkan::PipelineCreateError;
use std::marker::PhantomData;
use std::sync::Arc;
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::image::sampler::Sampler;
use vulkano::image::view::ImageView;
use vulkano::image::Image;
use vulkano::pipeline::{GraphicsPipeline, Pipeline};
use vulkano::{Validated, VulkanError};

pub struct TextureManager<T, const BINDING: u32> {
    sampler: Arc<Sampler>,
    desc_layout: Arc<DescriptorSetLayout>,
    desc_allocator: Arc<StandardDescriptorSetAllocator>,
    origin_marker: Arc<()>,
    _t: PhantomData<T>,
}

impl<T, const BINDING: u32> TextureManager<T, BINDING> {
    #[inline]
    pub fn basic(
        device: Arc<Device>,
        pipeline: &GraphicsPipeline,
        mode: ImageSamplerMode,
    ) -> Result<Self, PipelineCreateError> {
        Ok(Self::new(
            mode.create_texture_sampler(Arc::clone(&device))?,
            Arc::clone(&pipeline.layout().set_layouts()[0]),
            Arc::new(StandardDescriptorSetAllocator::new(
                device,
                StandardDescriptorSetAllocatorCreateInfo::default(),
            )),
        ))
    }

    pub fn new(
        sampler: Arc<Sampler>,
        desc_layout: Arc<DescriptorSetLayout>,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> Self {
        Self {
            sampler,
            desc_layout,
            desc_allocator,
            origin_marker: Arc::new(()),
            _t: PhantomData::default(),
        }
    }

    #[inline]
    pub fn sampler(&self) -> &Arc<Sampler> {
        &self.sampler
    }

    #[inline]
    pub fn prepare_texture(
        &self,
        image: Arc<Image>,
        descriptors: impl Iterator<Item = WriteDescriptorSet>,
    ) -> Result<TextureId<T>, Validated<VulkanError>> {
        self.prepare_texture_with(image, Arc::clone(&self.sampler), descriptors)
    }

    pub fn prepare_texture_with(
        &self,
        image: Arc<Image>,
        sampler: Arc<Sampler>,
        descriptors: impl Iterator<Item = WriteDescriptorSet>,
    ) -> Result<TextureId<T>, Validated<VulkanError>> {
        Ok(TextureId(Arc::new(TextureInner {
            origin: Arc::clone(&self.origin_marker),
            _image: Arc::clone(&image),
            descriptor: self.create_image_desc(image, sampler, descriptors)?,
            _t: Default::default(),
        })))
    }

    fn create_image_desc(
        &self,
        image: Arc<Image>,
        sampler: Arc<Sampler>,
        descriptors: impl Iterator<Item = WriteDescriptorSet>,
    ) -> Result<Arc<DescriptorSet>, Validated<VulkanError>> {
        DescriptorSet::new(
            Arc::clone(&self.desc_allocator) as Arc<_>,
            Arc::clone(&self.desc_layout),
            [WriteDescriptorSet::image_view_sampler(
                BINDING,
                ImageView::new_default(Arc::clone(&image))?,
                sampler,
            )]
            .into_iter()
            .chain(descriptors),
            [],
        )
    }

    #[inline]
    pub fn is_origin_of(&self, texture_id: &TextureId<T>) -> bool {
        texture_id.originates_from(&self.origin_marker)
    }
}

pub struct TextureId<T: ?Sized>(pub Arc<TextureInner<T>>);

impl<T> Clone for TextureId<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T> TextureId<T> {
    #[inline]
    pub fn originates_from(&self, origin: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.0.origin, origin)
    }
}

impl<T> TextureId<T> {
    #[inline]
    pub fn descriptor(&self) -> &Arc<DescriptorSet> {
        &self.0.descriptor
    }
}

impl<T> PartialEq for TextureId<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

pub struct TextureInner<T: ?Sized> {
    pub origin: Arc<()>,
    pub _image: Arc<Image>,
    pub descriptor: Arc<DescriptorSet>,
    _t: PhantomData<fn(T) -> T>,
}
