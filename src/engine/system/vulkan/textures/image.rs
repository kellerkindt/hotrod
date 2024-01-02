use crate::engine::system::vulkan::{PipelineCreateError, UploadError};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferAllocateError, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferToImageInfo};
use vulkano::format::Format;
use vulkano::image::{Image, ImageAllocateError, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::Validated;

pub struct ImageSystem {
    memo_allocator: Arc<StandardMemoryAllocator>,
}

impl ImageSystem {
    pub fn new(memo_allocator: Arc<StandardMemoryAllocator>) -> Result<Self, PipelineCreateError> {
        Ok(Self { memo_allocator })
    }

    pub fn create_and_upload_image<P>(
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    ) -> Result<Arc<Image>, UploadError> {
        let image = self.create_image(width, height)?;
        self.upload_image(builder, Arc::clone(&image), rgba)?;
        Ok(image)
    }

    pub fn create_image(
        &self,
        width: u32,
        height: u32,
    ) -> Result<Arc<Image>, Validated<ImageAllocateError>> {
        Image::new(
            &self.memo_allocator,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                extent: [width, height, 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..ImageCreateInfo::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..AllocationCreateInfo::default()
            },
        )
    }

    pub fn upload_image<P, I>(
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        image: Arc<Image>,
        rgba: I,
    ) -> Result<(), Validated<BufferAllocateError>>
    where
        I: IntoIterator<Item = u8>,
        I::IntoIter: ExactSizeIterator,
    {
        builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            Buffer::from_iter(
                &self.memo_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..BufferCreateInfo::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..AllocationCreateInfo::default()
                },
                rgba,
            )?,
            image,
        ))?;
        Ok(())
    }

    pub fn update_image<P, I>(
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        image: Arc<Image>,
        region: Option<([u32; 2], [u32; 2])>,
        rgba: I,
    ) -> Result<(), Validated<BufferAllocateError>>
    where
        I: IntoIterator<Item = u8>,
        I::IntoIter: ExactSizeIterator,
    {
        builder.copy_buffer_to_image({
            let mut copy_info = CopyBufferToImageInfo::buffer_image(
                Buffer::from_iter(
                    &self.memo_allocator,
                    BufferCreateInfo {
                        usage: BufferUsage::TRANSFER_SRC,
                        ..BufferCreateInfo::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..AllocationCreateInfo::default()
                    },
                    rgba,
                )?,
                image,
            );

            if let Some(([x, y], [width, height])) = region {
                copy_info.regions[0].image_offset[0] = x;
                copy_info.regions[0].image_offset[1] = y;
                copy_info.regions[0].image_extent[0] = width;
                copy_info.regions[0].image_extent[1] = height;
            }

            copy_info
        })?;

        Ok(())
    }
}
