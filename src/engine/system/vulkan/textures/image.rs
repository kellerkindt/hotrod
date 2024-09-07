use crate::engine::system::vulkan::{PipelineCreateError, UploadError};
use crossbeam::queue::SegQueue;
use std::sync::Arc;
use vulkano::buffer::{AllocateBufferError, Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::CopyBufferToImageInfo;
use vulkano::format::Format;
use vulkano::image::{AllocateImageError, Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter};
use vulkano::Validated;

pub struct ImageSystem {
    memo_allocator: Arc<dyn MemoryAllocator>,
    upload_queue: SegQueue<CopyBufferToImageInfo>,
}

impl ImageSystem {
    pub fn new(memo_allocator: impl MemoryAllocator) -> Result<Self, PipelineCreateError> {
        Ok(Self {
            memo_allocator: Arc::new(memo_allocator),
            upload_queue: Default::default(),
        })
    }

    /// Whether there are [`CopyBufferToImageInfo`]-requests enqueued.
    pub(crate) fn has_upload_info_enqueued(&self) -> bool {
        !self.upload_queue.is_empty()
    }

    /// Retrieves enqueued [`CopyBufferToImageInfo`]-requests.
    pub(crate) fn next_upload_info(&self) -> Option<CopyBufferToImageInfo> {
        self.upload_queue.pop()
    }

    /// Creates a new [`Image`] and enqueues an upload-request the given `rgba`-data as content.
    pub fn create_and_upload_image(
        &self,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    ) -> Result<Arc<Image>, UploadError> {
        let image = self.create_image(width, height)?;
        self.enqueue_image_upload(Arc::clone(&image), rgba)?;
        Ok(image)
    }

    #[inline]
    pub fn create_image(
        &self,
        width: u32,
        height: u32,
    ) -> Result<Arc<Image>, Validated<AllocateImageError>> {
        Image::new(
            Arc::clone(&self.memo_allocator),
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

    #[inline]
    pub fn enqueue_image_upload<I>(
        &self,
        image: Arc<Image>,
        rgba: I,
    ) -> Result<(), Validated<AllocateBufferError>>
    where
        I: IntoIterator<Item = u8>,
        I::IntoIter: ExactSizeIterator,
    {
        self.upload_queue
            .push(self.create_copy_buffer_to_image_image(image, rgba)?);
        Ok(())
    }

    fn create_copy_buffer_to_image_image<I>(
        &self,
        image: Arc<Image>,
        rgba: I,
    ) -> Result<CopyBufferToImageInfo, Validated<AllocateBufferError>>
    where
        I: IntoIterator<Item = u8>,
        I::IntoIter: ExactSizeIterator,
    {
        Ok(CopyBufferToImageInfo::buffer_image(
            Buffer::from_iter(
                Arc::clone(&self.memo_allocator),
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
        ))
    }

    pub fn enqueue_image_update<I>(
        &self,
        image: Arc<Image>,
        region: Option<([u32; 2], [u32; 2])>,
        rgba: I,
    ) -> Result<(), Validated<AllocateBufferError>>
    where
        I: IntoIterator<Item = u8>,
        I::IntoIter: ExactSizeIterator,
    {
        self.upload_queue.push({
            let mut copy_info = self.create_copy_buffer_to_image_image(image, rgba)?;

            if let Some(([x, y], [width, height])) = region {
                copy_info.regions[0].image_offset[0] = x;
                copy_info.regions[0].image_offset[1] = y;
                copy_info.regions[0].image_extent[0] = width;
                copy_info.regions[0].image_extent[1] = height;
            }

            copy_info
        });

        Ok(())
    }
}
