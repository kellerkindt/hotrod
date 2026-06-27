use crate::engine::system::vulkan::{PipelineCreateError, UploadError};
use crossbeam::queue::SegQueue;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use vulkano::buffer::{AllocateBufferError, Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::CopyBufferToImageInfo;
use vulkano::format::Format;
use vulkano::image::{AllocateImageError, Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter};
use vulkano::Validated;

pub struct ImageSystem {
    memo_allocator: Arc<dyn MemoryAllocator>,
    upload_queue: SegQueue<CopyRequest>,
    upload_queue_bypass: SegQueue<CopyRequest>,
    upload_queue_condvar: Condvar,
    upload_queue_mutex: Mutex<()>,
    upload_queue_required: SegQueue<CopyRequestWaiter>,
}

impl ImageSystem {
    pub fn new(memo_allocator: impl MemoryAllocator) -> Result<Self, PipelineCreateError> {
        Ok(Self {
            memo_allocator: Arc::new(memo_allocator),
            upload_queue: Default::default(),
            upload_queue_bypass: Default::default(),
            upload_queue_condvar: Default::default(),
            upload_queue_mutex: Default::default(),
            upload_queue_required: Default::default(),
        })
    }

    /// Whether there are [`CopyBufferToImageInfo`]-requests enqueued.
    pub(crate) fn has_upload_info_enqueued(&self) -> bool {
        !self.upload_queue.is_empty()
    }

    /// Retrieves enqueued [`CopyBufferToImageInfo`]-requests.
    pub(crate) fn next_upload_info(&self) -> Option<CopyRequest> {
        self.upload_queue.pop()
    }

    /// Retrieves enqueued [`CopyBufferToImageInfo`]-requests. Waits until one is available.
    pub(crate) fn wait_for_next_upload_info(&self, duration: Duration) -> Option<CopyRequest> {
        let until = Instant::now() + duration;
        self.wait_for_next_upload_info_until(until)
    }

    /// Retrieves enqueued [`CopyBufferToImageInfo`]-requests. Waits until one is available.
    pub(crate) fn wait_for_next_upload_info_until(&self, until: Instant) -> Option<CopyRequest> {
        let mut now = Instant::now();
        let mut guard = self.upload_queue_mutex.lock().unwrap();
        while now < until {
            if let Some(info) = self.upload_queue_bypass.pop() {
                return Some(info);
            } else if let Some(info) = self.upload_queue.pop() {
                return Some(info);
            } else {
                let result = self
                    .upload_queue_condvar
                    .wait_timeout(guard, until.saturating_duration_since(now))
                    .unwrap();
                guard = result.0;
                if result.1.timed_out() {
                    break;
                }
            }

            now = Instant::now();
        }
        None
    }

    /// Tries to retrieve the next [`CopyRequestWaiter`] that needs to finish before the next render
    /// call.
    pub(crate) fn next_required_waiter(&self) -> Option<CopyRequestWaiter> {
        self.upload_queue_required.pop()
    }

    /// Creates a new [`Image`] and enqueues an upload-request the given `rgba`-data as content.
    pub fn create_image_and_enqueue_upload<I>(
        &self,
        rgba: I,
        width: u32,
        height: u32,
        required_before_render: bool,
    ) -> Result<(Arc<Image>, CopyRequestWaiter), UploadError>
    where
        I: IntoIterator<Item = u8>,
        I::IntoIter: ExactSizeIterator,
    {
        let image = self.create_image(width, height)?;
        let waiter = self.enqueue_image_upload(Arc::clone(&image), rgba, required_before_render)?;
        Ok((image, waiter))
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
        required_before_render: bool,
    ) -> Result<CopyRequestWaiter, Validated<AllocateBufferError>>
    where
        I: IntoIterator<Item = u8>,
        I::IntoIter: ExactSizeIterator,
    {
        let rgba = rgba.into_iter();
        let estimated_bytes = rgba.len() * 4; // RGBA / Color32
        Ok(self.enqueue_copy_request(
            CopyInfo::Immediate(self.create_copy_buffer_to_image_image(image, rgba)?),
            estimated_bytes,
            required_before_render,
        ))
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
        required_before_render: bool,
    ) -> Result<CopyRequestWaiter, Validated<AllocateBufferError>>
    where
        I: IntoIterator<Item = u8>,
        I::IntoIter: ExactSizeIterator,
    {
        let rgba = rgba.into_iter();
        let estimated_bytes = rgba.len() * 4; // RGBA / Color32
        let info = self.create_copy_request(image, region, rgba)?;
        Ok(self.enqueue_copy_request(
            CopyInfo::Immediate(info),
            estimated_bytes,
            required_before_render,
        ))
    }

    pub fn create_copy_request<I>(
        &self,
        image: Arc<Image>,
        region: Option<([u32; 2], [u32; 2])>,
        rgba: I,
    ) -> Result<CopyBufferToImageInfo, Validated<AllocateBufferError>>
    where
        I: IntoIterator<Item = u8>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut copy_info = self.create_copy_buffer_to_image_image(image, rgba)?;

        if let Some(([x, y], [width, height])) = region {
            copy_info.regions[0].image_offset[0] = x;
            copy_info.regions[0].image_offset[1] = y;
            copy_info.regions[0].image_extent[0] = width;
            copy_info.regions[0].image_extent[1] = height;
        }

        Ok(copy_info)
    }

    pub fn enqueue_copy_request(
        &self,
        info: CopyInfo,
        estimated_bytes: usize,
        required_before_render: bool,
    ) -> CopyRequestWaiter {
        let condvar = Arc::new((Condvar::new(), Mutex::new(false)));
        let waiter = CopyRequestWaiter { condvar };
        let request = CopyRequest {
            info,
            response: CopyRequestNotifier {
                condvar: Arc::clone(&waiter.condvar),
            },
            estimated_bytes,
            required_before_render,
        };

        if required_before_render {
            self.upload_queue_required.push(waiter.clone());
            self.upload_queue_bypass.push(request);
        } else {
            self.upload_queue.push(request);
        }

        self.upload_queue_condvar.notify_all();
        waiter
    }
}

pub enum CopyInfo {
    // Allows more heavy computation being offloaded to the transfer thread
    Deferred(
        Box<
            dyn FnOnce(
                    &ImageSystem,
                )
                    -> Result<CopyBufferToImageInfo, Validated<AllocateBufferError>>
                + Send
                + Sync,
        >,
    ),
    Immediate(CopyBufferToImageInfo),
}

pub(crate) struct CopyRequest {
    pub info: CopyInfo,
    pub response: CopyRequestNotifier,
    pub estimated_bytes: usize,
    pub required_before_render: bool,
}

pub(crate) struct CopyRequestNotifier {
    condvar: Arc<(Condvar, Mutex<bool>)>,
}

impl CopyRequestNotifier {
    pub fn notify_completion(self) {
        debug!("Notifying completion");
        let mut guard = self.condvar.1.lock().unwrap();
        *guard = true;
        self.condvar.0.notify_all();
    }
}

#[derive(Clone)]
pub struct CopyRequestWaiter {
    condvar: Arc<(Condvar, Mutex<bool>)>,
}

impl CopyRequestWaiter {
    #[inline]
    pub fn is_complete(&self) -> bool {
        *self.condvar.1.lock().unwrap()
    }

    pub fn wait_for_completion(self) {
        let mut guard = self.condvar.1.lock().unwrap();
        while !*guard {
            guard = self.condvar.0.wait(guard).unwrap();
        }
        debug!("Completion awaited")
    }
}
