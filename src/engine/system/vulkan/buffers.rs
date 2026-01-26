use bytemuck::Pod;
use std::sync::Arc;
use vulkano::buffer::{AllocateBufferError, Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter};
use vulkano::Validated;

pub struct BasicBuffersManager {
    pub(crate) memo_allocator: Arc<dyn MemoryAllocator>,
}

impl BasicBuffersManager {
    #[inline]
    pub fn new(memo_allocator: impl MemoryAllocator) -> Self {
        Self {
            memo_allocator: Arc::new(memo_allocator),
        }
    }

    #[inline]
    pub fn create_index_buffer<I>(
        &self,
        indices: I,
    ) -> Result<Subbuffer<[u32]>, Validated<AllocateBufferError>>
    where
        I: IntoIterator<Item = u32>,
        I::IntoIter: ExactSizeIterator,
    {
        Buffer::from_iter(
            Arc::clone(&self.memo_allocator),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..BufferCreateInfo::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..AllocationCreateInfo::default()
            },
            indices,
        )
    }

    #[inline]
    pub fn create_vertex_buffer<I, T: Send + Sync + Pod>(
        &self,
        vertices: I,
    ) -> Result<Subbuffer<[T]>, Validated<AllocateBufferError>>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        #[cfg(debug_assertions)]
        let vertices = {
            let vertices = vertices.into_iter();
            if vertices.len() == 0 {
                warn!("Given empty vertex iterator to create vertex buffer")
            }
            vertices
        };

        Buffer::from_iter(
            Arc::clone(&self.memo_allocator),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..BufferCreateInfo::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..AllocationCreateInfo::default()
            },
            vertices,
        )
    }
}
