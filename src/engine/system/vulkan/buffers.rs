use bytemuck::Pod;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferAllocateError, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::Validated;

pub struct BasicBuffersManager {
    memo_allocator: Arc<StandardMemoryAllocator>,
}

impl BasicBuffersManager {
    #[inline]
    pub fn new(memo_allocator: Arc<StandardMemoryAllocator>) -> Self {
        Self { memo_allocator }
    }

    pub fn create_index_buffer<I>(
        &self,
        indices: I,
    ) -> Result<Subbuffer<[u32]>, Validated<BufferAllocateError>>
    where
        I: IntoIterator<Item = u32>,
        I::IntoIter: ExactSizeIterator,
    {
        Buffer::from_iter(
            &self.memo_allocator,
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

    pub fn create_vertex_buffer<I, T: Send + Sync + Pod>(
        &self,
        vertices: I,
    ) -> Result<Subbuffer<[T]>, Validated<BufferAllocateError>>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        Buffer::from_iter(
            &self.memo_allocator,
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
