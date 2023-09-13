use crate::engine::system::vulkan::Error;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::{WriteDescriptorSet, WriteDescriptorSetElements};
use vulkano::memory::allocator::{
    AllocationCreateInfo, GenericMemoryAllocator, MemoryTypeFilter, Suballocator,
};

pub mod binding_101_window_size;

pub trait WriteDescriptorSetOrigin {
    type BufferContents: BufferContents;
    type Data: ExactSizeIterator<Item = Self::BufferContents>;

    fn binding(&self) -> u32;

    fn data(&self) -> Self::Data;

    fn create_descriptor_set<T: Suballocator>(
        &self,
        memory_allocator: &GenericMemoryAllocator<T>,
    ) -> Result<WriteDescriptorSet, Error> {
        Ok(WriteDescriptorSet::buffer(
            self.binding(),
            Buffer::from_iter::<Self::BufferContents, _>(
                memory_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                    ..BufferCreateInfo::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..AllocationCreateInfo::default()
                },
                self.data(),
            )
            .map_err(|e| Error::FailedToAllocateWriteDescriptorBuffer(e, self.binding()))?,
        ))
    }

    fn update<T, A: CommandBufferAllocator>(
        &self,
        cmds: &mut AutoCommandBufferBuilder<T, A>,
        current: &WriteDescriptorSet,
    ) -> Result<(), Error> {
        if let WriteDescriptorSetElements::Buffer(buffer) = current.elements() {
            cmds.update_buffer(
                buffer[0]
                    .buffer
                    .clone()
                    .cast_aligned::<Self::BufferContents>(),
                self.data().collect::<Box<_>>(),
            )
            .map(drop)
            .map_err(|e| Error::FailedToUpdateWriteDescriptorBuffer(e, self.binding()))
        } else {
            unimplemented!()
        }
    }
}
