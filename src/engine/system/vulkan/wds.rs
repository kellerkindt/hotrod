use crate::engine::system::vulkan::desc::WriteDescriptorSetOrigin;
use crate::engine::system::vulkan::Error;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::{Validated, VulkanError};

pub struct WriteDescriptorSetManager {
    desc_allocator: Arc<StandardDescriptorSetAllocator>,
    memo_allocator: Arc<StandardMemoryAllocator>,
    write_descriptor_sets: HashMap<u32, WriteDescriptorSet, nohash_hasher::BuildNoHashHasher<u32>>,
}

impl WriteDescriptorSetManager {
    pub fn new(
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
        memo_allocator: Arc<StandardMemoryAllocator>,
    ) -> Self {
        Self {
            desc_allocator,
            memo_allocator,
            write_descriptor_sets: HashMap::default(),
        }
    }

    #[inline]
    pub fn descriptor_set_allocator(&self) -> &Arc<StandardDescriptorSetAllocator> {
        &self.desc_allocator
    }

    #[inline]
    pub fn memory_allocator(&self) -> &Arc<StandardMemoryAllocator> {
        &self.memo_allocator
    }

    #[inline]
    pub fn insert<W: WriteDescriptorSetOrigin>(
        &mut self,
        origin: impl Borrow<W>,
    ) -> Result<(), Error> {
        let origin = origin.borrow();
        self.write_descriptor_sets.insert(
            origin.binding(),
            origin.create_descriptor_set(Arc::clone(&self.memo_allocator))?,
        );
        Ok(())
    }

    #[inline]
    pub fn update<T, A: CommandBufferAllocator, W: WriteDescriptorSetOrigin>(
        &self,
        cmds: &mut AutoCommandBufferBuilder<T, A>,
        origin: impl Borrow<W>,
    ) -> Result<Option<&WriteDescriptorSet>, Error> {
        let origin = origin.borrow();
        self.write_descriptor_sets
            .get(&origin.binding())
            .map(|desc| origin.update(cmds, desc).map(|_| desc))
            .transpose()
    }

    #[inline]
    pub fn create_persistent_descriptor_set(
        &self,
        layout: &Arc<DescriptorSetLayout>,
    ) -> Result<Arc<PersistentDescriptorSet>, Validated<VulkanError>> {
        let descriptor_writes = self.get_required_descriptors(&layout);
        PersistentDescriptorSet::new(
            &self.desc_allocator,
            Arc::clone(layout),
            descriptor_writes,
            [],
        )
    }

    pub fn get_required_descriptors<'a>(
        &'a self,
        layout: &'a DescriptorSetLayout,
    ) -> impl Iterator<Item = WriteDescriptorSet> + 'a {
        layout
            .bindings()
            .keys()
            .flat_map(|binding| self.write_descriptor_sets.get(binding).cloned())
    }
}
