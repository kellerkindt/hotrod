use crate::engine::system::vulkan::system::WriteDescriptorSetCollection;
use std::sync::Arc;
use vulkano::descriptor_set::allocator::{DescriptorSetAllocator, StandardDescriptorSetAlloc};
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;
use vulkano::render_pass::{RenderPass, Subpass};
use vulkano::{Validated, VulkanError};

#[inline]
pub fn subpass_from_renderpass(
    render_pass: Arc<RenderPass>,
) -> Result<PipelineSubpassType, Validated<VulkanError>> {
    Ok(Subpass::from(render_pass, 0)
        .expect("Missing Subpass in single pass Renderpass")
        .into())
}

pub fn single_pass_render_pass_from_image_format(
    device: Arc<Device>,
    image_format: Format,
) -> Result<Arc<RenderPass>, Validated<VulkanError>> {
    vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                format: image_format,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        }
    )
}

pub fn create_persistent_descriptor_set_from_collection(
    layout: &Arc<DescriptorSetLayout>,
    allocator: &impl DescriptorSetAllocator<Alloc = StandardDescriptorSetAlloc>,
    write_descriptors: &WriteDescriptorSetCollection,
) -> Result<Arc<PersistentDescriptorSet>, Validated<VulkanError>> {
    PersistentDescriptorSet::new(
        allocator,
        Arc::clone(layout),
        write_descriptor_sets_from_collection(layout, write_descriptors),
        [],
    )
}

#[inline]
pub fn write_descriptor_sets_from_collection<'a>(
    layout: &'a DescriptorSetLayout,
    write_descriptors: &'a WriteDescriptorSetCollection,
) -> impl Iterator<Item = WriteDescriptorSet> + 'a {
    layout
        .bindings()
        .keys()
        .flat_map(|binding| write_descriptors.get(binding).cloned())
}
