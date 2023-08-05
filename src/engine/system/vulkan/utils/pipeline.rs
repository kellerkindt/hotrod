use std::sync::Arc;
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
