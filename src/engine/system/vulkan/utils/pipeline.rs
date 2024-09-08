use std::sync::Arc;
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::SampleCount;
use vulkano::render_pass::RenderPass;
use vulkano::{Validated, VulkanError};

pub fn single_pass_render_pass_from_image_format(
    device: Arc<Device>,
    image_format: Format,
    samples: SampleCount,
) -> Result<Arc<RenderPass>, Validated<VulkanError>> {
    if samples == SampleCount::Sample1 {
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
    } else {
        vulkano::single_pass_renderpass!(
            device,
            attachments: {
                intermediary: {
                    format: image_format,
                    // This has to match the image definition.
                    samples: samples,
                    load_op: Clear,
                    store_op: DontCare,
                },
                color: {
                    format: image_format,
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
            },
            pass: {
                color: [intermediary],
                color_resolve: [color],
                depth_stencil: {},
            }
        )
    }
}
