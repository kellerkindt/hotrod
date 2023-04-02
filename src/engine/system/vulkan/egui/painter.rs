use crate::ui::egui::epaint::ImageDelta;
use bytemuck::{Pod, Zeroable};
use egui::epaint::ahash::HashMap;
use egui::{Color32, ImageData, TextureId, TexturesDelta};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferAllocateInfo, BufferUsage};
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferToImageInfo};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceOwned, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{
    ImageCreateFlags, ImageDimensions, ImageError, ImageLayout, ImageUsage, StorageImage,
};
use vulkano::instance::InstanceCreationError;
use vulkano::memory::allocator::{MemoryAllocator, MemoryUsage};
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, BlendFactor, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreationError;
use vulkano::pipeline::{GraphicsPipeline, Pipeline};
use vulkano::render_pass::Subpass;
use vulkano::sampler::{
    Filter, Sampler, SamplerCreateInfo, SamplerCreationError, SamplerMipmapMode,
};
use vulkano::shader::DescriptorBindingRequirementsIncompatible::ImageViewType;
use vulkano::shader::ShaderModule;

pub struct EguiOnVulkanoPainter {
    pub pipeline: Arc<GraphicsPipeline>,
    pub texture_sampler: Arc<Sampler>,
    pub textures: HashMap<TextureId, Arc<PersistentDescriptorSet>>,
    pub textures_to_free: Vec<TextureId>,
    pub images: HashMap<TextureId, Arc<StorageImage>>,
}

impl EguiOnVulkanoPainter {
    pub fn new(device: Arc<Device>, subpass: Subpass) -> Result<Self, PainterCreationError> {
        Ok(Self {
            pipeline: Self::create_pipeline(Arc::clone(&device), subpass)?,
            texture_sampler: Self::create_texture_sampler(device)?,
            textures: HashMap::default(),
            textures_to_free: Vec::default(),
            images: HashMap::default(),
        })
    }

    fn create_pipeline(
        device: Arc<Device>,
        subpass: Subpass,
    ) -> Result<Arc<GraphicsPipeline>, GraphicsPipelineCreationError> {
        GraphicsPipeline::start()
            .vertex_input_state(AdapterVertex::per_vertex())
            .vertex_shader(
                Self::load_vertex_shader(Arc::clone(&device))
                    .entry_point("main")
                    .unwrap(),
                (),
            )
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_dynamic(1))
            .fragment_shader(
                Self::load_fragment_shader(Arc::clone(&device))
                    .entry_point("main")
                    .unwrap(),
                (),
            )
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
            .color_blend_state(
                ColorBlendState::new(subpass.num_color_attachments()).blend({
                    let mut blend = AttachmentBlend::alpha();
                    blend.color_source = BlendFactor::One;
                    blend
                }),
            )
            .render_pass(subpass)
            .build(device)
    }

    fn create_texture_sampler(device: Arc<Device>) -> Result<Arc<Sampler>, SamplerCreationError> {
        Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Linear,
                ..Default::default()
            },
        )
    }

    fn load_vertex_shader(device: Arc<Device>) -> Arc<ShaderModule> {
        mod shader {
            vulkano_shaders::shader!(
                ty: "vertex",
                path: "src/engine/system/vulkan/egui/egui.vert"
            );
        }
        shader::load(device).unwrap()
    }

    fn load_fragment_shader(device: Arc<Device>) -> Arc<ShaderModule> {
        mod shader {
            vulkano_shaders::shader!(
                ty: "fragment",
                path: "src/engine/system/vulkan/egui/egui.frag"
            );
        }
        shader::load(device).unwrap()
    }

    pub fn update_textures<P>(
        &mut self,
        textures_delta: TexturesDelta,
        builder: &mut AutoCommandBufferBuilder<P>,
    ) -> Result<(), UploadError>
    where
        P: CommandBufferAllocator,
    {
        self.textures_to_free.extend(textures_delta.free);

        for (texture_id, delta) in textures_delta.set {
            let image = if delta.is_whole() {
                let image = self.create_image(builder.device(), queue, &delta.image)?;
                let layout = &self.pipeline.layout().set_layouts()[0];

                let desc = PersistentDescriptorSet::new(
                    builder.device(),
                    Arc::clone(&layout),
                    [WriteDescriptorSet::image_view_sampler(
                        0,
                        ImageView::new_default(Arc::clone(&image))?,
                        Arc::clone(&self.texture_sampler),
                    )],
                )?;

                self.textures.insert(texture_id, desc);
                self.images.insert(texture_id, Arc::clone(&image));
                image
            } else {
                Arc::clone(&self.images[texture_id])
            };

            self.upload_image_or_delta(image, delta, builder)?;
        }

        Ok(())
    }

    fn create_image(
        &self,
        allocator: &(impl MemoryAllocator + ?Sized),
        queue: Arc<Queue>,
        image: &ImageData,
    ) -> Result<Arc<StorageImage>, ImageError> {
        let dimensions = ImageDimensions::Dim2d {
            width: image.width() as u32,
            height: image.height() as u32,
            array_layers: 1,
        };

        StorageImage::with_usage(
            allocator,
            dimensions,
            Format::R8G8B8A8_SRGB,
            ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ImageCreateFlags::empty(),
            [queue.queue_family_index()],
        )
    }

    fn upload_image_or_delta<P>(
        &mut self,
        image: Arc<StorageImage>,
        delta: ImageDelta,
        builder: &mut AutoCommandBufferBuilder<P>,
    ) -> Result<(), UploadError>
    where
        P: CommandBufferAllocator,
    {
        builder.copy_buffer_to_image({
            let copy_info = CopyBufferToImageInfo::buffer_image(
                Buffer::from_data(
                    builder.device(),
                    BufferAllocateInfo {
                        buffer_usage: BufferUsage::TRANSFER_SRC,
                        memory_usage: MemoryUsage::Upload,
                        ..BufferAllocateInfo::default()
                    },
                    match delta.image {
                        ImageData::Color(color_data) => color_data
                            .pixels
                            .iter()
                            .flat_map(Color32::to_array)
                            .collect::<Vec<_>>(),
                        ImageData::Font(font_data) => font_data
                            .srgba_pixels(Some(1.0_f32)) // TODO
                            .flat_map(Color32::to_array)
                            .collect::<Vec<_>>(),
                    },
                )?,
                image,
            );

            if !delta.is_whole() {
                copy_info.regions[0].image_offset[0] = delta.pos.unwrap_or_default()[0] as u32;
                copy_info.regions[0].image_offset[1] = delta.pos.unwrap_or_default()[1] as u32;
                copy_info.regions[0].image_extent[0] = delta.image.width() as u32;
                copy_info.regions[0].image_extent[1] = delta.image.height() as u32;
            }

            copy_info
        })?;

        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod, Vertex)]
struct AdapterVertex {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
    #[format(R32G32B32A32_SFLOAT)]
    color: [f32; 4],
}

impl From<&egui::epaint::Vertex> for AdapterVertex {
    #[inline]
    fn from(value: &egui::epaint::Vertex) -> Self {
        Self {
            pos: [value.pos.x, value.pos.y],
            uv: [value.uv.x, value.uv.y],
            color: value.color.to_array().map(|v| f32::from(v) / 255.0),
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum PainterCreationError {
    #[error("Failed to create graphics pipeline: {0}")]
    GraphicsPipelineCreationError(#[from] GraphicsPipelineCreationError),
    #[error("Failed to create texture sampler: {0}")]
    SamplerCreationError(#[from] SamplerCreationError),
}

#[derive(thiserror::Error, Debug)]
pub enum UploadError {
    #[error("Failed to upload to image: {0}")]
    ImageError(#[from] ImageError),
}
