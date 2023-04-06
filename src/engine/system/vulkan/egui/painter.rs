use bytemuck::{Pod, Zeroable};
use egui::epaint::ahash::HashMap;
use egui::{ClippedPrimitive, Color32, ImageData, Rect, TextureId, TexturesDelta};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferAllocateInfo, BufferError, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CopyBufferToImageInfo, CopyError, PipelineExecutionError,
    RenderPassError,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{
    DescriptorSetCreationError, PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::view::{ImageView, ImageViewCreationError};
use vulkano::image::{ImageCreateFlags, ImageDimensions, ImageError, ImageUsage, StorageImage};
use vulkano::memory::allocator::{MemoryUsage, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, BlendFactor, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::render_pass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Scissor, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreationError;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::sampler::{
    Filter, Sampler, SamplerCreateInfo, SamplerCreationError, SamplerMipmapMode,
};
use vulkano::shader::ShaderModule;

use crate::ui::egui::epaint::{ImageDelta, Primitive};

pub struct EguiOnVulkanoPainter {
    pub queue: Arc<Queue>,
    pub pipeline: Arc<GraphicsPipeline>,
    pub texture_sampler: Arc<Sampler>,
    pub textures: HashMap<TextureId, Arc<PersistentDescriptorSet>>,
    pub textures_to_free: Vec<TextureId>,
    pub images: HashMap<TextureId, Arc<StorageImage>>,
    pub desc_allocator: StandardDescriptorSetAllocator,
    pub memo_allocator: StandardMemoryAllocator,
}

impl EguiOnVulkanoPainter {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        image_format: Format,
    ) -> Result<Self, PainterCreationError> {
        Ok(Self {
            queue,
            desc_allocator: StandardDescriptorSetAllocator::new(Arc::clone(&device)),
            memo_allocator: StandardMemoryAllocator::new_default(Arc::clone(&device)),
            pipeline: Self::create_pipeline(Arc::clone(&device), image_format)?,
            texture_sampler: Self::create_texture_sampler(device)?,
            textures: HashMap::default(),
            textures_to_free: Vec::default(),
            images: HashMap::default(),
        })
    }

    fn create_pipeline(
        device: Arc<Device>,
        image_format: Format,
    ) -> Result<Arc<GraphicsPipeline>, GraphicsPipelineCreationError> {
        GraphicsPipeline::start()
            .vertex_input_state(AdapterVertex::per_vertex())
            .input_assembly_state(InputAssemblyState::new())
            .vertex_shader(
                Self::load_vertex_shader(Arc::clone(&device))
                    .entry_point("main")
                    .unwrap(),
                (),
            )
            .viewport_state(ViewportState::viewport_dynamic_scissor_dynamic(1))
            .fragment_shader(
                Self::load_fragment_shader(Arc::clone(&device))
                    .entry_point("main")
                    .unwrap(),
                (),
            )
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
            .color_blend_state(ColorBlendState::new(1).blend({
                let mut blend = AttachmentBlend::alpha();
                blend.color_source = BlendFactor::One;
                blend
            }))
            .render_pass(PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(image_format)],
                ..Default::default()
            })
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

    pub fn draw<P>(
        &mut self,

        builder: &mut AutoCommandBufferBuilder<P>,
        width: f32,
        height: f32,
        clipped_primitives: &[ClippedPrimitive],
    ) -> Result<(), DrawError> {
        builder
            //.next_subpass(SubpassContents::Inline)?
            .bind_pipeline_graphics(Arc::clone(&self.pipeline));

        let mut vertices = Vec::<AdapterVertex>::with_capacity(clipped_primitives.len() * 4);
        let mut indices = Vec::<u32>::with_capacity(clipped_primitives.len() * 6);
        let mut clip_rects = Vec::<Rect>::with_capacity(clipped_primitives.len());
        let mut texture_ids = Vec::<TextureId>::with_capacity(clipped_primitives.len());
        let mut offsets = Vec::<(usize, usize)>::with_capacity(clipped_primitives.len());

        for clipped in clipped_primitives {
            let mesh = match &clipped.primitive {
                Primitive::Mesh(mesh) => mesh,
                Primitive::Callback(_) => {
                    dbg!("NOT YET SUPPORTED", &clipped.primitive);
                    continue;
                }
            };

            if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                continue;
            }

            offsets.push((vertices.len(), indices.len()));
            texture_ids.push(mesh.texture_id);

            mesh.vertices.iter().for_each(|v| vertices.push(v.into()));
            mesh.indices.iter().for_each(|i| indices.push(*i));
            clip_rects.push(clipped.clip_rect);
        }

        if clip_rects.is_empty() {
            // nothing to do
            return Ok(());
        }

        offsets.push((vertices.len(), indices.len()));

        let (vertex_buffer, index_buffer) = self.create_buffers(vertices, indices)?;
        for (index, rect) in clip_rects.into_iter().enumerate() {
            let offset = offsets[index];
            let offset_end = offsets[index + 1];

            let vertices = vertex_buffer
                .clone()
                .slice(offset.0 as u64..offset_end.0 as u64);
            let indices = index_buffer
                .clone()
                .slice(offset.1 as u64..offset_end.1 as u64);

            if let Some(texture) = self.textures.get(&texture_ids[index]) {
                let index_count = indices.len() as u32;
                builder
                    .set_scissor(
                        0,
                        [Scissor {
                            origin: [rect.min.x as u32, rect.min.y as u32],
                            dimensions: [rect.width() as u32, rect.height() as u32],
                        }],
                    )
                    .bind_vertex_buffers(0, vertices)
                    .bind_index_buffer(indices)
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        Arc::clone(&self.pipeline.layout()),
                        0,
                        Arc::clone(texture),
                    )
                    .push_constants(Arc::clone(&self.pipeline.layout()), 0, [width, height])
                    .draw_indexed(index_count, 1, 0, 0, 0)?;
            }
        }

        // self.free_textures(); TODO
        Ok(())
    }

    fn create_buffers(
        &self,
        vertices: Vec<AdapterVertex>,
        indices: Vec<u32>,
    ) -> Result<(Subbuffer<[AdapterVertex]>, Subbuffer<[u32]>), BufferError> {
        let vertices = Buffer::from_iter(
            &self.memo_allocator,
            BufferAllocateInfo {
                buffer_usage: BufferUsage::VERTEX_BUFFER,
                ..BufferAllocateInfo::default()
            },
            vertices,
        )?;
        let indices = Buffer::from_iter(
            &self.memo_allocator,
            BufferAllocateInfo {
                buffer_usage: BufferUsage::INDEX_BUFFER,
                ..BufferAllocateInfo::default()
            },
            indices,
        )?;

        Ok((vertices, indices))
    }

    pub fn update_textures<P>(
        &mut self,
        textures_delta: &TexturesDelta,
        builder: &mut AutoCommandBufferBuilder<P>,
    ) -> Result<(), UploadError> {
        self.textures_to_free
            .extend(textures_delta.free.iter().copied());

        for (texture_id, delta) in &textures_delta.set {
            let image = if delta.is_whole() {
                let image = self.create_image(&*self.queue, &delta.image)?;
                let layout = &self.pipeline.layout().set_layouts()[0];

                let desc = PersistentDescriptorSet::new(
                    &self.desc_allocator,
                    Arc::clone(&layout),
                    [WriteDescriptorSet::image_view_sampler(
                        0,
                        ImageView::new_default(Arc::clone(&image))?,
                        Arc::clone(&self.texture_sampler),
                    )],
                )?;

                self.textures.insert(*texture_id, desc);
                self.images.insert(*texture_id, Arc::clone(&image));
                image
            } else {
                Arc::clone(&self.images[&texture_id])
            };

            self.upload_image_or_delta(image, delta, builder)?;
        }

        Ok(())
    }

    fn create_image(
        &self,
        queue: &Queue,
        image: &ImageData,
    ) -> Result<Arc<StorageImage>, ImageError> {
        let dimensions = ImageDimensions::Dim2d {
            width: image.width() as u32,
            height: image.height() as u32,
            array_layers: 1,
        };

        StorageImage::with_usage(
            &self.memo_allocator,
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
        delta: &ImageDelta,
        builder: &mut AutoCommandBufferBuilder<P>,
    ) -> Result<(), UploadError> {
        builder.copy_buffer_to_image({
            let mut copy_info = CopyBufferToImageInfo::buffer_image(
                Buffer::from_iter(
                    &self.memo_allocator,
                    BufferAllocateInfo {
                        buffer_usage: BufferUsage::TRANSFER_SRC,
                        memory_usage: MemoryUsage::Upload,
                        ..BufferAllocateInfo::default()
                    },
                    match &delta.image {
                        ImageData::Color(color_data) => color_data
                            .pixels
                            .iter()
                            .flat_map(Color32::to_array)
                            .collect::<Vec<_>>(),
                        ImageData::Font(font_data) => font_data
                            .srgba_pixels(None) // TODO
                            .flat_map(|c| c.to_array())
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
    #[error("Failed to upload the image: {0}")]
    ImageError(#[from] ImageError),
    #[error("Failed to create the image view: {0}")]
    ImageViewCreationError(#[from] ImageViewCreationError),
    #[error("Failed to create the descriptor set: {0}")]
    DescriptorSetCreationError(#[from] DescriptorSetCreationError),
    #[error("Failed to create the buffer: {0}")]
    BufferError(#[from] BufferError),
    #[error("Failed to copy to the buffer: {0}")]
    CopyError(#[from] CopyError),
}

#[derive(thiserror::Error, Debug)]
pub enum DrawError {
    #[error("Unable to configure rendering: {0}")]
    RenderPassError(#[from] RenderPassError),
    #[error("Unable to upload buffer contents: {0}")]
    BufferError(#[from] BufferError),
    #[error("Unable to execute the pipeline: {0}")]
    PipelineExecutionError(#[from] PipelineExecutionError),
}
