use crate::engine::system::vulkan::utils::pipeline::subpass_from_renderpass;
use crate::engine::system::vulkan::{DrawError, PipelineCreateError, ShaderLoadError, UploadError};
use crate::shader_from_path;
use bytemuck::{Pod, Zeroable};
use egui::epaint::ahash::HashMap;
use egui::{ClippedPrimitive, Color32, ImageData, Rect, TextureId, TexturesDelta};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferAllocateError, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferToImageInfo};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerCreateInfo, SamplerMipmapMode};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageAllocateError, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::cache::PipelineCache;
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, BlendFactor, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Scissor, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::RenderPass;
use vulkano::shader::EntryPoint;
use vulkano::{Validated, VulkanError};

use crate::ui::egui::epaint::{ImageDelta, Primitive};

pub struct EguiOnVulkanoPainter {
    pub queue: Arc<Queue>,
    pub pipeline: Arc<GraphicsPipeline>,
    pub texture_sampler: Arc<Sampler>,
    pub textures: HashMap<TextureId, Arc<PersistentDescriptorSet>>,
    pub textures_to_free: Vec<TextureId>,
    pub images: HashMap<TextureId, Arc<Image>>,
    pub desc_allocator: StandardDescriptorSetAllocator,
    pub memo_allocator: StandardMemoryAllocator,
}

impl EguiOnVulkanoPainter {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        render_pass: Arc<RenderPass>,
        cache: Option<Arc<PipelineCache>>,
    ) -> Result<Self, PipelineCreateError> {
        Ok(Self {
            queue,
            desc_allocator: StandardDescriptorSetAllocator::new(Arc::clone(&device)),
            memo_allocator: StandardMemoryAllocator::new_default(Arc::clone(&device)),
            pipeline: Self::create_pipeline(Arc::clone(&device), render_pass, cache)?,
            texture_sampler: Self::create_texture_sampler(device)?,
            textures: HashMap::default(),
            textures_to_free: Vec::default(),
            images: HashMap::default(),
        })
    }

    fn create_pipeline(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        cache: Option<Arc<PipelineCache>>,
    ) -> Result<Arc<GraphicsPipeline>, PipelineCreateError> {
        let vs = Self::load_vertex_shader(Arc::clone(&device))?;
        let fs = Self::load_fragment_shader(Arc::clone(&device))?;

        let vertex_input_state =
            AdapterVertex::per_vertex().definition(&vs.info().input_interface)?;

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            Arc::clone(&device),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(Arc::clone(&device))?,
        )?;

        Ok(GraphicsPipeline::new(
            Arc::clone(&device),
            cache,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(
                    InputAssemblyState::default().topology(PrimitiveTopology::TriangleList),
                ),
                viewport_state: Some(ViewportState::viewport_dynamic_scissor_dynamic(1)),
                rasterization_state: Some(RasterizationState::new().cull_mode(CullMode::None)),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::new(1).blend({
                    let mut blend = AttachmentBlend::alpha();
                    blend.src_color_blend_factor = BlendFactor::One;
                    blend
                })),
                subpass: Some(subpass_from_renderpass(render_pass)?),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?)
    }

    fn create_texture_sampler(device: Arc<Device>) -> Result<Arc<Sampler>, Validated<VulkanError>> {
        Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Linear,
                ..SamplerCreateInfo::default()
            },
        )
    }

    fn load_vertex_shader(device: Arc<Device>) -> Result<EntryPoint, ShaderLoadError> {
        shader_from_path!(device, "vertex", "src/engine/system/vulkan/egui/egui.vert")
    }

    fn load_fragment_shader(device: Arc<Device>) -> Result<EntryPoint, ShaderLoadError> {
        shader_from_path!(
            device,
            "fragment",
            "src/engine/system/vulkan/egui/egui.frag"
        )
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
            .bind_pipeline_graphics(Arc::clone(&self.pipeline))?;

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
                            offset: [rect.min.x as u32, rect.min.y as u32],
                            extent: [rect.width() as u32, rect.height() as u32],
                        }]
                        .into_iter()
                        .collect(),
                    )?
                    .bind_vertex_buffers(0, vertices)?
                    .bind_index_buffer(indices)?
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        Arc::clone(&self.pipeline.layout()),
                        0,
                        Arc::clone(texture),
                    )?
                    .push_constants(Arc::clone(&self.pipeline.layout()), 0, [width, height])?
                    .draw_indexed(index_count, 1, 0, 0, 0)?;
            }
        }

        self.free_textures();
        Ok(())
    }

    fn free_textures(&mut self) {
        for texture in self.textures_to_free.drain(..) {
            self.textures.remove(&texture);
            self.images.remove(&texture);
        }
    }

    fn create_buffers<V, I>(
        &self,
        vertices: V,
        indices: I,
    ) -> Result<(Subbuffer<[AdapterVertex]>, Subbuffer<[u32]>), Validated<BufferAllocateError>>
    where
        V: IntoIterator<Item = AdapterVertex>,
        V::IntoIter: ExactSizeIterator,
        I: IntoIterator<Item = u32>,
        I::IntoIter: ExactSizeIterator,
    {
        let allocation_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..AllocationCreateInfo::default()
        };

        Ok((
            Buffer::from_iter(
                &self.memo_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..BufferCreateInfo::default()
                },
                allocation_info.clone(),
                vertices,
            )?,
            Buffer::from_iter(
                &self.memo_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER,
                    ..BufferCreateInfo::default()
                },
                allocation_info,
                indices,
            )?,
        ))
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
                let image = self.create_image(&delta.image)?;
                let layout = &self.pipeline.layout().set_layouts()[0];

                let desc = PersistentDescriptorSet::new(
                    &self.desc_allocator,
                    Arc::clone(&layout),
                    [WriteDescriptorSet::image_view_sampler(
                        0,
                        ImageView::new_default(Arc::clone(&image))?,
                        Arc::clone(&self.texture_sampler),
                    )],
                    [],
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

    fn create_image(&self, image: &ImageData) -> Result<Arc<Image>, Validated<ImageAllocateError>> {
        Image::new(
            &self.memo_allocator,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                extent: [image.width() as u32, image.height() as u32, 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..ImageCreateInfo::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..AllocationCreateInfo::default()
            },
        )
    }

    fn upload_image_or_delta<P>(
        &mut self,
        image: Arc<Image>,
        delta: &ImageDelta,
        builder: &mut AutoCommandBufferBuilder<P>,
    ) -> Result<(), Validated<BufferAllocateError>> {
        builder.copy_buffer_to_image({
            let mut copy_info = CopyBufferToImageInfo::buffer_image(
                Buffer::from_iter(
                    &self.memo_allocator,
                    BufferCreateInfo {
                        usage: BufferUsage::TRANSFER_SRC,
                        ..BufferCreateInfo::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..AllocationCreateInfo::default()
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
