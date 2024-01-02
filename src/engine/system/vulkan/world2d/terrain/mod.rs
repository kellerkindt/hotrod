use crate::engine::system::vulkan::system::{VulkanSystem, WriteDescriptorSetCollection};
use crate::engine::system::vulkan::textures::{ImageSamplerMode, TextureId, TextureManager};
use crate::engine::system::vulkan::utils::pipeline::{
    subpass_from_renderpass, write_descriptor_sets_from_collection,
};
use crate::engine::system::vulkan::{DrawError, PipelineCreateError, ShaderLoadError};
use crate::shader_from_path;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use vulkano::buffer::{
    Buffer, BufferAllocateError, BufferCreateInfo, BufferUsage, IndexBuffer, Subbuffer,
};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::device::Device;
use vulkano::image::Image;
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::cache::PipelineCache;
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::RenderPass;
use vulkano::shader::EntryPoint;
use vulkano::{Validated, VulkanError};

/// This pipeline is used to draw the terrain of 2d worlds. A 2d world terrain consists of quadratic
/// tiles. It supports additional features besides painting the terrain like:
///  - shading a terrain tile
#[derive()]
pub struct World2dTerrainPipeline {
    pipeline: Arc<GraphicsPipeline>,
    memo_allocator: StandardMemoryAllocator,
    quad_index_buffer: IndexBuffer,
    quad_vertex_buffer: Subbuffer<[Vertex2d]>,
    write_descriptors: WriteDescriptorSetCollection,
    texture_manager: TextureManager<Self, 0>,
}

impl TryFrom<&VulkanSystem> for World2dTerrainPipeline {
    type Error = PipelineCreateError;

    #[inline]
    fn try_from(vs: &VulkanSystem) -> Result<Self, Self::Error> {
        Self::new(
            Arc::clone(vs.device()),
            Arc::clone(vs.render_pass()),
            vs.pipeline_cache().map(Arc::clone),
            vs.get_write_descriptor_sets().clone(),
        )
    }
}

impl World2dTerrainPipeline {
    pub fn new(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        cache: Option<Arc<PipelineCache>>,
        write_descriptors: WriteDescriptorSetCollection,
    ) -> Result<Self, PipelineCreateError> {
        let pipeline = Self::create_pipeline(Arc::clone(&device), render_pass, cache)?;
        let memo_allocator = StandardMemoryAllocator::new_default(Arc::clone(&device));
        Ok(Self {
            quad_index_buffer: Self::create_index_buffer(&memo_allocator, [0, 1, 2, 2, 3, 0])?
                .into(),
            quad_vertex_buffer: Self::create_vertex_buffer(
                &memo_allocator,
                vec![
                    Vertex2d { pos: [-0.5, -0.5] },
                    Vertex2d { pos: [0.5, -0.5] },
                    Vertex2d { pos: [0.5, 0.5] },
                    Vertex2d { pos: [-0.5, 0.5] },
                ],
            )?
            .into(),
            write_descriptors,
            memo_allocator,
            texture_manager: TextureManager::basic(
                device,
                &pipeline,
                ImageSamplerMode::PixelPerfect,
            )?,
            pipeline,
        })
    }

    fn create_pipeline(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        cache: Option<Arc<PipelineCache>>,
    ) -> Result<Arc<GraphicsPipeline>, PipelineCreateError> {
        let vs = Self::load_vertex_shader(Arc::clone(&device))?;
        let fs = Self::load_fragment_shader(Arc::clone(&device))?;

        let vertex_input_state = [Vertex2d::per_vertex(), InstanceData::per_instance()]
            .definition(&vs.info().input_interface)?;

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
                viewport_state: Some(ViewportState::viewport_dynamic_scissor_irrelevant()),
                rasterization_state: Some(RasterizationState::new().cull_mode(CullMode::None)),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::new(1).blend(AttachmentBlend::alpha())),
                subpass: Some(subpass_from_renderpass(render_pass)?),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?)
    }

    fn load_vertex_shader(device: Arc<Device>) -> Result<EntryPoint, ShaderLoadError> {
        shader_from_path!(
            device,
            "vertex",
            "src/engine/system/vulkan/world2d/terrain/terrain.vert"
        )
    }

    fn load_fragment_shader(device: Arc<Device>) -> Result<EntryPoint, ShaderLoadError> {
        shader_from_path!(
            device,
            "fragment",
            "src/engine/system/vulkan/world2d/terrain/terrain.frag"
        )
    }

    pub fn draw<P, I>(
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        texture: &TextureId<Self>,
        tiles: I,
    ) -> Result<(), DrawError>
    where
        I: IntoIterator<Item = InstanceData>,
        I::IntoIter: ExactSizeIterator,
    {
        if self.texture_manager.is_origin_of(texture) {
            let vertex_buffer = Self::create_vertex_buffer(&self.memo_allocator, tiles)?;
            let instance_count = vertex_buffer.len() as u32;

            builder
                .bind_pipeline_graphics(Arc::clone(&self.pipeline))?
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    Arc::clone(&self.pipeline.layout()),
                    0,
                    Arc::clone(&texture.0.descriptor),
                )?
                .bind_index_buffer(self.quad_index_buffer.clone())?
                .bind_vertex_buffers(
                    0,
                    [
                        self.quad_vertex_buffer.as_bytes().clone(),
                        vertex_buffer.into_bytes(),
                    ],
                )?
                .draw_indexed(6, instance_count, 0, 0, 0)?;

            Ok(())
        } else {
            todo!()
        }
    }

    fn create_vertex_buffer<I, T: Send + Sync + Pod>(
        memo_allocator: &impl MemoryAllocator,
        vertices: I,
    ) -> Result<Subbuffer<[T]>, Validated<BufferAllocateError>>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        Buffer::from_iter(
            memo_allocator,
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

    fn create_index_buffer<I>(
        memo_allocator: &impl MemoryAllocator,
        indices: I,
    ) -> Result<Subbuffer<[u32]>, Validated<BufferAllocateError>>
    where
        I: IntoIterator<Item = u32>,
        I::IntoIter: ExactSizeIterator,
    {
        Buffer::from_iter(
            memo_allocator,
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

    pub fn prepare_texture(
        &self,
        image: Arc<Image>,
    ) -> Result<TextureId<Self>, Validated<VulkanError>> {
        self.texture_manager.prepare_texture(
            image,
            write_descriptor_sets_from_collection(
                &self.pipeline.layout().set_layouts()[0],
                &self.write_descriptors,
            ),
        )
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod, Vertex)]
pub struct Vertex2d {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod, Vertex)]
pub struct InstanceData {
    #[format(R32G32_SFLOAT)]
    pub tile_pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub uv0: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub uv1: [f32; 2],
    #[format(R32_SFLOAT)]
    pub shading: f32,
}
