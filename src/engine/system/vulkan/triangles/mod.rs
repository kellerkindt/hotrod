use crate::engine::system::vulkan::system::VulkanSystem;
use crate::engine::system::vulkan::utils::pipeline::subpass_from_renderpass;
use crate::engine::system::vulkan::{DrawError, PipelineCreateError, ShaderLoadError};
use crate::shader_from_path;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferAllocateError, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::device::{Device, Features};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
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
    GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::RenderPass;
use vulkano::shader::EntryPoint;
use vulkano::Validated;

#[derive()]
pub struct TrianglesPipeline {
    pipeline: Arc<GraphicsPipeline>,
    memo_allocator: StandardMemoryAllocator,
}

impl TryFrom<&VulkanSystem> for TrianglesPipeline {
    type Error = PipelineCreateError;

    fn try_from(vs: &VulkanSystem) -> Result<Self, Self::Error> {
        Self::new(
            Arc::clone(vs.device()),
            Arc::clone(vs.render_pass()),
            vs.pipeline_cache().map(Arc::clone),
        )
    }
}

impl TrianglesPipeline {
    pub const REQUIRED_FEATURES: Features = Features {
        dynamic_rendering: true,
        ..Features::empty()
    };

    pub fn new(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        cache: Option<Arc<PipelineCache>>,
    ) -> Result<Self, PipelineCreateError> {
        Ok(Self {
            pipeline: Self::create_pipeline(Arc::clone(&device), render_pass, cache)?,
            memo_allocator: StandardMemoryAllocator::new_default(Arc::clone(&device)),
        })
    }

    fn create_pipeline(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        cache: Option<Arc<PipelineCache>>,
    ) -> Result<Arc<GraphicsPipeline>, PipelineCreateError> {
        let vs = Self::load_vertex_shader(Arc::clone(&device))?;
        let fs = Self::load_fragment_shader(Arc::clone(&device))?;

        let vertex_input_state = Vertex2d::per_vertex().definition(&vs.info().input_interface)?;

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
            "src/engine/system/vulkan/triangles/triangles.vert"
        )
    }

    fn load_fragment_shader(device: Arc<Device>) -> Result<EntryPoint, ShaderLoadError> {
        shader_from_path!(
            device,
            "fragment",
            "src/engine/system/vulkan/triangles/triangles.frag"
        )
    }

    pub fn draw<P>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<P>,
        width: f32,
        height: f32,
        triangles: &[Triangles],
    ) -> Result<(), DrawError> {
        let mut offset = 0;

        let vertex_buffer = self.create_vertex_buffer(
            triangles
                .iter()
                .flat_map(|l| l.vertices.iter().copied())
                .collect::<Vec<_>>(),
        )?;

        builder
            .bind_pipeline_graphics(Arc::clone(&self.pipeline))?
            .bind_vertex_buffers(0, vertex_buffer)?;

        for triangles in triangles {
            builder
                .push_constants(
                    Arc::clone(&self.pipeline.layout()),
                    0,
                    [
                        triangles.color[0],
                        triangles.color[1],
                        triangles.color[2],
                        triangles.color[3],
                        width,
                        height,
                    ],
                )?
                .draw(triangles.vertices.len() as u32, 1, offset, 0)?;
            offset += triangles.vertices.len() as u32;
        }

        Ok(())
    }

    pub fn draw_indexed<P>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<P>,
        width: f32,
        height: f32,
        triangles: &[TrianglesIndexed],
    ) -> Result<(), DrawError> {
        let mut offset_vertices = 0;
        let mut offset_indices = 0;

        let vertex_buffer = self.create_vertex_buffer(
            triangles
                .iter()
                .flat_map(|l| l.vertices.iter().copied())
                .collect::<Vec<_>>(),
        )?;

        let index_buffer = self.create_index_buffer(
            triangles
                .iter()
                .flat_map(|l| l.indices.iter().flat_map(|i| i.into_iter()).copied())
                .collect::<Vec<_>>(),
        )?;

        builder
            .bind_pipeline_graphics(Arc::clone(&self.pipeline))?
            .bind_index_buffer(index_buffer)?
            .bind_vertex_buffers(0, vertex_buffer)?;

        for triangles in triangles {
            let index_count = triangles.indices.len() as u32 * 3;

            builder
                .push_constants(
                    Arc::clone(&self.pipeline.layout()),
                    0,
                    [
                        triangles.color[0],
                        triangles.color[1],
                        triangles.color[2],
                        triangles.color[3],
                        width,
                        height,
                    ],
                )?
                .draw_indexed(index_count, 1, offset_indices, offset_vertices, 0)?;

            offset_vertices += triangles.vertices.len() as i32;
            offset_indices += index_count as u32;
        }

        Ok(())
    }

    fn create_vertex_buffer<I>(
        &self,
        vertices: I,
    ) -> Result<Subbuffer<[Vertex2d]>, Validated<BufferAllocateError>>
    where
        I: IntoIterator<Item = Vertex2d>,
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

    fn create_index_buffer<I>(
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
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod, Vertex)]
pub struct Vertex2d {
    #[format(R32G32_SFLOAT)]
    pub pos: [f32; 2],
}

pub struct Triangles {
    pub vertices: Vec<Vertex2d>,
    pub color: [f32; 4],
}

pub struct TrianglesIndexed {
    pub vertices: Vec<Vertex2d>,
    pub indices: Vec<[u32; 3]>,
    pub color: [f32; 4],
}
