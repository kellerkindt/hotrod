use crate::engine::system::vulkan::system::VulkanSystem;
use crate::engine::system::vulkan::textures::{ImageSamplerMode, TextureId, TextureManager};
use crate::engine::system::vulkan::utils::pipeline::subpass_from_renderpass;
use crate::engine::system::vulkan::wds::WriteDescriptorSetManager;
use crate::engine::system::vulkan::{DrawError, PipelineCreateError, ShaderLoadError};
use crate::shader_from_path;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferAllocateError, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::device::{Device, Features};
use vulkano::image::Image;
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
    GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::RenderPass;
use vulkano::shader::EntryPoint;
use vulkano::{Validated, VulkanError};

#[derive()]
pub struct TexturedPipeline {
    pipeline: Arc<GraphicsPipeline>,
    memo_allocator: StandardMemoryAllocator,
    write_descriptors: Arc<WriteDescriptorSetManager>,
    texture_manager: TextureManager<Self, 0>,
}

impl TryFrom<&VulkanSystem> for TexturedPipeline {
    type Error = PipelineCreateError;

    fn try_from(vs: &VulkanSystem) -> Result<Self, Self::Error> {
        Self::new(
            Arc::clone(vs.device()),
            Arc::clone(vs.render_pass()),
            vs.pipeline_cache().map(Arc::clone),
            vs.write_descriptor_set_manager().clone(),
        )
    }
}

impl TexturedPipeline {
    pub const REQUIRED_FEATURES: Features = Features {
        dynamic_rendering: true,
        ..Features::empty()
    };

    pub fn new(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        cache: Option<Arc<PipelineCache>>,
        write_descriptors: Arc<WriteDescriptorSetManager>,
    ) -> Result<Self, PipelineCreateError> {
        let pipeline = Self::create_pipeline(Arc::clone(&device), render_pass, cache)?;
        Ok(Self {
            memo_allocator: StandardMemoryAllocator::new_default(Arc::clone(&device)),
            write_descriptors,
            texture_manager: TextureManager::basic(device, &pipeline, ImageSamplerMode::Linear)?,
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

        let vertex_input_state = Vertex2dUv::per_vertex().definition(&vs.info().input_interface)?;

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
            "src/engine/system/vulkan/textured/textured.vert"
        )
    }

    fn load_fragment_shader(device: Arc<Device>) -> Result<EntryPoint, ShaderLoadError> {
        shader_from_path!(
            device,
            "fragment",
            "src/engine/system/vulkan/textured/textured.frag"
        )
    }

    pub fn draw<P>(
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        textured: &[Textured],
    ) -> Result<(), DrawError> {
        let mut offset = 0;
        let vertex_buffer = self.create_vertex_buffer(
            textured
                .iter()
                .flat_map(|l| l.vertices.iter().copied())
                .collect::<Vec<_>>(),
        )?;

        builder
            .bind_pipeline_graphics(Arc::clone(&self.pipeline))?
            .bind_vertex_buffers(0, vertex_buffer)?;

        for textured in textured {
            if self.texture_manager.is_origin_of(&textured.texture) {
                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        Arc::clone(&self.pipeline.layout()),
                        0,
                        Arc::clone(&textured.texture.0.descriptor),
                    )?
                    .draw(textured.vertices.len() as u32, 1, offset, 0)?;
            }

            offset += textured.vertices.len() as u32;
        }

        Ok(())
    }

    pub fn draw_indexed<P>(
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        textured: &[TexturedIndexed],
    ) -> Result<(), DrawError> {
        let mut offset_vertices = 0;
        let mut offset_indices = 0;

        let vertex_buffer = self.create_vertex_buffer(
            textured
                .iter()
                .flat_map(|l| l.vertices.iter().copied())
                .collect::<Vec<_>>(),
        )?;

        let index_buffer = self.create_index_buffer(
            textured
                .iter()
                .flat_map(|l| l.indices.iter().flat_map(|i| i.into_iter()).copied())
                .collect::<Vec<_>>(),
        )?;

        builder
            .bind_pipeline_graphics(Arc::clone(&self.pipeline))?
            .bind_index_buffer(index_buffer)?
            .bind_vertex_buffers(0, vertex_buffer)?;

        for textured in textured {
            let index_count = textured.indices.len() as u32 * 3;

            if self.texture_manager.is_origin_of(&textured.texture) {
                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        Arc::clone(&self.pipeline.layout()),
                        0,
                        Arc::clone(&textured.texture.0.descriptor),
                    )?
                    .draw_indexed(index_count, 1, offset_indices, offset_vertices, 0)?;
            }

            offset_vertices += textured.vertices.len() as i32;
            offset_indices += index_count;
        }

        Ok(())
    }

    fn create_vertex_buffer<I>(
        &self,
        vertices: I,
    ) -> Result<Subbuffer<[Vertex2dUv]>, Validated<BufferAllocateError>>
    where
        I: IntoIterator<Item = Vertex2dUv>,
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

    pub fn prepare_texture(
        &self,
        image: Arc<Image>,
    ) -> Result<TextureId<Self>, Validated<VulkanError>> {
        self.texture_manager.prepare_texture(
            image,
            self.write_descriptors
                .get_required_descriptors(&self.pipeline.layout().set_layouts()[0]),
        )
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod, Vertex)]
pub struct Vertex2dUv {
    #[format(R32G32_SFLOAT)]
    pub pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
}

pub struct Textured {
    pub vertices: Vec<Vertex2dUv>,
    pub texture: TextureId<TexturedPipeline>,
}

pub struct TexturedIndexed {
    pub vertices: Vec<Vertex2dUv>,
    pub indices: Vec<[u32; 3]>,
    pub texture: TextureId<TexturedPipeline>,
}
