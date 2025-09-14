use crate::engine::system::vulkan::buffers::BasicBuffersManager;
use crate::engine::system::vulkan::system::{GraphicsPipelineRenderPassInfo, VulkanSystem};
use crate::engine::system::vulkan::utils::Draw;
use crate::engine::system::vulkan::wds::WriteDescriptorSetManager;
use crate::engine::system::vulkan::{DrawError, PipelineCreateError, ShaderLoadError};
use crate::shader_from_path;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use vulkano::buffer::{IndexBuffer, Subbuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::Device;
use vulkano::pipeline::cache::PipelineCache;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::shader::EntryPoint;

pub struct GlowingBallsPipeline {
    pipeline: Arc<GraphicsPipeline>,
    buffers_manager: Arc<BasicBuffersManager>,
    quad_index_buffer: IndexBuffer,
    quad_vertex_buffer: Subbuffer<[Vertex2d]>,
    descriptor_set: Arc<DescriptorSet>,
}

impl TryFrom<&VulkanSystem> for GlowingBallsPipeline {
    type Error = PipelineCreateError;

    #[inline]
    fn try_from(vs: &VulkanSystem) -> Result<Self, Self::Error> {
        Self::new(
            Arc::clone(vs.device()),
            vs.graphics_pipeline_render_pass_info(),
            vs.pipeline_cache().map(Arc::clone),
            vs.write_descriptor_set_manager(),
            Arc::clone(vs.basic_buffers_manager()),
        )
    }
}

impl GlowingBallsPipeline {
    pub fn new(
        device: Arc<Device>,
        render_pass_info: GraphicsPipelineRenderPassInfo,
        cache: Option<Arc<PipelineCache>>,
        write_descriptors: &WriteDescriptorSetManager,
        buffers_manager: Arc<BasicBuffersManager>,
    ) -> Result<Self, PipelineCreateError> {
        let pipeline = Self::create_pipeline(Arc::clone(&device), render_pass_info, cache)?;
        Ok(Self {
            quad_index_buffer: buffers_manager
                .create_index_buffer([0, 1, 2, 2, 3, 0])?
                .into(),
            quad_vertex_buffer: buffers_manager
                .create_vertex_buffer(vec![
                    Vertex2d { pos: [-0.5, -0.5] },
                    Vertex2d { pos: [0.5, -0.5] },
                    Vertex2d { pos: [0.5, 0.5] },
                    Vertex2d { pos: [-0.5, 0.5] },
                ])?
                .into(),
            descriptor_set: write_descriptors
                .create_persistent_descriptor_set(&pipeline.layout().set_layouts()[0])?,
            pipeline,
            buffers_manager,
        })
    }

    fn create_pipeline(
        device: Arc<Device>,
        render_pass_info: GraphicsPipelineRenderPassInfo,
        cache: Option<Arc<PipelineCache>>,
    ) -> Result<Arc<GraphicsPipeline>, PipelineCreateError> {
        let vs = Self::load_vertex_shader(Arc::clone(&device))?;
        let fs = Self::load_fragment_shader(Arc::clone(&device))?;

        let vertex_input_state =
            [Vertex2d::per_vertex(), GlowingBall::per_instance()].definition(&vs)?;

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
                input_assembly_state: Some(InputAssemblyState {
                    topology: PrimitiveTopology::TriangleList,
                    ..InputAssemblyState::default()
                }),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState {
                    rasterization_samples: render_pass_info.rasterization_samples(),
                    ..MultisampleState::default()
                }),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    render_pass_info.num_color_attachments(),
                    ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend::alpha()),
                        ..ColorBlendAttachmentState::default()
                    },
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(render_pass_info.into_subpass_type()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?)
    }

    fn load_vertex_shader(device: Arc<Device>) -> Result<EntryPoint, ShaderLoadError> {
        shader_from_path!(
            device,
            "vertex",
            "src/engine/system/vulkan/glowing_balls/glowing_balls.vert"
        )
    }

    fn load_fragment_shader(device: Arc<Device>) -> Result<EntryPoint, ShaderLoadError> {
        shader_from_path!(
            device,
            "fragment",
            "src/engine/system/vulkan/glowing_balls/glowing_balls.frag"
        )
    }

    pub fn draw<P, I>(
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        balls: I,
    ) -> Result<(), DrawError>
    where
        I: IntoIterator<Item = GlowingBall>,
        I::IntoIter: ExactSizeIterator,
    {
        let vertex_buffer = self.buffers_manager.create_vertex_buffer(balls)?;
        let instance_count = vertex_buffer.len() as u32;

        builder
            .bind_pipeline_graphics(Arc::clone(&self.pipeline))?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                Arc::clone(&self.pipeline.layout()),
                0,
                Arc::clone(&self.descriptor_set),
            )?
            .bind_index_buffer(self.quad_index_buffer.clone())?
            .bind_vertex_buffers(
                0,
                [
                    self.quad_vertex_buffer.as_bytes().clone(),
                    vertex_buffer.into_bytes(),
                ],
            )?
            .hotrod_draw_indexed(6, instance_count, 0, 0, 0)?;

        Ok(())
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
pub struct GlowingBall {
    #[name("instance_pos")]
    #[format(R32G32_SFLOAT)]
    pub pos: [f32; 2],
    #[name("instance_color")]
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
    #[name("instance_radius")]
    #[format(R32_SFLOAT)]
    pub radius: f32,
    #[name("instance_corona")]
    #[format(R32_SFLOAT)]
    pub corona: f32,
    #[name("instance_lateAlpha")]
    #[format(R32_SFLOAT)]
    pub late_alpha: f32,
}
