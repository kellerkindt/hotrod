use crate::engine::system::vulkan::buffers::BasicBuffersManager;
use crate::engine::system::vulkan::system::{GraphicsPipelineRenderPassInfo, VulkanSystem};
use crate::engine::system::vulkan::utils::Draw;
use crate::engine::system::vulkan::wds::WriteDescriptorSetManager;
use crate::engine::system::vulkan::{DrawError, PipelineCreateError, ShaderLoadError};
use crate::shader_from_path;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::{Device, DeviceFeatures};
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

#[derive()]
pub struct TrianglesPipeline {
    pipeline: Arc<GraphicsPipeline>,
    buffers_manager: Arc<BasicBuffersManager>,
    descriptor_set: Arc<DescriptorSet>,
}

impl TryFrom<&VulkanSystem> for TrianglesPipeline {
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

impl TrianglesPipeline {
    pub const REQUIRED_FEATURES: DeviceFeatures = DeviceFeatures {
        dynamic_rendering: true,
        ..DeviceFeatures::empty()
    };

    pub fn new(
        device: Arc<Device>,
        render_pass_info: GraphicsPipelineRenderPassInfo,
        cache: Option<Arc<PipelineCache>>,
        write_descriptors: &WriteDescriptorSetManager,
        buffers_manager: Arc<BasicBuffersManager>,
    ) -> Result<Self, PipelineCreateError> {
        let pipeline = Self::create_pipeline(Arc::clone(&device), render_pass_info, cache)?;
        Ok(Self {
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

        let vertex_input_state = Vertex2d::per_vertex().definition(&vs)?;

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
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        triangles: &[Triangles],
    ) -> Result<(), DrawError> {
        let mut offset = 0;

        let vertex_buffer = self.buffers_manager.create_vertex_buffer(
            triangles
                .iter()
                .flat_map(|l| l.vertices.iter().copied())
                .collect::<Vec<_>>(),
        )?;

        builder
            .bind_pipeline_graphics(Arc::clone(&self.pipeline))?
            .bind_vertex_buffers(0, vertex_buffer)?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                Arc::clone(&self.pipeline.layout()),
                0,
                Arc::clone(&self.descriptor_set),
            )?;

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
                    ],
                )?
                .hotrod_draw(triangles.vertices.len() as u32, 1, offset, 0)?;
            offset += triangles.vertices.len() as u32;
        }

        Ok(())
    }

    pub fn draw_indexed<P>(
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        triangles: &[TrianglesIndexed],
    ) -> Result<(), DrawError> {
        let mut offset_vertices = 0;
        let mut offset_indices = 0;

        let vertex_buffer = self.buffers_manager.create_vertex_buffer(
            triangles
                .iter()
                .flat_map(|l| l.vertices.iter().copied())
                .collect::<Vec<_>>(),
        )?;

        let index_buffer = self.buffers_manager.create_index_buffer(
            triangles
                .iter()
                .flat_map(|l| l.indices.iter().flat_map(|i| i.into_iter()).copied())
                .collect::<Vec<_>>(),
        )?;

        builder
            .bind_pipeline_graphics(Arc::clone(&self.pipeline))?
            .bind_index_buffer(index_buffer)?
            .bind_vertex_buffers(0, vertex_buffer)?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                Arc::clone(&self.pipeline.layout()),
                0,
                Arc::clone(&self.descriptor_set),
            )?;

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
                    ],
                )?
                .hotrod_draw_indexed(index_count, 1, offset_indices, offset_vertices, 0)?;

            offset_vertices += triangles.vertices.len() as i32;
            offset_indices += index_count as u32;
        }

        Ok(())
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
