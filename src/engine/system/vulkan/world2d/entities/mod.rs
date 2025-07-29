use crate::engine::system::vulkan::buffers::BasicBuffersManager;
use crate::engine::system::vulkan::system::{GraphicsPipelineRenderPassInfo, VulkanSystem};
use crate::engine::system::vulkan::textures::{ImageSamplerMode, TextureId, TextureManager};
use crate::engine::system::vulkan::wds::WriteDescriptorSetManager;
use crate::engine::system::vulkan::{DrawError, PipelineCreateError, ShaderLoadError};
use crate::shader_from_path;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use vulkano::buffer::{IndexBuffer, Subbuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::device::Device;
use vulkano::image::Image;
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
use vulkano::{Validated, VulkanError};

/// This pipeline is used to draw the entities of 2d worlds. A 2d world entity consists of quadratic
/// area at a certain point and an individual size.
#[derive()]
pub struct World2dEntitiesPipeline {
    pipeline: Arc<GraphicsPipeline>,
    buffers_manager: Arc<BasicBuffersManager>,
    quad_index_buffer: IndexBuffer,
    quad_vertex_buffer: Subbuffer<[Vertex2d]>,
    write_descriptors: Arc<WriteDescriptorSetManager>,
    texture_manager: TextureManager<Self, 0>,
}

impl TryFrom<&VulkanSystem> for World2dEntitiesPipeline {
    type Error = PipelineCreateError;

    #[inline]
    fn try_from(vs: &VulkanSystem) -> Result<Self, Self::Error> {
        Self::new(
            Arc::clone(vs.device()),
            vs.graphics_pipeline_render_pass_info(),
            vs.pipeline_cache().map(Arc::clone),
            Arc::clone(vs.write_descriptor_set_manager()),
            Arc::clone(vs.basic_buffers_manager()),
        )
    }
}

impl World2dEntitiesPipeline {
    pub fn new(
        device: Arc<Device>,
        render_pass_info: GraphicsPipelineRenderPassInfo,
        cache: Option<Arc<PipelineCache>>,
        write_descriptors: Arc<WriteDescriptorSetManager>,
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
            write_descriptors,
            buffers_manager,
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
        render_pass_info: GraphicsPipelineRenderPassInfo,
        cache: Option<Arc<PipelineCache>>,
    ) -> Result<Arc<GraphicsPipeline>, PipelineCreateError> {
        let vs = Self::load_vertex_shader(Arc::clone(&device))?;
        let fs = Self::load_fragment_shader(Arc::clone(&device))?;

        let vertex_input_state = [Vertex2d::per_vertex(), EntityInstanceData::per_instance()]
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
            "src/engine/system/vulkan/world2d/entities/entities.vert"
        )
    }

    fn load_fragment_shader(device: Arc<Device>) -> Result<EntryPoint, ShaderLoadError> {
        shader_from_path!(
            device,
            "fragment",
            "src/engine/system/vulkan/world2d/entities/entities.frag"
        )
    }

    pub fn prepare_draw<I>(
        &self,
        texture: &TextureId<Self>,
        tiles: I,
    ) -> Result<EntityPreparedDraw, DrawError>
    where
        I: IntoIterator<Item = EntityInstanceData>,
        I::IntoIter: ExactSizeIterator,
    {
        if self.texture_manager.is_origin_of(texture) {
            let vertex_buffer = self.buffers_manager.create_vertex_buffer(tiles)?;
            Ok(EntityPreparedDraw {
                vertex_buffer,
                texture_id: texture.clone(),
            })
        } else {
            todo!()
        }
    }

    #[inline]
    pub fn draw_prepared<'a, P, I>(
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        prepared: I,
    ) -> Result<(), DrawError>
    where
        I: Iterator<Item = &'a EntityPreparedDraw> + 'a,
    {
        let mut prepared = prepared.peekable();

        // Call draw_with in batches so that for each batch the TextureId is the same
        while let Some(texture) = prepared.peek().map(|p| p.texture_id.clone()) {
            self.draw_with(
                builder,
                &texture,
                (&mut prepared)
                    .take_while(|prepared| prepared.texture_id == texture)
                    .map(|prepared| prepared.vertex_buffer.clone()),
            )?;
        }

        Ok(())
    }

    #[inline]
    pub fn draw<P, I>(
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        texture: &TextureId<Self>,
        tiles: I,
    ) -> Result<(), DrawError>
    where
        I: IntoIterator<Item = EntityInstanceData>,
        I::IntoIter: ExactSizeIterator,
    {
        let vertex_buffer = self.buffers_manager.create_vertex_buffer(tiles)?;
        self.draw_with(builder, texture, Some(vertex_buffer).into_iter())
    }

    fn draw_with<P, I>(
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        texture: &TextureId<Self>,
        vertex_buffers: I,
    ) -> Result<(), DrawError>
    where
        I: Iterator<Item = Subbuffer<[EntityInstanceData]>>,
    {
        if self.texture_manager.is_origin_of(texture) {
            builder
                .bind_pipeline_graphics(Arc::clone(&self.pipeline))?
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    Arc::clone(&self.pipeline.layout()),
                    0,
                    Arc::clone(&texture.0.descriptor),
                )?
                .bind_index_buffer(self.quad_index_buffer.clone())?;

            for vertex_buffer in vertex_buffers {
                let instance_count = vertex_buffer.len() as u32;
                builder
                    .bind_vertex_buffers(
                        0,
                        [
                            self.quad_vertex_buffer.as_bytes().clone(),
                            vertex_buffer.into_bytes(),
                        ],
                    )?
                    .draw_indexed(6, instance_count, 0, 0, 0)?;
            }

            Ok(())
        } else {
            todo!()
        }
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
pub struct Vertex2d {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod, Vertex)]
pub struct EntityInstanceData {
    #[format(R32G32_SFLOAT)]
    pub entity_pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub uv0: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub uv1: [f32; 2],
    #[format(R32_SFLOAT)]
    pub size: f32,
}

pub struct EntityPreparedDraw {
    vertex_buffer: Subbuffer<[EntityInstanceData]>,
    texture_id: TextureId<World2dEntitiesPipeline>,
}
