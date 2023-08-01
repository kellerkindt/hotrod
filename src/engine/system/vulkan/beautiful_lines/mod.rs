use crate::engine::system::vulkan::VulkanSystem;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferAllocateInfo, BufferError, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PipelineExecutionError};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::{Device, Features, Queue};
use vulkano::format::Format;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::render_pass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Scissor, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreationError;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, StateMode};
use vulkano::shader::ShaderModule;

pub struct VulkanBeautifulLineSystem {
    queue: Arc<Queue>,
    pipeline: Arc<GraphicsPipeline>,
    desc_allocator: StandardDescriptorSetAllocator,
    memo_allocator: StandardMemoryAllocator,
}

impl TryFrom<&VulkanSystem> for VulkanBeautifulLineSystem {
    type Error = GraphicsPipelineCreationError;

    fn try_from(vs: &VulkanSystem) -> Result<Self, Self::Error> {
        Self::new(
            Arc::clone(&vs.device()),
            Arc::clone(&vs.queue()),
            vs.image_format(),
        )
    }
}

impl VulkanBeautifulLineSystem {
    pub const REQUIRED_FEATURES: Features = Features {
        dynamic_rendering: true,
        wide_lines: true,
        ..Features::empty()
    };

    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        image_format: Format,
    ) -> Result<Self, GraphicsPipelineCreationError> {
        Ok(Self {
            queue,
            desc_allocator: StandardDescriptorSetAllocator::new(Arc::clone(&device)),
            memo_allocator: StandardMemoryAllocator::new_default(Arc::clone(&device)),
            pipeline: Self::create_pipeline(Arc::clone(&device), image_format)?,
        })
    }

    fn create_pipeline(
        device: Arc<Device>,
        image_format: Format,
    ) -> Result<Arc<GraphicsPipeline>, GraphicsPipelineCreationError> {
        GraphicsPipeline::start()
            .vertex_input_state(Vertex2d::per_vertex())
            .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::LineStrip))
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
            .rasterization_state(RasterizationState {
                cull_mode: StateMode::Fixed(CullMode::None),
                line_width: StateMode::Dynamic,
                ..RasterizationState::new()
            })
            .color_blend_state(ColorBlendState::new(1).blend(AttachmentBlend {
                // color_source: BlendFactor::One,
                // alpha_op: BlendOp::ReverseSubtract,
                // colo
                ..AttachmentBlend::alpha()
            }))
            .render_pass(PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(image_format)],
                ..Default::default()
            })
            .build(device)
    }

    fn load_vertex_shader(device: Arc<Device>) -> Arc<ShaderModule> {
        mod shader {
            vulkano_shaders::shader!(
                ty: "vertex",
                path: "src/engine/system/vulkan/beautiful_lines/lines.vert"
            );
        }
        shader::load(device).unwrap()
    }

    fn load_fragment_shader(device: Arc<Device>) -> Arc<ShaderModule> {
        mod shader {
            vulkano_shaders::shader!(
                ty: "fragment",
                path: "src/engine/system/vulkan/beautiful_lines/lines.frag"
            );
        }
        shader::load(device).unwrap()
    }

    pub fn draw<P>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<P>,
        width: f32,
        height: f32,
        lines: &[BeautifulLine],
    ) -> Result<(), DrawError> {
        builder.bind_pipeline_graphics(Arc::clone(&self.pipeline));

        let mut offset = 0;
        let vertex_buffer = self.create_vertex_buffer(
            lines
                .iter()
                .flat_map(|l| l.vertices.iter().copied())
                .collect::<Vec<_>>(),
        )?;

        for line in lines {
            let mut scissor = Scissor::irrelevant();
            for v in &line.vertices {
                scissor.origin[0] = scissor.origin[0].min(v.pos[0] as u32);
                scissor.origin[1] = scissor.origin[1].min(v.pos[1] as u32);
                scissor.dimensions[0] = scissor.dimensions[0].max(v.pos[0] as u32);
                scissor.dimensions[1] = scissor.dimensions[1].max(v.pos[1] as u32);
            }
            scissor.dimensions[0] -= scissor.origin[0];
            scissor.dimensions[1] -= scissor.origin[1];

            let vertices = vertex_buffer
                .clone()
                .slice(offset..(offset + line.vertices.len() as u64));

            offset += line.vertices.len() as u64;

            builder
                .set_line_width(line.width)
                .set_scissor(0, [scissor])
                .bind_vertex_buffers(0, vertices)
                .push_constants(
                    Arc::clone(&self.pipeline.layout()),
                    0,
                    [width, height, line.width],
                )
                .draw(line.vertices.len() as u32, 1, 0, 0)?;
        }

        Ok(())
    }

    fn create_vertex_buffer(
        &self,
        vertices: Vec<Vertex2d>,
    ) -> Result<Subbuffer<[Vertex2d]>, BufferError> {
        Buffer::from_iter(
            &self.memo_allocator,
            BufferAllocateInfo {
                buffer_usage: BufferUsage::VERTEX_BUFFER,
                ..BufferAllocateInfo::default()
            },
            vertices,
        )
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod, Vertex)]
pub struct Vertex2d {
    #[format(R32G32_SFLOAT)]
    pub pos: [f32; 2],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
}

pub struct BeautifulLine {
    pub vertices: Vec<Vertex2d>,
    pub width: f32,
}

#[derive(thiserror::Error, Debug)]
pub enum DrawError {
    #[error("Failed to load buffer: {0}")]
    BufferError(#[from] BufferError),
    #[error("Failed to execute the pipeline: {0}")]
    PipelineExecutionError(#[from] PipelineExecutionError),
}
