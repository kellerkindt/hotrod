use crate::engine::system::vulkan::buffers::BasicBuffersManager;
use crate::engine::system::vulkan::color::Color;
use crate::engine::system::vulkan::path::Path2d;
use crate::engine::system::vulkan::system::{GraphicsPipelineRenderPassInfo, VulkanSystem};
use crate::engine::system::vulkan::triangles::{Mode, TriangleCanvas};
use crate::engine::system::vulkan::utils::Draw;
use crate::engine::system::vulkan::wds::WriteDescriptorSetManager;
use crate::engine::system::vulkan::{DrawError, PipelineCreateError, ShaderLoadError};
use crate::engine::types::world2d::Pos;
use crate::shader_from_path;
use crate::support::world2d::view::Map2dView;
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

pub struct LinePipeline {
    pipeline: Arc<GraphicsPipeline>,
    buffers_manager: Arc<BasicBuffersManager>,
    descriptor_set: Arc<DescriptorSet>,
}

impl TryFrom<&VulkanSystem> for LinePipeline {
    type Error = PipelineCreateError;

    #[inline]
    fn try_from(vs: &VulkanSystem) -> Result<Self, Self::Error> {
        Self::new(
            Arc::clone(vs.device()),
            vs.graphics_pipeline_render_pass_info(),
            vs.pipeline_cache().map(Arc::clone),
            vs.write_descriptor_set_manager(),
            Arc::clone(&vs.basic_buffers_manager()),
        )
    }
}

impl LinePipeline {
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
                    topology: PrimitiveTopology::LineStrip,
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
            "src/engine/system/vulkan/lines/lines.vert"
        )
    }

    fn load_fragment_shader(device: Arc<Device>) -> Result<EntryPoint, ShaderLoadError> {
        shader_from_path!(
            device,
            "fragment",
            "src/engine/system/vulkan/lines/lines.frag"
        )
    }

    pub fn draw<P>(
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        lines: &[Line],
    ) -> Result<(), DrawError> {
        let mut offset = 0;
        let vertex_buffer = self.buffers_manager.create_vertex_buffer(
            lines
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

        for line in lines {
            builder
                .push_constants(
                    Arc::clone(&self.pipeline.layout()),
                    0,
                    [line.color[0], line.color[1], line.color[2], line.color[3]],
                )?
                .hotrod_draw(line.vertices.len() as u32, 1, offset, 0)?;

            offset += line.vertices.len() as u32;
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

pub struct Line {
    pub vertices: Vec<Vertex2d>,
    pub color: [f32; 4],
}

/// A simple helper type that constructs and collects [`Line`]s via simple and nice API calls in
/// memory until flushed with a [`LinePipeline`] to a command buffer (see [`LineCanvas::flush_to`]).
#[derive(Default)]
pub struct LineCanvas {
    color: Color,
    lines: Vec<Line>,
}

impl LineCanvas {
    pub fn new(color: impl Into<Color>) -> Self {
        Self {
            color: color.into(),
            lines: Vec::new(),
        }
    }

    pub fn set_color(&mut self, color: impl Into<Color>) {
        self.color = color.into();
    }

    pub fn with_color(mut self, color: impl Into<Color>) -> Self {
        self.set_color(color);
        self
    }

    pub fn colored(&mut self, color: impl Into<Color>) -> &mut Self {
        self.set_color(color);
        self
    }

    pub fn draw_line_screen_space<T: Into<Pos<f32>>>(&mut self, points: impl Iterator<Item = T>) {
        self.lines.push(Line {
            color: self.color.rgba,
            vertices: points
                .map(|position| {
                    let pos = position.into();
                    Vertex2d { pos: pos.into() }
                })
                .collect(),
        });
    }

    #[inline]
    pub fn draw_line_world_space<T: Into<Pos<f32>>>(
        &mut self,
        points: impl Iterator<Item = T>,
        view: &Map2dView,
    ) {
        self.draw_line_screen_space(points.map(|world| {
            let screen = view.position_world_to_screen(world.into());
            (screen.x, screen.y)
        }))
    }

    pub fn draw_path_screen_space(&mut self, f: impl FnOnce(&mut Path2d)) {
        let mut path = Path2d::default();
        f(&mut path);
        for line in path.into_lines() {
            self.draw_line_screen_space(line.into_iter());
        }
    }

    pub fn draw_path_world_space(&mut self, f: impl FnOnce(&mut Path2d), view: &Map2dView) {
        let mut path = Path2d::default().with_tolerance(Path2d::DEFAULT_TOLERANCE / view.zoom());
        f(&mut path);
        for line in path.into_lines() {
            self.draw_line_world_space(line.into_iter(), view);
        }
    }

    pub fn draw_mesh(&mut self, triangle_canvas: &TriangleCanvas) {
        for mode in &triangle_canvas.triangles {
            match mode {
                Mode::Simple(triangles) => {
                    for triangle in triangles {
                        self.lines.push(Line {
                            color: self.color.rgba,
                            vertices: triangle
                                .vertices
                                .iter()
                                .map(|v| Vertex2d { pos: v.pos })
                                .collect(),
                        });
                    }
                }
                Mode::Indexed(indexed) => {
                    for indexed in indexed {
                        for indices in &indexed.indices {
                            self.lines.push(Line {
                                color: self.color.rgba,
                                vertices: indices
                                    .into_iter()
                                    .map(|i| indexed.vertices[*i as usize].pos)
                                    .map(|p| Vertex2d { pos: p })
                                    .collect(),
                            });
                        }
                    }
                }
            }
        }
    }

    /// Flushes the content via the given [`LinePipeline`] as draw commands to the given command
    /// buffer. Will clear known lines on success. Does not free memory until dropped. If you want
    /// to prevent regular re-allocations, then keep this [`LineCanvas`] instance alive and re-use
    /// it for the next batch.
    pub fn flush_to<P>(
        &mut self,
        pipeline: &LinePipeline,
        builder: &mut AutoCommandBufferBuilder<P>,
    ) -> Result<(), DrawError> {
        if self.lines.is_empty() {
            Ok(())
        } else {
            match pipeline.draw(builder, &self.lines[..]) {
                Ok(()) => {
                    self.lines.clear();
                    Ok(())
                }
                Err(e) => Err(e),
            }
        }
    }
}

impl Drop for LineCanvas {
    fn drop(&mut self) {
        if !self.lines.is_empty() {
            warn!(
                "Dropping {} without being flushed.",
                core::any::type_name::<Self>()
            );
        }
    }
}
