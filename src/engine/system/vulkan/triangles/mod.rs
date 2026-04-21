use crate::engine::system::vulkan::buffers::BasicBuffersManager;
use crate::engine::system::vulkan::color::Color;
use crate::engine::system::vulkan::path::Path2d;
use crate::engine::system::vulkan::system::{GraphicsPipelineRenderPassInfo, VulkanSystem};
use crate::engine::system::vulkan::utils::Draw;
use crate::engine::system::vulkan::wds::WriteDescriptorSetManager;
use crate::engine::system::vulkan::{DrawError, PipelineCreateError, ShaderLoadError};
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

#[derive(Debug)]
pub struct Triangles {
    pub vertices: Vec<Vertex2d>,
    pub color: [f32; 4],
}

#[derive(Debug)]
pub struct TrianglesIndexed {
    pub vertices: Vec<Vertex2d>,
    pub indices: Vec<[u32; 3]>,
    pub color: [f32; 4],
}

pub(crate) enum Mode {
    Simple(Vec<Triangles>),
    Indexed(Vec<TrianglesIndexed>),
}

/// A simple helper type that constructs and collects [`Triangles`] via simple and nice API calls in
/// memory until flushed via a [`TrianglesPipeline`]to a command buffer (see
/// [`TriangleCanvas::flush_to`]).
#[derive(Default)]
pub struct TriangleCanvas {
    color: Color,
    pub(crate) triangles: Vec<Mode>,
}

impl TriangleCanvas {
    pub fn new(color: impl Into<Color>) -> Self {
        Self {
            color: color.into(),
            triangles: Vec::new(),
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

    fn extend_simple(&mut self, triangles: impl IntoIterator<Item = Triangles>) {
        match self.triangles.last_mut() {
            None | Some(Mode::Indexed(_)) => {
                self.triangles
                    .push(Mode::Simple(triangles.into_iter().collect()));
            }
            Some(Mode::Simple(current)) => {
                current.extend(triangles);
            }
        }
    }

    fn extend_indexed(&mut self, indexed: impl IntoIterator<Item = TrianglesIndexed>) {
        match self.triangles.last_mut() {
            None | Some(Mode::Simple(_)) => {
                self.triangles
                    .push(Mode::Indexed(indexed.into_iter().collect()));
            }
            Some(Mode::Indexed(current)) => {
                current.extend(indexed);
            }
        }
    }

    /// Flushes the content via the given [`TrianglesPipeline`] as draw commands to the given
    /// command buffer. Will clear known triangles on success. Does not free memory until dropped.
    /// If you want to prevent regular re-allocations, then keep this [`TriangleCanvas`] instance
    /// alive and re-use it for the next batch
    pub fn flush_to<P>(
        &mut self,
        pipeline: &TrianglesPipeline,
        builder: &mut AutoCommandBufferBuilder<P>,
    ) -> Result<(), DrawError> {
        if self.triangles.is_empty() {
            Ok(())
        } else {
            for mode in &self.triangles {
                match mode {
                    Mode::Simple(triangles) => {
                        if !triangles.is_empty() {
                            pipeline.draw(builder, &triangles[..])?;
                        }
                    }
                    Mode::Indexed(indexed) => {
                        if !indexed.is_empty() {
                            pipeline.draw_indexed(builder, &indexed[..])?;
                        }
                    }
                }
            }
            self.triangles.clear();
            Ok(())
        }
    }
}

#[cfg(feature = "lyon_tessellation")]
mod lyon_tesselation {
    use super::*;
    use crate::engine::types::world2d::Pos;
    use lyon_geom::point;
    use lyon_tessellation::geometry_builder::simple_builder;
    use lyon_tessellation::math::Point;
    use lyon_tessellation::path::Path;
    use lyon_tessellation::{
        FillOptions, FillTessellator, StrokeOptions, StrokeTessellator, TessellationError,
        VertexBuffers,
    };

    impl TriangleCanvas {
        pub fn fill_path_world_space(
            &mut self,
            f: impl FnOnce(&mut Path2d),
            mut fill_options: FillOptions,
            view: &Map2dView,
        ) {
            let mut result = Vec::default();
            let mut path =
                Path2d::default().with_tolerance(Path2d::DEFAULT_TOLERANCE / view.zoom());

            f(&mut path);

            fill_options.tolerance /= view.zoom();

            for segment in path.into_lines() {
                match Self::fill_path(
                    self.color.rgba,
                    segment
                        .into_iter()
                        .map(|p| view.position_world_to_screen(p)),
                    fill_options,
                ) {
                    Err(e) => {
                        warn!("Failed to tessellate path: {}", e);
                    }
                    Ok(indexed) => {
                        if !indexed.vertices.is_empty() {
                            result.push(indexed)
                        }
                    }
                }
            }

            if !result.is_empty() {
                self.extend_indexed(result);
            }
        }

        pub fn fill_path_screen_space(
            &mut self,
            f: impl FnOnce(&mut Path2d),
            fill_options: FillOptions,
        ) {
            let mut result = Vec::default();
            let mut path = Path2d::default().with_tolerance(Path2d::DEFAULT_TOLERANCE);

            f(&mut path);

            for segment in path.into_lines() {
                match Self::fill_path(self.color.rgba, segment.into_iter(), fill_options) {
                    Err(e) => {
                        warn!("Failed to tessellate path: {}", e);
                    }
                    Ok(indexed) => {
                        if !indexed.vertices.is_empty() {
                            result.push(indexed)
                        }
                    }
                }
            }

            if !result.is_empty() {
                self.extend_indexed(result);
            }
        }

        fn fill_path(
            color: [f32; 4],
            mut segment: impl Iterator<Item = Pos<f32>>,
            fill_options: FillOptions,
        ) -> Result<TrianglesIndexed, TessellationError> {
            let mut tessellator = FillTessellator::new();

            let mut path = Path::builder();
            let mut vertex_builder: VertexBuffers<Point, u16> = VertexBuffers::<Point, u16>::new();

            if let Some(p) = segment.next() {
                path.begin(point(p.x, p.y));
            }

            for p in segment {
                path.line_to(point(p.x, p.y));
            }
            path.end(true);

            tessellator.tessellate_path(
                &path.build(),
                &fill_options,
                &mut simple_builder(&mut vertex_builder),
            )?;

            Ok(TrianglesIndexed {
                color,
                vertices: vertex_builder
                    .vertices
                    .into_iter()
                    .map(|v| Vertex2d { pos: [v.x, v.y] })
                    .collect::<Vec<_>>(),
                indices: vertex_builder
                    .indices
                    .chunks(3)
                    .map(|chunk| [chunk[0] as u32, chunk[1] as u32, chunk[2] as u32])
                    .collect::<Vec<_>>(),
            })
        }

        pub fn stroke_path_screen_space(
            &mut self,
            f: impl FnOnce(&mut Path2d),
            stroke_options: StrokeOptions,
        ) {
            let mut result = Vec::default();
            let mut path = Path2d::default();

            f(&mut path);

            for segment in path.into_lines() {
                match Self::stroke_path(self.color.rgba, segment.into_iter(), stroke_options) {
                    Err(e) => {
                        warn!("Failed to tessellate path: {}", e);
                    }
                    Ok(indexed) => {
                        if !indexed.vertices.is_empty() {
                            result.push(indexed)
                        }
                    }
                }
            }

            if !result.is_empty() {
                self.extend_indexed(result);
            }
        }

        pub fn stroke_path_world_space(
            &mut self,
            f: impl FnOnce(&mut Path2d),
            mut stroke_options: StrokeOptions,
            view: &Map2dView,
        ) {
            let mut result = Vec::default();
            let mut path =
                Path2d::default().with_tolerance(Path2d::DEFAULT_TOLERANCE / view.zoom());

            f(&mut path);

            stroke_options.line_width *= view.zoom();
            stroke_options.tolerance /= view.zoom();

            for segment in path.into_lines() {
                match Self::stroke_path(
                    self.color.rgba,
                    segment
                        .into_iter()
                        .map(|p| view.position_world_to_screen(p)),
                    stroke_options,
                ) {
                    Err(e) => {
                        warn!("Failed to tessellate path: {}", e);
                    }
                    Ok(indexed) => {
                        if !indexed.vertices.is_empty() {
                            result.push(indexed)
                        }
                    }
                }
            }

            if !result.is_empty() {
                self.extend_indexed(result);
            }
        }

        fn stroke_path(
            color: [f32; 4],
            mut segment: impl Iterator<Item = Pos<f32>>,
            stroke_options: StrokeOptions,
        ) -> Result<TrianglesIndexed, TessellationError> {
            let mut tessellator = StrokeTessellator::new();

            let mut path = Path::builder();
            let mut vertex_builder: VertexBuffers<Point, u16> = VertexBuffers::<Point, u16>::new();

            if let Some(p) = segment.next() {
                path.begin(point(p.x, p.y));
            }

            for p in segment {
                path.line_to(point(p.x, p.y));
            }
            path.end(true);

            tessellator.tessellate_path(
                &path.build(),
                &stroke_options,
                &mut simple_builder(&mut vertex_builder),
            )?;

            Ok(TrianglesIndexed {
                color,
                vertices: vertex_builder
                    .vertices
                    .into_iter()
                    .map(|v| Vertex2d { pos: [v.x, v.y] })
                    .collect::<Vec<_>>(),
                indices: vertex_builder
                    .indices
                    .chunks(3)
                    .map(|chunk| [chunk[0] as u32, chunk[1] as u32, chunk[2] as u32])
                    .collect::<Vec<_>>(),
            })
        }
    }
}

impl Drop for TriangleCanvas {
    fn drop(&mut self) {
        if !self.triangles.is_empty() {
            warn!(
                "Dropping {} without being flushed.",
                core::any::type_name::<Self>()
            );
        }
    }
}
