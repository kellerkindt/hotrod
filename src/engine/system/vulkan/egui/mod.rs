use crate::engine::system::egui::EguiSystem;
use crate::engine::system::vulkan::buffers::BasicBuffersManager;
use crate::engine::system::vulkan::system::{GraphicsPipelineRenderPassInfo, VulkanSystem};
use crate::engine::system::vulkan::textures::{
    CopyInfo, CopyRequestWaiter, ImageSamplerMode, ImageSystem, TextureId, TextureManager,
};
use crate::engine::system::vulkan::utils::Draw;
use crate::engine::system::vulkan::{DrawError, PipelineCreateError, ShaderLoadError, UploadError};
use crate::shader_from_path;
use crate::ui::egui::epaint::{ImageDelta, Primitive};
use crate::ui::egui::{TextureFilter, TextureWrapMode};
use bytemuck::{Pod, Zeroable};
use egui::{Color32, ImageData, Rect, TextureId as EguiTextureId, TextureOptions, TexturesDelta};
use nohash_hasher::NoHashHasher;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::ops::DerefMut;
use std::sync::{Arc, RwLock};
use vulkano::buffer::AllocateBufferError;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::device::{Device, Queue};
use vulkano::image::sampler::{Filter, Sampler, SamplerCreateInfo, SamplerMipmapMode};
use vulkano::image::{AllocateImageError, Image};
use vulkano::pipeline::cache::PipelineCache;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Scissor, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::shader::EntryPoint;
use vulkano::{Validated, VulkanError};

type TextureSamplers = HashMap<TextureOptions, Arc<Sampler>>;

struct TextureState {
    id: TextureId<EguiPipeline>,
    image: Arc<Image>,
    waiter: Option<CopyRequestWaiter>,
}

struct Inner {
    pub textures: HashMap<IdWrapper, TextureState, BuildHasherDefault<NoHashHasher<u64>>>,
    pub textures_to_free: Vec<EguiTextureId>,
    pub texture_samplers: TextureSamplers,
}

pub struct EguiPipeline {
    pub queue: Arc<Queue>,
    pub pipeline: Arc<GraphicsPipeline>,
    buffers_manager: Arc<BasicBuffersManager>,
    image_system: Arc<ImageSystem>,
    texture_manager: TextureManager<Self, 0>,
    inner: Arc<RwLock<Inner>>,
    device: Arc<Device>,
}

impl TryFrom<&VulkanSystem> for EguiPipeline {
    type Error = PipelineCreateError;

    fn try_from(vs: &VulkanSystem) -> Result<Self, Self::Error> {
        Self::new(
            Arc::clone(vs.device()),
            Arc::clone(vs.queue()),
            vs.graphics_pipeline_render_pass_info(),
            vs.pipeline_cache().map(Arc::clone),
            Arc::clone(vs.basic_buffers_manager()),
            Arc::clone(vs.image_system()),
        )
    }
}

impl EguiPipeline {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        render_pass_info: GraphicsPipelineRenderPassInfo,
        cache: Option<Arc<PipelineCache>>,
        buffers_manager: Arc<BasicBuffersManager>,
        image_system: Arc<ImageSystem>,
    ) -> Result<Self, PipelineCreateError> {
        let pipeline = Self::create_pipeline(Arc::clone(&device), render_pass_info, cache)?;
        let texture_manager =
            TextureManager::basic(Arc::clone(&device), &pipeline, ImageSamplerMode::Linear)?;
        Ok(Self {
            queue,
            inner: Arc::new(RwLock::new(Inner {
                textures: HashMap::default(),
                textures_to_free: Vec::default(),
                texture_samplers: [(
                    TextureOptions {
                        magnification: TextureFilter::Linear,
                        minification: TextureFilter::Linear,
                        wrap_mode: TextureWrapMode::ClampToEdge,
                        mipmap_mode: None,
                    },
                    Arc::clone(&texture_manager.sampler()),
                )]
                .into_iter()
                .collect::<HashMap<_, _>>(),
            })),
            device,
            buffers_manager,
            image_system,
            texture_manager,
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

        let vertex_input_state = AdapterVertex::per_vertex().definition(&vs)?;

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
                viewport_state: Some(ViewportState::default()), // Some(ViewportState::viewport_dynamic_scissor_dynamic(1)),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState {
                    rasterization_samples: render_pass_info.rasterization_samples(),
                    ..MultisampleState::default()
                }),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    render_pass_info.num_color_attachments(),
                    ColorBlendAttachmentState {
                        // was before - erroneous?
                        // .src_color_blend_factor = BlendFactor::One;
                        // .dst_color_blend_factor = BlendFactor::OneMinusSrcAlpha;
                        // .color_blend_op = BlendOp::Add;
                        // .src_alpha_blend_factor = BlendFactor::OneMinusDstAlpha;
                        // .dst_alpha_blend_factor = BlendFactor::One;
                        // .alpha_blend_op = BlendOp::Add;
                        blend: Some(AttachmentBlend::alpha()),
                        ..ColorBlendAttachmentState::default()
                    },
                )),
                subpass: Some(render_pass_info.into_subpass_type()),
                dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                    .into_iter()
                    .collect(),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?)
    }

    fn create_texture_sampler(
        device: Arc<Device>,
        options: TextureOptions,
    ) -> Result<Arc<Sampler>, Validated<VulkanError>> {
        fn from_egui_filter(filter: TextureFilter) -> Filter {
            match filter {
                TextureFilter::Nearest => Filter::Nearest,
                TextureFilter::Linear => Filter::Linear,
            }
        }

        Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: from_egui_filter(options.magnification),
                min_filter: from_egui_filter(options.minification),
                mipmap_mode: match from_egui_filter(options.minification) {
                    Filter::Linear => SamplerMipmapMode::Linear,
                    _ => SamplerMipmapMode::Nearest,
                },
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

    #[inline]
    pub fn prepare(&self, egui: &EguiSystem) -> Result<(), UploadError> {
        self.update_textures(&egui.texture_delta)
    }

    #[inline]
    pub fn draw<P>(
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        egui: &EguiSystem,
    ) -> Result<(), DrawError> {
        let pixels_per_point = egui.pixels_per_point();
        let points_view_area = egui.points_view_area();
        let clipped_primitives = &egui.clipped_primitives;

        let mut vertices = Vec::<AdapterVertex>::with_capacity(clipped_primitives.len() * 4);
        let mut indices = Vec::<u32>::with_capacity(clipped_primitives.len() * 6);
        let mut clip_rects = Vec::<Rect>::with_capacity(clipped_primitives.len());
        let mut texture_ids = Vec::<EguiTextureId>::with_capacity(clipped_primitives.len());
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

        let vertex_buffer = self.buffers_manager.create_vertex_buffer(vertices)?;
        let index_buffer = self.buffers_manager.create_index_buffer(indices)?;

        builder
            //.next_subpass(SubpassContents::Inline)?
            .bind_pipeline_graphics(Arc::clone(&self.pipeline))?
            .bind_index_buffer(index_buffer)?
            .bind_vertex_buffers(0, vertex_buffer)?
            .push_constants(
                Arc::clone(&self.pipeline.layout()),
                0,
                [points_view_area.width(), points_view_area.height()],
            )?;

        let mut inner = self.inner.write().unwrap();
        for (index, rect) in clip_rects.into_iter().enumerate() {
            let (offset_vertex, offset_index) = offsets[index];
            let (_offset_vertex_end, offset_index_end) = offsets[index + 1];

            let texture_id = IdWrapper::from(texture_ids[index]);
            if let Some(texture) = inner.textures.get_mut(&texture_id) {
                if let Some(waiter) = &texture.waiter {
                    if waiter.is_complete() {
                        // no need to check again
                        texture.waiter = None;
                    } else {
                        // can't use it for now
                        continue;
                    }
                };

                builder
                    .set_scissor(
                        0,
                        [Scissor {
                            offset: [
                                (rect.min.x * pixels_per_point).floor() as u32,
                                (rect.min.y * pixels_per_point).floor() as u32,
                            ],
                            extent: [
                                (rect.width() * pixels_per_point).ceil() as u32,
                                (rect.height() * pixels_per_point).ceil() as u32,
                            ],
                        }]
                        .into_iter()
                        .collect(),
                    )?
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        Arc::clone(&self.pipeline.layout()),
                        0,
                        Arc::clone(texture.id.descriptor()),
                    )?
                    .hotrod_draw_indexed(
                        (offset_index_end - offset_index) as u32,
                        1,
                        offset_index as u32,
                        offset_vertex as i32,
                        0,
                    )?;
            }
        }

        drop(inner);
        self.free_textures();
        Ok(())
    }

    fn free_textures(&self) {
        let mut inner = self.inner.write().unwrap();
        let inner = inner.deref_mut();
        for texture in inner.textures_to_free.drain(..).map(IdWrapper::from) {
            inner.textures.remove(&texture);
        }
    }

    /// To prevent flickering, some texture updates that must be executed before the next draw call.
    fn update_textures(&self, textures_delta: &TexturesDelta) -> Result<(), UploadError> {
        let mut inner = self.inner.write().unwrap();

        inner
            .textures_to_free
            .extend(textures_delta.free.iter().copied());

        // this clone is cheap (just a few bytes, the image is inside an Arc)
        for (texture_id, delta) in textures_delta.set.iter().cloned() {
            // the egui default texture is high priority, everything else can wait a frame if needed
            let high_priority = texture_id == egui::TextureId::default();
            let texture_id = IdWrapper::from(texture_id);

            if delta.is_whole() {
                let image = self.create_image(&delta.image)?;
                let texture = self.prepare_texture(&mut inner.texture_samplers, &delta, &image)?;

                inner.textures.insert(
                    texture_id,
                    TextureState {
                        id: texture,
                        image: Arc::clone(&image),
                        waiter: Self::enqueue_image_upload_or_delta_update(
                            &self.image_system,
                            image,
                            delta,
                            high_priority,
                        )?,
                    },
                );
            } else {
                if let Some(state) = inner.textures.get_mut(&texture_id) {
                    state.waiter = Self::enqueue_image_upload_or_delta_update(
                        &self.image_system,
                        Arc::clone(&state.image),
                        delta,
                        high_priority,
                    )?;
                }
            }
        }

        Ok(())
    }

    fn prepare_texture(
        &self,
        texture_samplers: &mut TextureSamplers,
        delta: &ImageDelta,
        image: &Arc<Image>,
    ) -> Result<TextureId<EguiPipeline>, Validated<VulkanError>> {
        self.texture_manager.prepare_texture_with(
            Arc::clone(&image),
            Arc::clone(
                texture_samplers
                    .entry(delta.options.clone())
                    .or_insert_with(|| {
                        Self::create_texture_sampler(
                            Arc::clone(&self.device),
                            delta.options.clone(),
                        )
                        .unwrap()
                    }),
            ),
            [].into_iter(),
        )
    }

    #[inline]
    fn create_image(&self, image: &ImageData) -> Result<Arc<Image>, Validated<AllocateImageError>> {
        self.image_system
            .create_image(image.width() as u32, image.height() as u32)
    }

    #[inline]
    fn enqueue_image_upload_or_delta_update(
        image_system: &ImageSystem,
        image: Arc<Image>,
        delta: ImageDelta,
        high_priority: bool,
    ) -> Result<Option<CopyRequestWaiter>, Validated<AllocateBufferError>> {
        let estimated_size = {
            let pixels = delta
                .pos
                .map(|[x, y]| {
                    let widht = delta.image.width().saturating_sub(x);
                    let height = delta.image.height().saturating_sub(y);

                    widht * height
                })
                .unwrap_or_else(|| match &delta.image {
                    ImageData::Color(color_data) => {
                        let [width, height] = color_data.size;
                        width * height
                    }
                });
            pixels * 4 // RGBA / Color32
        };

        let request = move |image_system: &ImageSystem| {
            image_system.create_copy_request(
                image,
                delta.pos.map(|[x, y]| {
                    (
                        [x as u32, y as u32],
                        [delta.image.width() as u32, delta.image.height() as u32],
                    )
                }),
                match &delta.image {
                    ImageData::Color(color_data) => color_data
                        .pixels
                        .iter()
                        .flat_map(Color32::to_array)
                        .collect::<Vec<_>>(),
                },
            )
        };

        Ok(Some(image_system.enqueue_copy_request(
            CopyInfo::Deferred(Box::new(request)),
            estimated_size,
            high_priority,
        )))
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

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
struct IdWrapper(u64);

impl From<EguiTextureId> for IdWrapper {
    fn from(value: EguiTextureId) -> Self {
        match value {
            EguiTextureId::Managed(id) | EguiTextureId::User(id) if id.leading_zeros() == 0 => {
                panic!("First bit of the texture id is reserved for user texture flag!")
            }
            EguiTextureId::Managed(id) => Self(id),
            EguiTextureId::User(id) => Self(id | 1_u64.rotate_right(1)),
        }
    }
}
