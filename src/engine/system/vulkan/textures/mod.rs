use crate::engine::system::vulkan::VulkanSystem;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferAllocateInfo, BufferError, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CopyBufferToImageInfo, CopyError, PipelineExecutionError,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{
    DescriptorSetCreationError, PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::{Device, Features, Queue};
use vulkano::format::Format;
use vulkano::image::view::{ImageView, ImageViewCreationError};
use vulkano::image::{ImageCreateFlags, ImageDimensions, ImageError, ImageUsage, StorageImage};
use vulkano::memory::allocator::{MemoryUsage, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::render_pass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreationError;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint, StateMode};
use vulkano::sampler::{
    Filter, Sampler, SamplerCreateInfo, SamplerCreationError, SamplerMipmapMode,
};
use vulkano::shader::ShaderModule;

#[derive()]
pub struct VulkanTextureSystem {
    queue: Arc<Queue>,
    pipeline: Arc<GraphicsPipeline>,
    desc_allocator: StandardDescriptorSetAllocator,
    memo_allocator: StandardMemoryAllocator,
    texture_id_gen: usize,
    texture_sampler: Arc<Sampler>,
    textures: HashMap<TextureId, Arc<PersistentDescriptorSet>>,
    textures_to_free: Vec<TextureId>,
    images: HashMap<TextureId, Arc<StorageImage>>,
}

impl TryFrom<&VulkanSystem> for VulkanTextureSystem {
    type Error = CreationError;

    fn try_from(vs: &VulkanSystem) -> Result<Self, Self::Error> {
        Self::new(
            Arc::clone(&vs.device()),
            Arc::clone(&vs.queue()),
            vs.image_format(),
        )
    }
}

impl VulkanTextureSystem {
    pub const REQUIRED_FEATURES: Features = Features {
        dynamic_rendering: true,
        ..Features::empty()
    };

    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        image_format: Format,
    ) -> Result<Self, CreationError> {
        Ok(Self {
            queue,
            pipeline: Self::create_pipeline(Arc::clone(&device), image_format)?,
            desc_allocator: StandardDescriptorSetAllocator::new(Arc::clone(&device)),
            memo_allocator: StandardMemoryAllocator::new_default(Arc::clone(&device)),
            texture_id_gen: 0,
            texture_sampler: Self::create_texture_sampler(device)?,
            textures: HashMap::default(),
            textures_to_free: Vec::default(),
            images: Default::default(),
        })
    }

    fn create_pipeline(
        device: Arc<Device>,
        image_format: Format,
    ) -> Result<Arc<GraphicsPipeline>, GraphicsPipelineCreationError> {
        GraphicsPipeline::start()
            .vertex_input_state(Vertex2dUv::per_vertex())
            .input_assembly_state(InputAssemblyState::new())
            .vertex_shader(
                Self::load_vertex_shader(Arc::clone(&device))
                    .entry_point("main")
                    .unwrap(),
                (),
            )
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(
                Self::load_fragment_shader(Arc::clone(&device))
                    .entry_point("main")
                    .unwrap(),
                (),
            )
            .rasterization_state(RasterizationState {
                cull_mode: StateMode::Fixed(CullMode::None),
                // line_width: StateMode::Dynamic,
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
                path: "src/engine/system/vulkan/textures/textures.vert"
            );
        }
        shader::load(device).unwrap()
    }

    fn load_fragment_shader(device: Arc<Device>) -> Arc<ShaderModule> {
        mod shader {
            vulkano_shaders::shader!(
                ty: "fragment",
                path: "src/engine/system/vulkan/textures/textures.frag"
            );
        }
        shader::load(device).unwrap()
    }

    fn create_texture_sampler(device: Arc<Device>) -> Result<Arc<Sampler>, SamplerCreationError> {
        Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Linear,
                ..Default::default()
            },
        )
    }

    pub fn draw<P>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<P>,
        width: f32,
        height: f32,
        textured: &[Textured],
    ) -> Result<(), DrawError> {
        builder.bind_pipeline_graphics(Arc::clone(&self.pipeline));

        let mut offset = 0;
        let vertex_buffer = self.create_vertex_buffer(
            textured
                .iter()
                .flat_map(|l| l.vertices.iter().copied())
                .collect::<Vec<_>>(),
        )?;

        for textured in textured {
            let vertices = vertex_buffer
                .clone()
                .slice(offset..(offset + textured.vertices.len() as u64));

            offset += textured.vertices.len() as u64;

            if let Some(texture) = self.textures.get(&textured.texture) {
                builder
                    .bind_vertex_buffers(0, vertices)
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        Arc::clone(&self.pipeline.layout()),
                        0,
                        Arc::clone(texture),
                    )
                    .push_constants(Arc::clone(&self.pipeline.layout()), 0, [width, height])
                    .draw(textured.vertices.len() as u32, 1, 0, 0)?;
            }
        }

        self.free_textures();
        Ok(())
    }

    fn free_textures(&mut self) {
        for texture in self.textures_to_free.drain(..) {
            self.textures.remove(&texture);
            self.images.remove(&texture);
        }
    }

    fn create_vertex_buffer(
        &self,
        vertices: Vec<Vertex2dUv>,
    ) -> Result<Subbuffer<[Vertex2dUv]>, BufferError> {
        Buffer::from_iter(
            &self.memo_allocator,
            BufferAllocateInfo {
                buffer_usage: BufferUsage::VERTEX_BUFFER,
                ..BufferAllocateInfo::default()
            },
            vertices,
        )
    }

    pub fn create_texture<P>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<P>,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    ) -> Result<TextureId, UploadError> {
        let (image, desc) = self.create_image_desc(width, height)?;
        self.upload_image(builder, Arc::clone(&image), rgba)?;

        let texture_id = self.next_texture_id();
        self.textures.insert(texture_id, Arc::clone(&desc));
        self.images.insert(texture_id, image);

        Ok(texture_id)
    }

    fn next_texture_id(&mut self) -> TextureId {
        let texture_id = TextureId(self.texture_id_gen);
        self.texture_id_gen += 1;
        texture_id
    }

    fn create_image_desc(
        &self,
        width: u32,
        height: u32,
    ) -> Result<(Arc<StorageImage>, Arc<PersistentDescriptorSet>), UploadError> {
        let image = self.create_image(self.queue.as_ref(), width, height)?;
        let layout = &self.pipeline.layout().set_layouts()[0];

        match PersistentDescriptorSet::new(
            &self.desc_allocator,
            Arc::clone(&layout),
            [WriteDescriptorSet::image_view_sampler(
                0,
                ImageView::new_default(Arc::clone(&image))?,
                Arc::clone(&self.texture_sampler),
            )],
        ) {
            Ok(desc) => Ok((image, desc)),
            Err(e) => Err(e.into()),
        }
    }

    fn create_image(
        &self,
        queue: &Queue,
        width: u32,
        height: u32,
    ) -> Result<Arc<StorageImage>, ImageError> {
        let dimensions = ImageDimensions::Dim2d {
            width,
            height,
            array_layers: 1,
        };

        StorageImage::with_usage(
            &self.memo_allocator,
            dimensions,
            Format::R8G8B8A8_SRGB,
            ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ImageCreateFlags::empty(),
            [queue.queue_family_index()],
        )
    }

    fn upload_image<P>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<P>,
        image: Arc<StorageImage>,
        rgba: Vec<u8>,
    ) -> Result<(), UploadError> {
        builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            Buffer::from_iter(
                &self.memo_allocator,
                BufferAllocateInfo {
                    buffer_usage: BufferUsage::TRANSFER_SRC,
                    memory_usage: MemoryUsage::Upload,
                    ..BufferAllocateInfo::default()
                },
                rgba.to_vec(),
            )?,
            image,
        ))?;
        Ok(())
    }

    pub fn destroy_texture(&mut self, texture: TextureId) {
        self.textures_to_free.push(texture);
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
    pub texture: TextureId,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct TextureId(usize);

#[derive(thiserror::Error, Debug)]
pub enum DrawError {
    #[error("Failed to load buffer: {0}")]
    BufferError(#[from] BufferError),
    #[error("Failed to execute the pipeline: {0}")]
    PipelineExecutionError(#[from] PipelineExecutionError),
}

#[derive(thiserror::Error, Debug)]
pub enum UploadError {
    #[error("Failed to upload the image: {0}")]
    ImageError(#[from] ImageError),
    #[error("Failed to create the image view: {0}")]
    ImageViewCreationError(#[from] ImageViewCreationError),
    #[error("Failed to create the descriptor set: {0}")]
    DescriptorSetCreationError(#[from] DescriptorSetCreationError),
    #[error("Failed to create the buffer: {0}")]
    BufferError(#[from] BufferError),
    #[error("Failed to copy to the buffer: {0}")]
    CopyError(#[from] CopyError),
}

#[derive(thiserror::Error, Debug)]
pub enum CreationError {
    #[error("Failed to create graphics pipeline: {0}")]
    GraphicsPipelineCreationError(#[from] GraphicsPipelineCreationError),
    #[error("Failed to create texture sampler: {0}")]
    SamplerCreationError(#[from] SamplerCreationError),
}
