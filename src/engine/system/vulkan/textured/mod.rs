use crate::engine::system::vulkan::system::{VulkanSystem, WriteDescriptorSetCollection};
use crate::engine::system::vulkan::utils::pipeline::{
    subpass_from_renderpass, write_descriptor_sets_from_collection,
};
use crate::engine::system::vulkan::{DrawError, PipelineCreateError, ShaderLoadError, UploadError};
use crate::shader_from_path;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferAllocateError, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferToImageInfo};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Features};
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerCreateInfo, SamplerMipmapMode};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageAllocateError, ImageCreateInfo, ImageType, ImageUsage};
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
pub struct TexturesPipeline {
    pipeline: Arc<GraphicsPipeline>,
    desc_allocator: StandardDescriptorSetAllocator,
    memo_allocator: StandardMemoryAllocator,
    texture_sampler: Arc<Sampler>,
    origin_marker: Arc<()>,
    write_descriptors: WriteDescriptorSetCollection,
}

impl TryFrom<&VulkanSystem> for TexturesPipeline {
    type Error = PipelineCreateError;

    fn try_from(vs: &VulkanSystem) -> Result<Self, Self::Error> {
        Self::new(
            Arc::clone(vs.device()),
            Arc::clone(vs.render_pass()),
            vs.pipeline_cache().map(Arc::clone),
            vs.get_write_descriptor_sets().clone(),
        )
    }
}

impl TexturesPipeline {
    pub const REQUIRED_FEATURES: Features = Features {
        dynamic_rendering: true,
        ..Features::empty()
    };

    pub fn new(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        cache: Option<Arc<PipelineCache>>,
        write_descriptors: WriteDescriptorSetCollection,
    ) -> Result<Self, PipelineCreateError> {
        Ok(Self {
            pipeline: Self::create_pipeline(Arc::clone(&device), render_pass, cache)?,
            desc_allocator: StandardDescriptorSetAllocator::new(Arc::clone(&device)),
            memo_allocator: StandardMemoryAllocator::new_default(Arc::clone(&device)),
            texture_sampler: Self::create_texture_sampler(device)?,
            origin_marker: Arc::new(()),
            write_descriptors,
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

    fn create_texture_sampler(device: Arc<Device>) -> Result<Arc<Sampler>, Validated<VulkanError>> {
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
            if Arc::ptr_eq(&self.origin_marker, &textured.texture.0.origin) {
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

            if Arc::ptr_eq(&self.origin_marker, &textured.texture.0.origin) {
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

    pub fn create_texture<P>(
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    ) -> Result<TextureId, UploadError> {
        let (image, descriptor) = self.create_image_desc(width, height)?;
        self.upload_image(builder, Arc::clone(&image), rgba)?;
        Ok(TextureId(Arc::new(TextureInner {
            origin: Arc::clone(&self.origin_marker),
            _image: image,
            descriptor,
        })))
    }

    fn create_image_desc(
        &self,
        width: u32,
        height: u32,
    ) -> Result<(Arc<Image>, Arc<PersistentDescriptorSet>), UploadError> {
        let image = self.create_image(width, height)?;
        let layout = &self.pipeline.layout().set_layouts()[0];

        match PersistentDescriptorSet::new(
            &self.desc_allocator,
            Arc::clone(&layout),
            [WriteDescriptorSet::image_view_sampler(
                0,
                ImageView::new_default(Arc::clone(&image))?,
                Arc::clone(&self.texture_sampler),
            )]
            .into_iter()
            .chain(write_descriptor_sets_from_collection(
                layout,
                &self.write_descriptors,
            )),
            [],
        ) {
            Ok(desc) => Ok((image, desc)),
            Err(e) => Err(e.into()),
        }
    }

    fn create_image(
        &self,
        width: u32,
        height: u32,
    ) -> Result<Arc<Image>, Validated<ImageAllocateError>> {
        Image::new(
            &self.memo_allocator,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                extent: [width, height, 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..ImageCreateInfo::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..AllocationCreateInfo::default()
            },
        )
    }

    fn upload_image<P, I>(
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        image: Arc<Image>,
        rgba: I,
    ) -> Result<(), Validated<BufferAllocateError>>
    where
        I: IntoIterator<Item = u8>,
        I::IntoIter: ExactSizeIterator,
    {
        builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            Buffer::from_iter(
                &self.memo_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..BufferCreateInfo::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..AllocationCreateInfo::default()
                },
                rgba,
            )?,
            image,
        ))?;
        Ok(())
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

pub struct TexturedIndexed {
    pub vertices: Vec<Vertex2dUv>,
    pub indices: Vec<[u32; 3]>,
    pub texture: TextureId,
}

#[derive(Clone)]
pub struct TextureId(pub Arc<TextureInner>);

pub struct TextureInner {
    pub origin: Arc<()>,
    pub _image: Arc<Image>,
    pub descriptor: Arc<PersistentDescriptorSet>,
}
