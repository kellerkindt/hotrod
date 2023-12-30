use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::sync::Arc;
use vulkano::buffer::{
    Buffer, BufferAllocateError, BufferCreateInfo, BufferUsage, IndexBuffer, Subbuffer,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferToImageInfo};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerCreateInfo, SamplerMipmapMode};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageAllocateError, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::cache::PipelineCache;
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition, VertexInputState};
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::RenderPass;
use vulkano::shader::EntryPoint;
use vulkano::{Validated, VulkanError};

use crate::engine::system::vulkan::system::{VulkanSystem, WriteDescriptorSetCollection};
use crate::engine::system::vulkan::textures::{TextureId, TextureInner};
use crate::engine::system::vulkan::utils::pipeline::{
    subpass_from_renderpass, write_descriptor_sets_from_collection,
};
use crate::engine::system::vulkan::{DrawError, PipelineCreateError, ShaderLoadError, UploadError};
use crate::engine::types::world2d::{Dim, Pos};
use crate::shader_from_path;

/// This pipeline is used to draw the terrain of 2d worlds. A 2d world terrain consists of quadratic
/// tiles. It supports additional features besides painting the terrain like:
///  - shading a terrain tile
#[derive()]
pub struct World2dTerrainPipeline {
    pipeline: Arc<GraphicsPipeline>,
    desc_allocator: StandardDescriptorSetAllocator,
    memo_allocator: StandardMemoryAllocator,
    quad_index_buffer: IndexBuffer,
    quad_vertex_buffer: Subbuffer<[Vertex2d]>,
    texture_sampler: Arc<Sampler>,
    origin_marker: Arc<()>,
    write_descriptors: WriteDescriptorSetCollection,
}

impl TryFrom<&VulkanSystem> for World2dTerrainPipeline {
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

impl World2dTerrainPipeline {
    pub fn new(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        cache: Option<Arc<PipelineCache>>,
        write_descriptors: WriteDescriptorSetCollection,
    ) -> Result<Self, PipelineCreateError> {
        let pipeline = Self::create_pipeline(Arc::clone(&device), render_pass, cache)?;
        let memo_allocator = StandardMemoryAllocator::new_default(Arc::clone(&device));
        Ok(Self {
            desc_allocator: StandardDescriptorSetAllocator::new(Arc::clone(&device)),
            texture_sampler: Self::create_texture_sampler(device)?,
            quad_index_buffer: Self::create_index_buffer(&memo_allocator, [0, 1, 2, 2, 3, 0])?
                .into(),
            quad_vertex_buffer: Self::create_vertex_buffer(
                &memo_allocator,
                vec![
                    Vertex2d { pos: [-0.5, -0.5] },
                    Vertex2d { pos: [0.5, -0.5] },
                    Vertex2d { pos: [0.5, 0.5] },
                    Vertex2d { pos: [-0.5, 0.5] },
                ],
            )?
            .into(),
            origin_marker: Arc::new(()),
            write_descriptors,
            pipeline,
            memo_allocator,
        })
    }

    fn create_pipeline(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        cache: Option<Arc<PipelineCache>>,
    ) -> Result<Arc<GraphicsPipeline>, PipelineCreateError> {
        let vs = Self::load_vertex_shader(Arc::clone(&device))?;
        let fs = Self::load_fragment_shader(Arc::clone(&device))?;

        let vertex_input_state = [Vertex2d::per_vertex(), InstanceData::per_instance()]
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
            "src/engine/system/vulkan/world2d/terrain/terrain.vert"
        )
    }

    fn load_fragment_shader(device: Arc<Device>) -> Result<EntryPoint, ShaderLoadError> {
        shader_from_path!(
            device,
            "fragment",
            "src/engine/system/vulkan/world2d/terrain/terrain.frag"
        )
    }

    fn create_texture_sampler(device: Arc<Device>) -> Result<Arc<Sampler>, Validated<VulkanError>> {
        Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                mipmap_mode: SamplerMipmapMode::Nearest,
                ..Default::default()
            },
        )
    }

    pub fn draw<P>(
        &self,
        builder: &mut AutoCommandBufferBuilder<P>,
        terrain: &World2dTerrain,
    ) -> Result<(), DrawError> {
        if Arc::ptr_eq(&self.origin_marker, &terrain.texture.0.origin) {
            let vertex_buffer = Self::create_vertex_buffer(
                &self.memo_allocator,
                terrain
                    .tiles
                    .iter()
                    .map(|(tile, shading)| InstanceData {
                        tile_pos: [tile.x, tile.y],
                        uv0: terrain.tile_uv[0].into(),
                        uv1: terrain.tile_uv[1].into(),
                        shading: *shading,
                    })
                    .collect::<Vec<_>>(),
            )?;

            builder
                .bind_pipeline_graphics(Arc::clone(&self.pipeline))?
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    Arc::clone(&self.pipeline.layout()),
                    0,
                    Arc::clone(&terrain.texture.0.descriptor),
                )?
                .bind_index_buffer(self.quad_index_buffer.clone())?
                .bind_vertex_buffers(
                    0,
                    [
                        self.quad_vertex_buffer.as_bytes().clone(),
                        vertex_buffer.as_bytes().clone(),
                    ],
                )?
                .draw_indexed(6, terrain.tiles.len() as u32, 0, 0, 0)?;

            Ok(())
        } else {
            todo!()
        }
    }

    fn create_vertex_buffer<I, T: Send + Sync + Pod>(
        memo_allocator: &impl MemoryAllocator,
        vertices: I,
    ) -> Result<Subbuffer<[T]>, Validated<BufferAllocateError>>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        Buffer::from_iter(
            memo_allocator,
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
        memo_allocator: &impl MemoryAllocator,
        indices: I,
    ) -> Result<Subbuffer<[u32]>, Validated<BufferAllocateError>>
    where
        I: IntoIterator<Item = u32>,
        I::IntoIter: ExactSizeIterator,
    {
        Buffer::from_iter(
            memo_allocator,
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
pub struct Vertex2d {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod, Vertex)]
pub struct InstanceData {
    #[format(R32G32_SFLOAT)]
    tile_pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    uv0: [f32; 2],
    #[format(R32G32_SFLOAT)]
    uv1: [f32; 2],
    #[format(R32_SFLOAT)]
    shading: f32,
}

pub struct World2dTerrain<'a> {
    pub texture: TextureId,
    pub tile_uv: [Dim<f32>; 2],
    pub tile_size: Dim<f32>,
    pub tiles: Cow<'a, [(Pos<f32>, f32)]>,
}
