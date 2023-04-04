use crate::engine::system::vulkan::egui::binding::Sdl2EguiMapping;
use crate::engine::system::vulkan::egui::painter::{PainterCreationError, UploadError};
use bytemuck::Pod;
use bytemuck::Zeroable;
use egui::{Context, RawInput};
use painter::EguiOnVulkanoPainter;
use sdl2::event::Event;
use std::sync::Arc;
use std::time::UNIX_EPOCH;
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::device::{Device, Queue};
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, BlendFactor, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreationError;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::Subpass;
use vulkano::sampler::{
    Filter, Sampler, SamplerCreateInfo, SamplerCreationError, SamplerMipmapMode,
};
use vulkano::shader::{EntryPoint, ShaderModule};

mod binding;
mod painter;

pub struct EguiSystem {
    painter: EguiOnVulkanoPainter,
    context: Context,
    binding: Sdl2EguiMapping,
    width: f32,
    height: f32,
}

impl EguiSystem {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        subpass: Subpass,
        width: f32,
        height: f32,
    ) -> Result<Self, PainterCreationError> {
        Ok(Self {
            painter: EguiOnVulkanoPainter::new(device, queue, subpass)?,
            context: Context::default(),
            binding: Sdl2EguiMapping::default(),
            width,
            height,
        })
    }

    #[inline]
    pub fn wants_input(&self) -> bool {
        self.context.wants_keyboard_input() || self.context.wants_pointer_input()
    }

    #[inline]
    pub fn on_sdl2_event(&mut self, event: &Event) {
        self.binding.on_sdl2_event(event)
    }

    #[inline]
    pub fn set_sdl2_view_area<I: Into<sdl2::rect::Rect>>(&mut self, area: I) {
        let area = area.into();
        self.width = area.width() as f32;
        self.height = area.height() as f32;
        self.binding.set_sdl2_view_area(area);
    }

    pub fn render<P>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<P>,
    ) -> Result<(), RenderError>
    where
        P: CommandBufferAllocator,
    {
        let input = self.binding.take_input();
        let output = self.context.run(input, |ctx| {
            //
        });

        self.painter
            .update_textures(output.textures_delta, builder)?;
        self.painter.draw(
            builder,
            [self.width, self.height],
            &self.context,
            output.shapes,
        )?;

        if let Some(cursor) = self
            .binding
            .cursor_icon_to_cursor(output.platform_output.cursor_icon)
        {
            cursor.set();
        }

        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum RenderError {
    #[error(transparent)]
    UploadError(#[from] UploadError),
}
