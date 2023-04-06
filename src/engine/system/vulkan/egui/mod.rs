use crate::engine::system::vulkan::egui::binding::Sdl2EguiMapping;
use crate::engine::system::vulkan::egui::painter::{DrawError, PainterCreationError, UploadError};
use crate::ui::egui::ClippedPrimitive;
use egui::{Context, TexturesDelta};
use painter::EguiOnVulkanoPainter;
use sdl2::event::Event;
use std::sync::Arc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::device::{Device, Queue};
use vulkano::format::Format;

mod binding;
mod painter;

pub struct EguiSystem {
    painter: EguiOnVulkanoPainter,
    context: Context,
    binding: Sdl2EguiMapping,
    width: f32,
    height: f32,
    /// [`TexturesDelta`] to upload next
    texture_delta: TexturesDelta,
    /// [`ClippedPrimitive`] to render next
    clipped_primitives: Vec<ClippedPrimitive>,
}

impl EguiSystem {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        image_format: Format,
        width: f32,
        height: f32,
    ) -> Result<Self, PainterCreationError> {
        Ok(Self {
            painter: EguiOnVulkanoPainter::new(device, queue, image_format)?,
            context: Context::default(),
            binding: Sdl2EguiMapping::default(),
            width,
            height,
            texture_delta: TexturesDelta::default(),
            clipped_primitives: Vec::default(),
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

    /// Updates the [`Context`]. This updates the state for calls to [`Self::prepare_render`] and
    /// [`Self::render`].
    pub fn update_egui(&mut self, ui: impl FnOnce(&Context)) {
        let input = self.binding.take_input();
        let output = self.context.run(input, |ctx| {
            ui(&ctx);
        });

        if let Some(cursor) = self
            .binding
            .cursor_icon_to_cursor(output.platform_output.cursor_icon)
        {
            cursor.set();
        }

        self.texture_delta = output.textures_delta;
        self.clipped_primitives = self.context.tessellate(output.shapes);
    }

    #[inline]
    pub fn prepare_render<P>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<P>,
    ) -> Result<(), UploadError> {
        self.painter.update_textures(&self.texture_delta, builder)
    }

    #[inline]
    pub fn render<P>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<P>,
    ) -> Result<(), DrawError> {
        self.painter
            .draw(builder, self.width, self.height, &self.clipped_primitives)
    }
}
