use crate::ui::egui::ClippedPrimitive;
use binding::Sdl2EguiMapping;
use egui::{Context, CursorIcon, TexturesDelta};
use sdl2::event::Event;

mod binding;

#[derive(Default)]
pub struct EguiSystem {
    context: Context,
    binding: Sdl2EguiMapping,
    current_cursor: Option<CursorIcon>,
    pub(crate) width: f32,
    pub(crate) height: f32,
    /// [`TexturesDelta`] to upload next
    pub(crate) texture_delta: TexturesDelta,
    /// [`ClippedPrimitive`] to render next
    pub(crate) clipped_primitives: Vec<ClippedPrimitive>,
}

impl EguiSystem {
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

    /// Updates the [`Context`]. This updates the state for calls to [`EguiPipeline::prepare`] and
    /// [`EguiPipeline::draw`].
    pub fn update(&mut self, width: u32, height: u32, ui: impl FnOnce(&Context)) {
        self.set_sdl2_view_area(sdl2::rect::Rect::new(0, 0, width, height));

        let input = self.binding.take_input();
        let output = self.context.run(input, |ctx| {
            ui(&ctx);
        });

        if self
            .current_cursor
            .filter(|c| *c == output.platform_output.cursor_icon)
            .is_none()
        {
            self.current_cursor = Some(output.platform_output.cursor_icon);
            if let Some(cursor) = self
                .binding
                .cursor_icon_to_cursor(output.platform_output.cursor_icon)
            {
                cursor.set();
            }
        }

        self.texture_delta = output.textures_delta;
        self.clipped_primitives = self
            .context
            .tessellate(output.shapes, output.pixels_per_point);
    }
}
