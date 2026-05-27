use crate::engine::parts::sdl::SdlParts;
use crate::ui::egui::ClippedPrimitive;
use binding::Sdl2EguiMapping;
use egui::{Context, CursorIcon, Key, OutputCommand, RawInput, TexturesDelta, Ui};
use sdl2::clipboard::ClipboardUtil;
use sdl2::event::Event;

mod binding;
pub mod extensions;
pub mod styling;

#[derive(Default)]
pub struct EguiSystem {
    context: Context,
    binding: Sdl2EguiMapping,
    current_cursor: Option<CursorIcon>,
    /// [`TexturesDelta`] to upload next
    pub(crate) texture_delta: TexturesDelta,
    /// [`ClippedPrimitive`] to render next
    pub(crate) clipped_primitives: Vec<ClippedPrimitive>,
}

impl EguiSystem {
    #[inline]
    pub fn wants_input(&self) -> bool {
        self.context.egui_wants_keyboard_input()
            || self.context.egui_wants_pointer_input()
            || self.context.egui_is_using_pointer()
    }

    #[inline]
    pub fn on_sdl2_event(&mut self, event: &Event) {
        self.binding.on_sdl2_event(event)
    }

    #[inline]
    pub fn set_target_frame_rate(&mut self, fps: u16) {
        self.binding.set_target_frame_rate(fps)
    }

    #[inline]
    pub fn set_fullscreen(&mut self, fullscreen: bool) {
        self.binding.set_fullscreen(fullscreen)
    }

    #[inline]
    pub fn pixels_view_area(&self) -> sdl2::rect::Rect {
        self.binding.pixels_view_area()
    }

    #[inline]
    pub fn points_view_area(&self) -> egui::emath::Rect {
        self.binding.points_view_area()
    }

    #[inline]
    pub fn pixels_per_point(&self) -> f32 {
        self.binding.pixels_per_point()
    }

    #[inline]
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Updates the [`Context`]. This updates the state for calls to [`EguiPipeline::prepare`] and
    /// [`EguiPipeline::draw`].
    pub fn update(
        &mut self,
        width: u32,
        height: u32,
        sdl: &mut SdlParts,
        ui: impl FnOnce(&mut Ui),
    ) {
        self.binding
            .set_view_area(sdl2::rect::Rect::new(0, 0, width, height));

        let input = RawInputShim(self.binding.take_input())
            .with_injected_shortcuts(|| sdl.video_subsystem.clipboard());

        let output = self.context.run_ui(input, {
            let mut ui = Some(ui);
            move |ctx| {
                if let Some(ui) = ui.take() {
                    ui(ctx);
                }
            }
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

        for command in &output.platform_output.commands {
            match command {
                OutputCommand::CopyText(text) => {
                    if let Err(e) = sdl.video_subsystem.clipboard().set_clipboard_text(text) {
                        error!("Failed to update clipboard text: {e}");
                    }
                }
                OutputCommand::CopyImage(_) => {}
                OutputCommand::OpenUrl(_) => {}
            }
        }

        self.texture_delta = output.textures_delta;
        self.clipped_primitives = self
            .context
            .tessellate(output.shapes, output.pixels_per_point);

        self.binding.set_pixels_per_point(output.pixels_per_point);
    }
}

struct RawInputShim(RawInput);

impl RawInputShim {
    #[inline]
    pub fn with_injected_shortcuts(self, clipboard: impl FnOnce() -> ClipboardUtil) -> RawInput {
        self.inject_shortcuts(clipboard).0
    }

    pub fn inject_shortcuts(mut self, clipboard: impl FnOnce() -> ClipboardUtil) -> Self {
        if self.0.modifiers.command {
            if self.is_key_pressed(Key::C) {
                self.0.events.push(egui::Event::Copy)
            } else if self.is_key_pressed(Key::X) {
                self.0.events.push(egui::Event::Cut)
            } else if self.is_key_pressed(Key::V) {
                match clipboard().clipboard_text() {
                    Ok(text) => self.0.events.push(egui::Event::Paste(text)),
                    Err(e) => error!("Failed to read clipboard: {e}"),
                }
            }
        }
        self
    }

    fn is_key_pressed(&self, cmd_and_key: Key) -> bool {
        self.0.events.iter().any(|k| {
            if let egui::Event::Key { key, pressed, .. } = k {
                *pressed && *key == cmd_and_key
            } else {
                false
            }
        })
    }
}
