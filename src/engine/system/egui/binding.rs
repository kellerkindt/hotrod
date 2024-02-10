use egui::{
    CursorIcon, DroppedFile, HoveredFile, Key, PointerButton, Pos2, RawInput, Rect, Vec2,
    ViewportEvent, ViewportId, ViewportInfo,
};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::mouse::Cursor;
use sdl2::mouse::SystemCursor;
use std::path::PathBuf;
use std::time::UNIX_EPOCH;

pub(crate) struct Sdl2EguiMapping {
    input: RawInput,
}

impl Default for Sdl2EguiMapping {
    fn default() -> Self {
        Self {
            input: RawInput {
                viewport_id: ViewportId::ROOT,
                viewports: [(ViewportId::ROOT, ViewportInfo::default())]
                    .into_iter()
                    .collect(),
                ..Default::default()
            },
        }
    }
}

impl Sdl2EguiMapping {
    pub fn take_input(&mut self) -> RawInput {
        RawInput {
            viewport_id: self.input.viewport_id,
            viewports: self.input.viewports.clone(),
            screen_rect: self.input.screen_rect,
            max_texture_side: self.input.max_texture_side,
            time: Some(UNIX_EPOCH.elapsed().unwrap_or_default().as_secs_f64()),
            predicted_dt: self.input.predicted_dt,
            modifiers: core::mem::take(&mut self.input.modifiers),
            events: core::mem::take(&mut self.input.events),
            hovered_files: core::mem::take(&mut self.input.hovered_files),
            dropped_files: core::mem::take(&mut self.input.dropped_files),
            focused: self.input.focused,
        }
    }

    fn cursor_icon_to_system_cursor(icon: CursorIcon) -> SystemCursor {
        match icon {
            CursorIcon::Default => SystemCursor::Arrow,
            CursorIcon::None => SystemCursor::Arrow,
            CursorIcon::ContextMenu => SystemCursor::Arrow,
            CursorIcon::Help => SystemCursor::Arrow,
            CursorIcon::PointingHand => SystemCursor::Hand,
            CursorIcon::Progress => SystemCursor::WaitArrow,
            CursorIcon::Wait => SystemCursor::WaitArrow,
            CursorIcon::Cell => SystemCursor::Arrow,
            CursorIcon::Crosshair => SystemCursor::Crosshair,
            CursorIcon::Text => SystemCursor::IBeam,
            CursorIcon::VerticalText => SystemCursor::IBeam,
            CursorIcon::Alias => SystemCursor::Arrow,
            CursorIcon::Copy => SystemCursor::Hand,
            CursorIcon::Move => SystemCursor::Hand,
            CursorIcon::NoDrop => SystemCursor::No,
            CursorIcon::NotAllowed => SystemCursor::No,
            CursorIcon::Grab => SystemCursor::Hand,
            CursorIcon::Grabbing => SystemCursor::Hand,
            CursorIcon::AllScroll => SystemCursor::SizeAll,
            CursorIcon::ResizeHorizontal => SystemCursor::SizeWE,
            CursorIcon::ResizeNeSw => SystemCursor::SizeNESW,
            CursorIcon::ResizeNwSe => SystemCursor::SizeNWSE,
            CursorIcon::ResizeVertical => SystemCursor::SizeNS,
            CursorIcon::ZoomIn => SystemCursor::SizeNS,
            CursorIcon::ZoomOut => SystemCursor::SizeNS,
            CursorIcon::ResizeEast => SystemCursor::SizeWE,
            CursorIcon::ResizeSouthEast => SystemCursor::SizeAll,
            CursorIcon::ResizeSouth => SystemCursor::SizeNS,
            CursorIcon::ResizeSouthWest => SystemCursor::SizeAll,
            CursorIcon::ResizeWest => SystemCursor::SizeWE,
            CursorIcon::ResizeNorthWest => SystemCursor::SizeAll,
            CursorIcon::ResizeNorth => SystemCursor::SizeNS,
            CursorIcon::ResizeNorthEast => SystemCursor::SizeAll,
            CursorIcon::ResizeColumn => SystemCursor::SizeWE,
            CursorIcon::ResizeRow => SystemCursor::SizeNS,
        }
    }

    #[inline]
    pub fn cursor_icon_to_cursor(&self, cursor_icon: CursorIcon) -> Option<Cursor> {
        Cursor::from_system(Self::cursor_icon_to_system_cursor(cursor_icon)).ok()
    }

    pub fn set_sdl2_view_area<I: Into<sdl2::rect::Rect>>(&mut self, area: I) {
        let area = area.into();
        let x = area.x() as f32;
        let y = area.y() as f32;
        let w = area.width() as f32;
        let h = area.height() as f32;
        self.input.screen_rect = Some(Rect {
            min: Pos2::new(x, y),
            max: Pos2::new(x + w, y + h),
        });
    }

    pub fn set_target_frame_rate(&mut self, fps: u32) {
        self.input.predicted_dt = 1.0_f32 / fps as f32
    }

    pub fn set_fullscreen(&mut self, fullscreen: bool) {
        self.on_current_viewport_mut(|viewport| viewport.fullscreen = Some(fullscreen));
    }

    pub fn on_sdl2_event(&mut self, event: &Event) {
        match event {
            Event::KeyDown { keycode, .. } | Event::KeyUp { keycode, .. } => {
                let pressed = matches!(event, Event::KeyDown { .. });
                let key = match keycode {
                    Some(Keycode::Down) => Key::ArrowDown,
                    Some(Keycode::Left) => Key::ArrowLeft,
                    Some(Keycode::Right) => Key::ArrowRight,
                    Some(Keycode::Up) => Key::ArrowUp,

                    Some(Keycode::Escape) => Key::Escape,
                    Some(Keycode::Tab) => Key::Tab,
                    Some(Keycode::Backspace) => Key::Backspace,
                    Some(Keycode::Return) => Key::Enter,
                    Some(Keycode::Space) => Key::Space,

                    Some(Keycode::Insert) => Key::Insert,
                    Some(Keycode::Delete) => Key::Delete,
                    Some(Keycode::Home) => Key::Home,
                    Some(Keycode::End) => Key::End,
                    Some(Keycode::PageUp) => Key::PageUp,
                    Some(Keycode::PageDown) => Key::PageDown,

                    Some(Keycode::Num0) => Key::Num0,
                    Some(Keycode::Num1) => Key::Num1,
                    Some(Keycode::Num2) => Key::Num2,
                    Some(Keycode::Num3) => Key::Num3,
                    Some(Keycode::Num4) => Key::Num4,
                    Some(Keycode::Num5) => Key::Num5,
                    Some(Keycode::Num6) => Key::Num6,
                    Some(Keycode::Num7) => Key::Num7,
                    Some(Keycode::Num8) => Key::Num8,
                    Some(Keycode::Num9) => Key::Num9,

                    Some(Keycode::A) => Key::A, // Used for cmd+A (select All)
                    Some(Keycode::B) => Key::B,
                    Some(Keycode::C) => Key::C,
                    Some(Keycode::D) => Key::D,
                    Some(Keycode::E) => Key::E,
                    Some(Keycode::F) => Key::F,
                    Some(Keycode::G) => Key::G,
                    Some(Keycode::H) => Key::H,
                    Some(Keycode::I) => Key::I,
                    Some(Keycode::J) => Key::J,
                    Some(Keycode::K) => Key::K, // Used for ctrl+K (delete text after cursor)
                    Some(Keycode::L) => Key::L,
                    Some(Keycode::M) => Key::M,
                    Some(Keycode::N) => Key::N,
                    Some(Keycode::O) => Key::O,
                    Some(Keycode::P) => Key::P,
                    Some(Keycode::Q) => Key::Q,
                    Some(Keycode::R) => Key::R,
                    Some(Keycode::S) => Key::S,
                    Some(Keycode::T) => Key::T,
                    Some(Keycode::U) => Key::U, // Used for ctrl+U (delete text before cursor)
                    Some(Keycode::V) => Key::V,
                    Some(Keycode::W) => Key::W, // Used for ctrl+W (delete previous word)
                    Some(Keycode::X) => Key::X,
                    Some(Keycode::Y) => Key::Y,
                    Some(Keycode::Z) => Key::Z, // Used for cmd+Z (undo)

                    Some(Keycode::F1) => Key::F1,
                    Some(Keycode::F2) => Key::F2,
                    Some(Keycode::F3) => Key::F3,
                    Some(Keycode::F4) => Key::F4,
                    Some(Keycode::F5) => Key::F5,
                    Some(Keycode::F6) => Key::F6,
                    Some(Keycode::F7) => Key::F7,
                    Some(Keycode::F8) => Key::F8,
                    Some(Keycode::F9) => Key::F9,
                    Some(Keycode::F10) => Key::F10,
                    Some(Keycode::F11) => Key::F11,
                    Some(Keycode::F12) => Key::F12,
                    Some(Keycode::F13) => Key::F13,
                    Some(Keycode::F14) => Key::F14,
                    Some(Keycode::F15) => Key::F15,
                    Some(Keycode::F16) => Key::F16,
                    Some(Keycode::F17) => Key::F17,
                    Some(Keycode::F18) => Key::F18,
                    Some(Keycode::F19) => Key::F19,
                    Some(Keycode::F20) => Key::F20,
                    // Some(Keycode::F21) => Key::F21,
                    // Some(Keycode::F22) => Key::F22,
                    // Some(Keycode::F23) => Key::F23,
                    // Some(Keycode::F24) => Key::F24,
                    Some(Keycode::LAlt) | Some(Keycode::RAlt) => {
                        self.input.modifiers.alt = pressed;
                        return;
                    }
                    Some(Keycode::LCtrl) | Some(Keycode::RCtrl) => {
                        self.input.modifiers.ctrl = pressed;
                        self.input.modifiers.command = pressed;
                        return;
                    }
                    Some(Keycode::LShift) | Some(Keycode::RShift) => {
                        self.input.modifiers.shift = pressed;
                        return;
                    }

                    _ => return,
                };
                self.input.events.push(egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed,
                    modifiers: self.input.modifiers,
                    repeat: false,
                });
            }
            Event::TextInput { text, .. } => {
                self.input.events.push(egui::Event::Text(text.clone()));
            }
            Event::MouseMotion { x, y, .. } => self
                .input
                .events
                .push(egui::Event::PointerMoved(Pos2::new(*x as f32, *y as f32))),
            Event::MouseButtonDown {
                x, y, mouse_btn, ..
            }
            | Event::MouseButtonUp {
                x, y, mouse_btn, ..
            } => {
                let button = match mouse_btn {
                    sdl2::mouse::MouseButton::Left => PointerButton::Primary,
                    sdl2::mouse::MouseButton::Middle => PointerButton::Middle,
                    sdl2::mouse::MouseButton::Right => PointerButton::Secondary,
                    _ => return,
                };
                self.input.events.push(egui::Event::PointerButton {
                    pos: Pos2::new(*x as f32, *y as f32),
                    button,
                    pressed: matches!(event, Event::MouseButtonDown { .. }),
                    modifiers: self.input.modifiers,
                });
            }
            Event::MouseWheel { x, y, .. } => self
                .input
                .events
                .push(egui::Event::Scroll(Vec2::new(*x as f32, *y as f32))),
            Event::DropFile { filename, .. } => {
                self.input.hovered_files.push(HoveredFile {
                    path: Some(PathBuf::from(filename)),
                    mime: String::new(),
                });
            }
            Event::DropComplete { .. } => {
                let files = core::mem::take(&mut self.input.hovered_files);
                self.input.dropped_files = files
                    .into_iter()
                    .map(|h| DroppedFile {
                        path: h.path,
                        name: String::new(),
                        mime: String::new(),
                        last_modified: None,
                        bytes: None,
                    })
                    .collect();
            }
            Event::Quit { .. } => {
                self.on_current_viewport_mut(|viewport| viewport.events.push(ViewportEvent::Close));
            }
            Event::Window { win_event, .. } => match win_event {
                WindowEvent::Minimized => {
                    self.on_current_viewport_mut(|v| {
                        v.minimized = Some(true);
                        v.maximized = Some(false);
                    });
                }
                WindowEvent::Maximized => {
                    self.on_current_viewport_mut(|v| {
                        v.minimized = Some(false);
                        v.maximized = Some(true);
                    });
                }
                WindowEvent::FocusGained => {
                    self.on_current_viewport_mut(|viewport| viewport.focused = Some(true));
                }
                WindowEvent::FocusLost => {
                    self.on_current_viewport_mut(|viewport| viewport.focused = Some(false));
                }
                WindowEvent::Close => {
                    self.on_current_viewport_mut(|viewport| {
                        viewport.events.push(ViewportEvent::Close)
                    });
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn on_current_viewport_mut(&mut self, f: impl FnOnce(&mut ViewportInfo)) {
        let viewport_id = self.input.viewport_id;
        if let Some(viewport) = self.input.viewports.get_mut(&viewport_id) {
            f(viewport);
        }
    }
}
