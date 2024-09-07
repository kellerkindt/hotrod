use crate::engine::{Engine, Error};
use crate::support::image::RawRgbaImage;
use std::borrow::Cow;
use vulkano::instance::InstanceCreateInfo;

pub struct EngineBuilder<'a> {
    pub(crate) window_icon: Option<RawRgbaImage>,
    pub(crate) window_title: Cow<'a, str>,
    pub(crate) window_width: u32,
    pub(crate) window_height: u32,
    pub(crate) instance_info: InstanceCreateInfo,
    #[cfg(feature = "ttf-sdl2")]
    pub(crate) font_renderer_ttf: Option<Cow<'static, [u8]>>,
}

impl EngineBuilder<'_> {
    /// Tries to set the specified image as the icon which is displayed for the new window.
    ///
    /// ### Wayland compatibility note
    ///
    /// On wayland the xdg-toplevel-icon-v1 protocol was just recently (August 2024) implemented.
    /// Depending on your SDL2 version, this might not be available for you yet, and your specified
    /// image will be ignored.
    ///
    /// See:
    ///  - https://discourse.libsdl.org/t/sdl-wayland-add-support-for-setting-window-icons-via-the-xdg-toplevel-icon-v1-protocol/53896
    ///  - https://github.com/libsdl-org/SDL/commit/5d5a685a8004fe8ceaf3dc5b3e9b431b32603e4b
    #[inline]
    pub fn with_window_icon(mut self, icon: impl Into<RawRgbaImage>) -> Self {
        self.window_icon = Some(icon.into());
        self
    }

    #[inline]
    pub fn with_ttf_font_renderer(
        mut self,
        font_renderer_ttf: impl Into<Cow<'static, [u8]>>,
    ) -> Self {
        self.font_renderer_ttf = Some(font_renderer_ttf.into());
        self
    }

    #[inline]
    pub fn build(self) -> Result<Engine, Error> {
        Engine::new(self)
    }
}

impl Default for EngineBuilder<'static> {
    #[inline]
    fn default() -> Self {
        Self {
            window_icon: None,
            window_title: Cow::Borrowed("HotRod Engine - Default Configuration"),
            window_width: 1024,
            window_height: 768,
            instance_info: InstanceCreateInfo::application_from_cargo_toml(),
            #[cfg(feature = "ttf-sdl2")]
            font_renderer_ttf: None,
        }
    }
}
