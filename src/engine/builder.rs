use crate::engine::{Engine, Error};
use crate::support::image::RawRgbaImage;
use std::borrow::Cow;
use vulkano::image::SampleCount;
use vulkano::instance::InstanceCreateInfo;

pub struct EngineBuilder<'a> {
    pub(crate) window_icon: Option<RawRgbaImage>,
    pub(crate) window_title: Cow<'a, str>,
    pub(crate) window_width: u32,
    pub(crate) window_height: u32,
    pub(crate) fullscreen: bool,
    pub(crate) instance_info: InstanceCreateInfo,
    pub(crate) target_frame_rate: u16,
    pub(crate) background_clear_color: Option<[f32; 4]>,
    #[cfg(feature = "ttf-sdl2")]
    pub(crate) font_renderer_ttf: Option<Cow<'static, [u8]>>,
    pub(crate) msaa: Option<SampleCount>,
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
    pub fn with_window_title(mut self, title: impl Into<Cow<'static, str>>) -> Self {
        self.window_title = title.into();
        self
    }

    #[inline]
    pub fn with_window_width(mut self, width: u32) -> Self {
        self.window_width = width;
        self
    }

    #[inline]
    pub fn with_window_height(mut self, height: u32) -> Self {
        self.window_height = height;
        self
    }

    #[inline]
    pub fn with_fullscreen(mut self, fullscreen: bool) -> Self {
        self.fullscreen = fullscreen;
        self
    }

    #[inline]
    pub fn with_target_frame_rate(mut self, target_frame_rate: u16) -> Self {
        self.target_frame_rate = target_frame_rate;
        self
    }

    #[inline]
    pub fn with_background_clear_color(mut self, color: [f32; 4]) -> Self {
        self.background_clear_color = Some(color);
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
    pub fn with_msaa(mut self, msaa: SampleCount) -> Self {
        self.msaa = Some(msaa);
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
            fullscreen: false,
            instance_info: InstanceCreateInfo::application_from_cargo_toml(),
            target_frame_rate: 60,
            background_clear_color: None,
            #[cfg(feature = "ttf-sdl2")]
            font_renderer_ttf: None,
            msaa: None,
        }
    }
}
