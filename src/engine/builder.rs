use std::borrow::Cow;
use vulkano::instance::InstanceCreateInfo;
use crate::engine::{Engine, Error};

pub struct EngineBuilder<'a> {
    pub(crate) window_title: Cow<'a, str>,
    pub(crate) window_width: u32,
    pub(crate) window_height: u32,
    pub(crate) instance_info: InstanceCreateInfo,
    #[cfg(feature = "ttf-sdl2")]
    pub(crate) font_renderer_ttf: Option<Cow<'static, [u8]>>,
}

impl EngineBuilder<'_> {
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
            window_title: Cow::Borrowed("HotRod Engine - Default Configuration"),
            window_width: 1024,
            window_height: 768,
            instance_info: InstanceCreateInfo::application_from_cargo_toml(),
            #[cfg(feature = "ttf-sdl2")]
            font_renderer_ttf: None,
        }
    }
}
