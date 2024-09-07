use crate::support::image::RawRgbaImage;
use sdl2::pixels::PixelFormatEnum;
use sdl2::surface::Surface;
use sdl2::ttf::Sdl2TtfContext;
use sdl2::video::Window;
use sdl2::{EventPump, Sdl, VideoSubsystem};

pub struct SdlParts {
    pub video_subsystem: VideoSubsystem,
    pub event_pump: EventPump,
    pub window: Window,
    pub window_maximized: bool,
    #[cfg(feature = "ttf-sdl2")]
    pub ttf: Sdl2TtfContext,
    pub context: Sdl,
    pub window_icon: Option<Surface<'static>>,
}

impl SdlParts {
    pub(crate) fn maybe_with_window_icon(mut self, image: Option<RawRgbaImage>) -> Self {
        if let Some(image) = image {
            self.set_window_icon(image);
        }
        self
    }

    pub(crate) fn set_window_icon(&mut self, image: RawRgbaImage) {
        let (data, width, height) = image.destruct();
        let mut data = data.into_owned();
        let source = Surface::from_data(
            &mut data,
            width,
            height,
            width * 4,
            PixelFormatEnum::RGBA8888,
        )
        .unwrap();

        let mut target = Surface::new(width, height, PixelFormatEnum::RGBA8888).unwrap();

        source.blit(None, &mut target, None).unwrap();

        self.window.set_icon(&target);
        self.window_icon = Some(target);
    }
}
