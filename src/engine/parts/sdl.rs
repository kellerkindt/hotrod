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
}
