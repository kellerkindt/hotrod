use sdl2::video::Window;
use sdl2::{EventPump, Sdl, VideoSubsystem};

pub struct SdlParts {
    pub video_subsystem: VideoSubsystem,
    pub event_pump: EventPump,
    pub window: Window,
    pub context: Sdl,
}
