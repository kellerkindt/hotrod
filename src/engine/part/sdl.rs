use sdl2::{EventPump, Sdl, VideoSubsystem};
use sdl2::video::Window;

pub struct SdlPart {
    pub context: Sdl,
    pub video_subsystem: VideoSubsystem,
    pub event_pump: EventPump,
    pub window: Window,
}