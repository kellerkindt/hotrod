use std::time::{Duration, Instant};

pub struct FpsManager {
    target_frame_rate: u32,
    target_duration: Duration,
    last_instant: Option<Instant>,
}

impl FpsManager {
    pub fn new(target_frame_rate: u32) -> Self {
        Self {
            target_frame_rate,
            target_duration: Self::target_duration(target_frame_rate),
            last_instant: None,
        }
    }

    pub fn set_target_frame_rate(&mut self, target_frame_rate: u32) {
        self.target_frame_rate = target_frame_rate;
    }

    pub fn delay(&mut self) -> Duration {
        if let Some(before) = self.last_instant.take() {
            let duration = before.elapsed();
            let target_duration = self.target_duration;
            if duration < target_duration {
                let to_sleep = target_duration - duration;
                std::thread::sleep(to_sleep);
                return to_sleep;
            }
        }
        self.last_instant = Some(Instant::now());
        Duration::ZERO
    }

    #[inline]
    fn target_duration(target_frame_rate: u32) -> Duration {
        Duration::from_secs_f32(1.0_f32 / (target_frame_rate as f32))
    }
}
