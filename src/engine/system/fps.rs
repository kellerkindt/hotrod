use std::time::{Duration, Instant};

pub struct FpsManager {
    target_duration: Duration,
    last_instant: Option<Instant>,
}

impl FpsManager {
    pub fn new(target_frame_rate: u16) -> Self {
        Self {
            target_duration: Self::target_duration(target_frame_rate),
            last_instant: None,
        }
    }

    pub fn set_target_frame_rate(&mut self, target_frame_rate: u16) {
        self.target_duration = Self::target_duration(target_frame_rate);
    }

    pub fn delay(&mut self) -> Duration {
        let mut slept = Duration::ZERO;
        if let Some(before) = self.last_instant.take() {
            let duration = before.elapsed();
            let target_duration = self.target_duration;
            if duration < target_duration {
                slept = target_duration - duration;
                std::thread::sleep(slept);
            }
        }
        self.last_instant = Some(Instant::now());
        slept
    }

    #[inline]
    fn target_duration(target_frame_rate: u16) -> Duration {
        Duration::from_secs_f32(1.0_f32 / (target_frame_rate as f32))
    }
}
