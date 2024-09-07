use std::ops::Mul;

pub struct InterpolatedScalar {
    current: f32,
    target: f32,
}

impl From<f32> for InterpolatedScalar {
    #[inline]
    fn from(value: f32) -> Self {
        Self {
            current: value,
            target: value,
        }
    }
}

impl InterpolatedScalar {
    #[inline]
    pub fn update(&mut self, delta_seconds: f32) {
        self.update_with(
            delta_seconds,
            |target, current| target - current,
            |current, diff| current + diff,
        );
    }

    #[inline]
    pub fn update_radial_degrees(&mut self, delta_seconds: f32) {
        self.update_with(
            delta_seconds,
            |target, current| {
                let diff = target - current;
                if diff > 180.0 {
                    diff - 360.0
                } else if diff < -180.0 {
                    360.0 + diff
                } else {
                    diff
                }
            },
            |current, diff| (7200.0 + current + diff) % 360.0,
        );
    }

    pub fn update_with(
        &mut self,
        delta_seconds: f32,
        with_diff: impl FnOnce(f32, f32) -> f32,
        with_result: impl FnOnce(f32, f32) -> f32,
    ) {
        let diff = with_diff(self.target, self.current);
        let diff = diff
            .mul((8.0_f32 / 9.0_f32).powf(delta_seconds * 1000.0))
            .abs()
            .min(diff.abs())
            * diff.signum();
        self.current = with_result(self.current, diff);
    }

    #[inline]
    pub fn set(&mut self, value: f32) {
        self.current = value;
        self.target = value;
    }

    #[inline]
    pub fn set_target(&mut self, value: f32) {
        self.target = value;
    }

    #[inline]
    pub fn current(&self) -> f32 {
        self.current
    }
}
