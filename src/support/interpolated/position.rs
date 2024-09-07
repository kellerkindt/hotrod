use crate::support::interpolated::InterpolatedScalar;

pub struct InterpolatedPosition {
    x: InterpolatedScalar,
    y: InterpolatedScalar,
}

impl InterpolatedPosition {
    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x: InterpolatedScalar::from(x),
            y: InterpolatedScalar::from(y),
        }
    }

    pub fn update(&mut self, delta_seconds: f32) {
        self.x.update(delta_seconds);
        self.y.update(delta_seconds);
    }

    #[inline]
    pub fn set(&mut self, x: f32, y: f32) {
        self.x.set(x);
        self.y.set(y);
    }

    #[inline]
    pub fn set_target(&mut self, x: f32, y: f32) {
        self.x.set_target(x);
        self.y.set_target(y);
    }

    #[inline]
    pub fn current(&self) -> (f32, f32) {
        (self.x.current(), self.y.current())
    }
}
