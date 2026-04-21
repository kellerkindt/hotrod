use crate::engine::types::world2d::Pos;

pub struct Path2d {
    position: Pos<f32>,
    lines: Vec<Vec<Pos<f32>>>,
    tolerance: f32,
}

impl Default for Path2d {
    fn default() -> Self {
        Self {
            position: Pos::new(0.0, 0.0),
            lines: Vec::new(),
            tolerance: Self::DEFAULT_TOLERANCE,
        }
    }
}

impl Path2d {
    pub const DEFAULT_TOLERANCE: f32 = 0.1;

    pub fn set_tolerance(&mut self, tolerance: f32) {
        self.tolerance = tolerance;
    }

    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.set_tolerance(tolerance);
        self
    }

    pub fn move_to(&mut self, position: impl Into<Pos<f32>>) -> &mut Self {
        self.position = position.into();
        if let Some(last) = self.lines.last() {
            if !last.is_empty() {
                self.lines.push(Vec::new());
            }
        }
        self
    }

    pub fn line_to(&mut self, position: impl Into<Pos<f32>>) -> &mut Self {
        let position = position.into();
        match self.lines.last_mut() {
            Some(last) => {
                if last.is_empty() {
                    last.push(self.position);
                }
                last.push(position)
            }
            None => {
                self.lines.push(vec![self.position, position]);
            }
        }
        self.position = position;
        self
    }

    /// Connects the end with the beginning.
    pub fn close(&mut self) -> &mut Self {
        if let Some(first) = self
            .lines
            .first()
            .and_then(|segment| segment.first().copied())
        {
            if let Some(current) = &mut self.lines.last_mut() {
                current.push(first);
            }
        }
        self
    }

    fn extend_current_line(&mut self, points: impl IntoIterator<Item = Pos<f32>>) {
        match self.lines.last_mut() {
            Some(last) => {
                last.extend(points);
                if let Some(last) = last.last() {
                    self.position = *last;
                }
            }
            None => {
                self.lines.push(points.into_iter().collect::<Vec<_>>());
                if let Some(last) = self.lines.last().and_then(|l| l.last()) {
                    self.position = *last;
                }
            }
        }
    }

    #[inline]
    pub fn into_lines(self) -> impl ExactSizeIterator<Item = Vec<Pos<f32>>> {
        self.lines.into_iter()
    }
}

#[cfg(feature = "lyon-geom")]
mod path_with_lyon_geom {
    use super::*;
    use lyon_geom::euclid::{Point2D, Vector2D};
    use lyon_geom::Angle;

    impl Path2d {
        pub fn arc(
            &mut self,
            x: f32,
            y: f32,
            radius: f32,
            start_angle: Angle<f32>,
            sweep_angle: Angle<f32>,
        ) -> &mut Self {
            self.extend_current_line(
                lyon_geom::Arc::<f32> {
                    center: Point2D::new(x, y),
                    radii: Vector2D::splat(radius),
                    start_angle,
                    sweep_angle,
                    x_rotation: Angle::zero(),
                }
                .flattened(self.tolerance)
                .map(|p| Pos::new(p.x, p.y)),
            );
            self
        }
    }
}
