use crate::engine::types::world2d::{Dim, Pos};
use crate::support::world2d::view::{DragSource, Map2dView, SelectionSource, ZoomChangeSource};
use egui::{InputState, PointerButton};

impl ZoomChangeSource for &InputState {
    fn update_zoom_at_screen_position(&self, view: &mut Map2dView) {
        if let Some((new_zoom, pos)) = self.pointer.interact_pos().and_then(|pos| {
            if self.smooth_scroll_delta.y > 0.0 {
                Some((view.zoom() * 1.2, pos))
            } else if self.smooth_scroll_delta.y < 0.0 {
                Some((view.zoom() / 1.2, pos))
            } else {
                None
            }
        }) {
            view.update_zoom_at_screen_position(
                new_zoom,
                Pos::new(pos.x * self.pixels_per_point, pos.y * self.pixels_per_point),
            );
        }
    }
}

impl DragSource for &InputState {
    fn update_view_position_by_drag_delta(&self, view: &mut Map2dView) {
        if self.pointer.is_decidedly_dragging()
            && self.pointer.button_down(PointerButton::Secondary)
        {
            let velocity = self.pointer.delta();
            view.move_by_screen_delta(
                velocity.x * self.pixels_per_point,
                velocity.y * self.pixels_per_point,
            );
        }
    }
}

impl SelectionSource for &InputState {
    fn capture_screen_selection(&self) -> Option<(Pos<f32>, Dim<f32>)> {
        if self.pointer.is_decidedly_dragging() && self.pointer.button_down(PointerButton::Primary)
        {
            if let Some(origin) = self.pointer.interact_pos() {
                let origin = Pos::new(
                    origin.x * self.pixels_per_point,
                    origin.y * self.pixels_per_point,
                );
                let distance = Dim::new(
                    self.pointer.delta().x * self.pixels_per_point,
                    self.pointer.delta().y * self.pixels_per_point,
                );
                return Some((origin, distance));
            }
        }
        None
    }
}
