use crate::engine::types::world2d::Pos;
use crate::support::world2d::view::{DragSource, Map2dView, ZoomChangeSource};
use egui::InputState;

impl ZoomChangeSource for &InputState {
    fn update_zoom_at_screen_position(&self, view: &mut Map2dView) {
        if let Some((new_zoom, pos)) = self.pointer.interact_pos().and_then(|pos| {
            if self.raw_scroll_delta.y > 0.0 {
                Some((view.zoom() * 1.2, pos))
            } else if self.raw_scroll_delta.y < 0.0 {
                Some((view.zoom() / 1.2, pos))
            } else {
                None
            }
        }) {
            view.update_zoom_at_screen_position(new_zoom, Pos::new(pos.x, pos.y));
        }
    }
}

impl DragSource for &InputState {
    fn update_view_position_by_drag_delta(&self, view: &mut Map2dView) {
        if self.pointer.is_decidedly_dragging() {
            let velocity = self.pointer.delta();
            view.move_by_screen_delta(velocity.x, velocity.y);
        }
    }
}
