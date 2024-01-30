use crate::engine::system::vulkan::desc::binding_201_world_2d_view::World2dView;
use crate::engine::types::world2d::{Dim, Pos};

pub struct Map2dView {
    screen_width: u32,
    screen_height: u32,
    view_x: f32,
    view_y: f32,
    zoom: f32,
}

impl Map2dView {
    pub fn new(screen_width: u32, screen_height: u32) -> Self {
        Self {
            screen_width,
            screen_height,
            view_x: 0.0,
            view_y: 0.0,
            zoom: 1.0f32,
        }
    }

    #[inline]
    pub fn update_screen_size(&mut self, width: u32, height: u32) {
        self.screen_width = width;
        self.screen_height = height;
    }

    #[inline]
    pub fn move_by_screen_delta(&mut self, dx: f32, dy: f32) {
        self.view_x -= dx / self.zoom;
        self.view_y -= dy / self.zoom;
    }

    #[inline]
    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    #[inline]
    pub fn position_world_to_screen(&self, pos: Pos<f32>) -> Pos<f32> {
        Pos::new(
            (self.screen_width as f32 / 2_f32) + ((pos.x - self.view_x) * self.zoom),
            (self.screen_height as f32 / 2_f32) + ((pos.y - self.view_y) * self.zoom),
        )
    }

    #[inline]
    pub fn position_screen_to_world(&self, pos: Pos<f32>) -> Pos<f32> {
        Pos::new(
            ((pos.x - (self.screen_width as f32 / 2_f32)) / self.zoom) + self.view_x,
            ((pos.y - (self.screen_height as f32 / 2_f32)) / self.zoom) + self.view_y,
        )
    }

    #[inline]
    pub fn distance_world_to_screen(&self, dim: Dim<f32>) -> Dim<f32> {
        Dim::new(dim.x * self.zoom, dim.y * self.zoom)
    }

    pub fn update_zoom_at_screen_position(&mut self, new_zoom: f32, pos: Pos<f32>) {
        let world_pos_before = self.position_screen_to_world(pos);
        self.zoom = new_zoom;

        let world_pos_after = self.position_screen_to_world(pos);
        let world_pos_diff = world_pos_after - world_pos_before;

        self.view_x -= world_pos_diff.x;
        self.view_y -= world_pos_diff.y;
    }

    #[inline]
    pub fn to_world_2d_view(&self) -> World2dView {
        World2dView::from([self.view_x, self.view_y, self.zoom])
    }
}
