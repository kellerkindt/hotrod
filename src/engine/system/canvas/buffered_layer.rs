use crate::engine::system::vulkan::lines::{Line, Vertex2d, VulkanLineSystem};
use crate::engine::types::world2d::Pos;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};

pub struct BufferedCanvasLayer {
    size: [u32; 2],
    actions: Vec<Action>,
    color: [f32; 4],
}

impl From<[u32; 2]> for BufferedCanvasLayer {
    fn from(size: [u32; 2]) -> Self {
        Self {
            size,
            actions: Vec::default(),
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

impl BufferedCanvasLayer {
    pub fn set_draw_color(&mut self, color: [f32; 4]) {
        self.color = color;
    }

    pub fn draw_line<P: Into<Pos<f32>> + Copy>(&mut self, from: P, to: P) {
        self.draw_path(&[from, to])
    }

    pub fn draw_path<P: Into<Pos<f32>> + Copy>(&mut self, positions: &[P]) {
        let line = Line {
            vertices: positions
                .iter()
                .copied()
                .map(|pos| Vertex2d {
                    pos: pos.into().into(),
                })
                .collect(),
            color: self.color,
        };
        if let Some(Action::Lines(lines)) = self.actions.last_mut() {
            lines.push(line);
        } else {
            self.actions.push(Action::Lines(vec![line]));
        }
    }

    pub fn submit_to_render_pass(
        self,
        cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        line_system: &mut VulkanLineSystem,
    ) {
        for action in self.actions {
            match action {
                Action::Lines(lines) => line_system
                    .draw(cmd, self.size[0] as f32, self.size[1] as f32, &lines)
                    .unwrap(),
            }
        }
    }
}

enum Action {
    Lines(Vec<Line>),
}
