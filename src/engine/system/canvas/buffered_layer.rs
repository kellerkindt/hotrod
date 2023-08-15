use crate::engine::system::vulkan::lines::{Line, Vertex2d};
use crate::engine::system::vulkan::pipelines::VulkanPipelines;
use crate::engine::system::vulkan::textures::{TextureId, Textured, Vertex2dUv};
use crate::engine::system::vulkan::triangles::Triangles;
use crate::engine::types::world2d::{Dim, Pos};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};

type Uv<T> = Pos<T>;

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

    #[inline]
    pub fn draw_line<P: Into<Pos<f32>> + Copy>(&mut self, from: P, to: P) {
        self.draw_path(&[from, to])
    }

    #[inline]
    pub fn draw_rect<P: Into<Pos<f32>>, D: Into<Dim<f32>>>(&mut self, pos: P, dim: D) {
        let pos = pos.into();
        let dim = dim.into();
        self.draw_path(&[
            pos,
            pos + Dim::new(dim.x, 0.0),
            pos + dim,
            pos + Dim::new(0.0, dim.y),
            pos,
        ])
    }

    #[inline]
    pub fn fill_rect<P: Into<Pos<f32>>, D: Into<Dim<f32>>>(&mut self, pos: P, dim: D) {
        let pos = pos.into();
        let dim = dim.into();
        let triangle = Triangles {
            vertices: [
                pos,
                pos + Dim::new(dim.x, 0.0),
                pos + dim,
                pos + dim,
                pos + Dim::new(0.0, dim.y),
                pos,
            ]
            .into_iter()
            .map(|pos| crate::engine::system::vulkan::triangles::Vertex2d { pos: pos.into() })
            .collect::<Vec<_>>(),
            color: self.color,
        };
        if let Some(Action::Triangles(triangles)) = self.actions.last_mut() {
            triangles.push(triangle);
        } else {
            self.actions.push(Action::Triangles(vec![triangle]));
        }
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

    #[inline]
    pub fn draw_textured_rect<P: Into<Pos<f32>>, D: Into<Dim<f32>>>(
        &mut self,
        pos: P,
        dim: D,
        texture: TextureId,
    ) {
        let pos = pos.into();
        let dim = dim.into();
        self.draw_textured_triangles(
            [
                (pos, Uv::new(0.0, 0.0)),
                (pos + Dim::new(dim.x, 0.0), Uv::new(1.0, 0.0)),
                (pos + dim, Uv::new(1.0, 1.0)),
                (pos + dim, Uv::new(1.0, 1.0)),
                (pos + Dim::new(0.0, dim.y), Uv::new(0.0, 1.0)),
                (pos, Uv::new(0.0, 0.0)),
            ]
            .into_iter(),
            texture,
        );
    }

    pub fn draw_textured_triangles<P: Into<Pos<f32>>, U: Into<Uv<f32>>>(
        &mut self,
        pos_uv: impl Iterator<Item = (P, U)>,
        texture: TextureId,
    ) {
        let triangles = Textured {
            vertices: pos_uv
                .map(|(pos, uv)| {
                    let pos = pos.into();
                    let uv = uv.into();
                    Vertex2dUv {
                        pos: pos.into(),
                        uv: uv.into(),
                    }
                })
                .collect(),
            texture,
        };
        if let Some(Action::TexturedTriangle(textured)) = self.actions.last_mut() {
            textured.push(triangles);
        } else {
            self.actions.push(Action::TexturedTriangle(vec![triangles]));
        }
    }

    pub fn submit_to_render_pass(
        self,
        cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipeline: &mut VulkanPipelines,
    ) {
        for action in self.actions {
            match action {
                Action::Lines(lines) => {
                    if let Err(e) =
                        pipeline
                            .line
                            .draw(cmd, self.size[0] as f32, self.size[1] as f32, &lines)
                    {
                        eprintln!("{e:?}");
                    }
                }
                Action::Triangles(triangles) => {
                    if let Err(e) = pipeline.triangles.draw(
                        cmd,
                        self.size[0] as f32,
                        self.size[1] as f32,
                        &triangles,
                    ) {
                        eprintln!("{e:?}");
                    }
                }
                Action::TexturedTriangle(textured) => {
                    if let Err(e) = pipeline.texture.draw(
                        cmd,
                        self.size[0] as f32,
                        self.size[1] as f32,
                        &textured,
                    ) {
                        eprintln!("{e:?}");
                    }
                }
            }
        }
    }
}

enum Action {
    Lines(Vec<Line>),
    Triangles(Vec<Triangles>),
    TexturedTriangle(Vec<Textured>),
}
