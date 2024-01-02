use crate::engine::system::vulkan::lines::{Line, Vertex2d};
use crate::engine::system::vulkan::pipelines::VulkanPipelines;
use crate::engine::system::vulkan::system::RenderContext;
use crate::engine::system::vulkan::textured::{TextureId, Textured, Vertex2dUv};
use crate::engine::system::vulkan::triangles::Triangles;
use crate::engine::system::vulkan::DrawError;
use crate::engine::types::world2d::{Dim, Pos};
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};

type Uv<T> = Pos<T>;

pub struct BufferedCanvasLayer {
    color: [f32; 4],
    sink: ActionSink,
}

impl Default for BufferedCanvasLayer {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 1.0],
            sink: ActionSink::Buffer(Vec::default()),
        }
    }
}

impl BufferedCanvasLayer {
    pub fn new(
        builder: AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
        pipelines: Arc<VulkanPipelines>,
    ) -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 1.0],
            sink: ActionSink::Commands {
                current: None,
                builder,
                pipelines,
            },
        }
    }

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

    pub fn fill_rect<P: Into<Pos<f32>>, D: Into<Dim<f32>>>(&mut self, pos: P, dim: D) {
        let pos = pos.into();
        let dim = dim.into();
        self.sink.append(Triangles {
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
        });
    }

    pub fn draw_path<P: Into<Pos<f32>> + Copy>(&mut self, positions: &[P]) {
        self.sink.append(Line {
            vertices: positions
                .iter()
                .copied()
                .map(|pos| Vertex2d {
                    pos: pos.into().into(),
                })
                .collect(),
            color: self.color,
        });
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
        self.sink.append(Textured {
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
        });
    }

    #[must_use]
    pub fn flush(
        self,
        ctx: &RenderContext,
        pipelines: &VulkanPipelines,
    ) -> Arc<SecondaryAutoCommandBuffer> {
        self.sink.flush(ctx, pipelines)
    }
}

enum ActionSink {
    Buffer(Vec<Action>),
    Commands {
        current: Option<Action>,
        builder: AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
        pipelines: Arc<VulkanPipelines>,
    },
}

impl ActionSink {
    pub fn append(&mut self, action: impl Into<Action>) {
        let action = action.into();
        if let Some(current) = self.action_mut() {
            if let Some(action) = current.try_push(action) {
                self.push_action(action);
            }
        } else {
            self.push_action(action);
        }
    }

    pub fn action_mut(&mut self) -> Option<&mut Action> {
        match self {
            ActionSink::Buffer(buffer) => buffer.last_mut(),
            ActionSink::Commands { current, .. } => current.as_mut(),
        }
    }

    pub fn push_action(&mut self, action: Action) {
        match self {
            ActionSink::Buffer(buffer) => buffer.push(action),
            ActionSink::Commands {
                current,
                builder,
                pipelines,
            } => {
                if let Some(prev) = current.replace(action) {
                    if let Err(e) = prev.flush(builder, pipelines) {
                        eprintln!("{e:?}");
                    }
                }
            }
        }
    }

    pub fn flush(
        self,
        ctx: &RenderContext,
        pipelines: &VulkanPipelines,
    ) -> Arc<SecondaryAutoCommandBuffer> {
        match self {
            ActionSink::Buffer(buffer) => {
                let mut builder = ctx.create_render_buffer_builder().unwrap();
                for action in buffer {
                    if let Err(e) = action.flush(&mut builder, pipelines) {
                        eprintln!("{e:?}");
                    }
                }
                builder.build().unwrap()
            }
            ActionSink::Commands {
                current,
                mut builder,
                pipelines,
            } => {
                if let Some(action) = current {
                    if let Err(e) = action.flush(&mut builder, &pipelines) {
                        eprintln!("{e:?}");
                    }
                }
                builder.build().unwrap()
            }
        }
    }
}

enum Action {
    Lines(Vec<Line>),
    Triangles(Vec<Triangles>),
    TexturedTriangle(Vec<Textured>),
}

impl Action {
    pub fn try_push(&mut self, other: Action) -> Option<Action> {
        macro_rules! try_push {
            ($($ty:ident, )+) => {
                match (self, other) {
                    $(
                        (Action::$ty(dst), Action::$ty(src)) => {
                            dst.extend(src.into_iter());
                            None
                        }
                        (Action::$ty(_), other) => Some(other),
                    )+
                }
            }
        }

        try_push!(Lines, Triangles, TexturedTriangle,)
    }

    pub fn flush<L>(
        self,
        builder: &mut AutoCommandBufferBuilder<L>,
        pipelines: &VulkanPipelines,
    ) -> Result<(), DrawError> {
        match self {
            Action::Lines(lines) => pipelines.line.draw(builder, &lines),
            Action::Triangles(triangles) => pipelines.triangles.draw(builder, &triangles),
            Action::TexturedTriangle(textured) => pipelines.texture.draw(builder, &textured),
        }
    }
}

impl From<Line> for Action {
    fn from(value: Line) -> Self {
        Action::Lines(vec![value])
    }
}

impl From<Triangles> for Action {
    fn from(value: Triangles) -> Self {
        Action::Triangles(vec![value])
    }
}

impl From<Textured> for Action {
    fn from(value: Textured) -> Self {
        Action::TexturedTriangle(vec![value])
    }
}
