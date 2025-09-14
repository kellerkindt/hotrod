use crate::ui::egui::{Rect, Shape};
use egui::layers::ShapeIdx;
use egui::{
    Area, CollapsingHeader, Context, Frame, Id, LayerId, Painter, Response, Ui, WidgetText,
};

pub trait StylableContextExt {
    #[must_use]
    fn stylable_window<R>(
        &self,
        id: impl Into<Id>,
        conf: impl FnOnce(Area, Frame) -> (Area, Frame),
        ui: impl FnOnce(&mut Ui) -> R,
    ) -> StylableResponse<R>;
}

pub trait StylableUiExt {
    #[must_use]
    fn stylable_collapsing<R>(
        &mut self,
        heading: impl Into<WidgetText>,
        conf: impl FnOnce(Frame) -> Frame,
        add_contents: impl FnOnce(&mut Ui) -> R,
    ) -> StylableResponse<Option<R>>;

    #[must_use]
    fn stylized_frame<R>(
        &mut self,
        conf: impl FnOnce(Frame) -> Frame,
        add_contents: impl FnOnce(&mut Ui) -> R,
    ) -> StylableResponse<R>;
}

#[must_use]
pub struct StylableResponse<R> {
    idx: ShapeIdx,
    response: Response,
    returned: R,
}

impl<R> StylableResponse<R> {
    #[inline]
    pub fn stylize(self, style: impl FnOnce(Rect) -> Vec<Shape>) -> R {
        PainterInfo::from((self.idx, self.response)).render(style);
        self.returned
    }
}

#[must_use]
pub struct PainterInfo {
    idx: ShapeIdx,
    layer_id: LayerId,
    rect: Rect,
    ctx: Context,
}

impl From<(ShapeIdx, Response)> for PainterInfo {
    #[inline]
    fn from((idx, response): (ShapeIdx, Response)) -> Self {
        Self {
            idx,
            layer_id: response.layer_id,
            rect: response.rect,
            ctx: response.ctx,
        }
    }
}

impl PainterInfo {
    #[inline]
    pub fn render(self, shape: impl FnOnce(Rect) -> Vec<Shape>) {
        Painter::new(self.ctx, self.layer_id, self.rect).set(self.idx, shape(self.rect))
    }
}

impl StylableContextExt for Context {
    fn stylable_window<R>(
        &self,
        id: impl Into<Id>,
        conf: impl FnOnce(Area, Frame) -> (Area, Frame),
        ui_cb: impl FnOnce(&mut Ui) -> R,
    ) -> StylableResponse<R> {
        let (area, frame) = conf(Area::new(id.into()), Frame::NONE);
        let response = area.show(self, |ui| {
            let idx = ui.painter().add(Shape::Noop);
            let response = frame.show(ui, ui_cb);
            (idx, response)
        });
        StylableResponse {
            idx: response.inner.0,
            response: response.response,
            returned: response.inner.1.inner,
        }
    }
}

impl StylableUiExt for Ui {
    fn stylable_collapsing<R>(
        &mut self,
        heading: impl Into<WidgetText>,
        conf: impl FnOnce(Frame) -> Frame,
        add_contents: impl FnOnce(&mut Ui) -> R,
    ) -> StylableResponse<Option<R>> {
        let idx = self.painter().add(Shape::Noop);
        let response = conf(Frame::NONE).show(self, |ui| {
            CollapsingHeader::new(heading).show(ui, add_contents)
        });
        StylableResponse {
            idx,
            response: response.response,
            returned: response.inner.body_returned,
        }
    }

    fn stylized_frame<R>(
        &mut self,
        conf: impl FnOnce(Frame) -> Frame,
        add_contents: impl FnOnce(&mut Ui) -> R,
    ) -> StylableResponse<R> {
        let idx = self.painter().add(Shape::Noop);
        let response = conf(Frame::NONE).show(self, add_contents);
        StylableResponse {
            idx,
            response: response.response,
            returned: response.inner,
        }
    }
}
