use egui::Context;

#[derive(Default)]
pub struct EguiParts {
    pub content_callback: Option<Box<dyn FnMut(&Context)>>,
}
