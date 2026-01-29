use crate::engine::system::vulkan::textures::{TextureId, TextureInner};
use crate::engine::system::vulkan::PipelineTextureLoader;
use egui::ahash::AHashMap;
use std::any::{Any, TypeId};
use std::collections::hash_map::Entry;
use std::sync::Arc;
use vulkano::image::Image;
use vulkano::{Validated, VulkanError};

mod loader;
pub use loader::*;

/// The [`TextureRegistry`] has the purpose to provide a centralized location to register and
/// query image resources for various lookup keys.
#[derive(Default)]
pub struct TextureRegistry {
    register: AHashMap<TypeId, Vec<(Arc<dyn Any + Send + Sync + 'static>, TextureView)>>,
}

impl TextureRegistry {
    pub fn register_lookup<T>(&mut self, lookup: T, view: TextureView)
    where
        T: Any + Send + Sync + 'static,
    {
        let lookup = Arc::new(lookup) as Arc<dyn Any + Send + Sync + 'static>;
        match self.register.entry(TypeId::of::<T>()) {
            Entry::Occupied(mut vec) => {
                vec.get_mut().push((lookup, view));
            }
            Entry::Vacant(entry) => {
                entry.insert(vec![(lookup, view)]);
            }
        }
    }

    pub fn get<T: PartialEq + 'static>(&self, lookup: &T) -> Option<&TextureView> {
        self.register.get(&TypeId::of::<T>()).and_then(|vec| {
            vec.iter()
                .find(|(l, _)| l.downcast_ref::<T>() == Some(lookup))
                .map(|(_, v)| v)
        })
    }
}

#[derive(Clone)]
pub struct TextureView {
    texture: Arc<Texture>,
    uv0: Option<[f32; 2]>,
    uv1: Option<[f32; 2]>,
}

impl TextureView {
    #[inline]
    pub fn texture(&self) -> &Texture {
        &self.texture
    }

    #[inline]
    pub fn uv0(&self) -> Option<[f32; 2]> {
        self.uv0
    }

    pub fn uv0_or_default(&self) -> [f32; 2] {
        self.uv0.unwrap_or([0.0; 2])
    }

    #[inline]
    pub fn uv1(&self) -> Option<[f32; 2]> {
        self.uv1
    }

    pub fn uv1_or_default(&self) -> [f32; 2] {
        self.uv1.unwrap_or([1.0; 2])
    }

    pub fn width(&self) -> f32 {
        let w = self.uv1_or_default()[0] - self.uv0_or_default()[0];
        self.texture.width() as f32 * w
    }

    pub fn height(&self) -> f32 {
        let h = self.uv1_or_default()[1] - self.uv0_or_default()[1];
        self.texture.height() as f32 * h
    }

    #[cfg_attr(feature = "ui-egui", inline)]
    #[cfg(feature = "ui-egui")]
    pub fn egui_image(&self) -> Option<egui::Image<'_>> {
        self.texture.egui_texture().map(|texture| {
            let width = self.texture.width();
            let height = self.texture.height();
            let rect = egui::Rect {
                min: self.uv0_or_default().into(),
                max: self.uv1_or_default().into(),
            };
            let width = width as f32 * rect.width();
            let height = height as f32 * rect.height();
            egui::Image::new(egui::load::SizedTexture::new(
                texture.id(),
                egui::Vec2::new(width, height),
            ))
            .uv(rect)
            .maintain_aspect_ratio(true)
        })
    }

    pub fn subview(
        &self,
        uv0: impl Into<Option<[f32; 2]>>,
        uv1: impl Into<Option<[f32; 2]>>,
    ) -> Self {
        let uv0 = uv0.into();
        let uv1 = uv1.into();

        let this_uv0 = self.uv0.unwrap_or([0.0; 2]);
        let this_uv1 = self.uv1.unwrap_or([1.0; 2]);
        let width = this_uv1[0] - this_uv0[0];
        let height = this_uv1[1] - this_uv0[1];

        Self {
            texture: Arc::clone(&self.texture),
            uv0: uv0
                .map(|uv0| {
                    [
                        this_uv0[0] + uv0[0] * width,  //
                        this_uv0[1] + uv0[1] * height, //
                    ]
                })
                .or(self.uv0),
            uv1: uv1
                .map(|uv1| {
                    [
                        this_uv1[0] - (width - (uv1[0] * width)),   //
                        this_uv1[1] - (height - (uv1[1] * height)), //
                    ]
                })
                .or(self.uv1),
        }
    }
}

pub struct Texture {
    vulkan_image: Arc<Image>,
    #[cfg(feature = "image")]
    memory_image: image::DynamicImage,
    #[cfg(feature = "ui-egui")]
    egui_texture: Option<egui::TextureHandle>,
    texture_ids: Vec<(TypeId, Arc<dyn Any + Send + Sync>)>,
}

impl Texture {
    #[cfg_attr(feature = "image", inline)]
    #[cfg(feature = "image")]
    pub fn memory_image(&self) -> &image::DynamicImage {
        &self.memory_image
    }

    #[inline]
    pub fn vulkan_image(&self) -> &Arc<Image> {
        &self.vulkan_image
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.vulkan_image.extent()[0]
    }

    pub fn height(&self) -> u32 {
        self.vulkan_image.extent()[1]
    }

    #[cfg_attr(feature = "ui-egui", inline)]
    #[cfg(feature = "ui-egui")]
    pub fn egui_texture(&self) -> Option<&egui::TextureHandle> {
        self.egui_texture.as_ref()
    }

    #[cfg_attr(feature = "ui-egui", inline)]
    #[cfg(feature = "ui-egui")]
    pub fn egui_sized_texture(&self) -> Option<egui::load::SizedTexture> {
        self.egui_texture
            .as_ref()
            .map(egui::load::SizedTexture::from_handle)
    }

    #[cfg(feature = "ui-egui")]
    pub fn set_egui_texture(&mut self, texture_handle: impl Into<Option<egui::TextureHandle>>) {
        self.egui_texture = texture_handle.into();
    }

    pub fn get_texture_id<T>(&self) -> Option<TextureId<T>>
    where
        T: ?Sized + 'static,
        TextureInner<T>: Send + Sync,
    {
        self.texture_ids
            .iter()
            .find(|(id, _)| *id == TypeId::of::<T>())
            .and_then(|(_id, dyn_texture_id)| {
                match Arc::downcast::<TextureInner<T>>(Arc::clone(dyn_texture_id)) {
                    Ok(inner) => Some(TextureId(inner)),
                    Err(e) => {
                        #[cfg(debug_assertions)]
                        warn!(
                            "Failed to downcast to TextureId<{:?}>: {e:?}",
                            std::any::type_name::<T>()
                        );
                        #[cfg(not(debug_assertions))]
                        let _ = e;
                        None
                    }
                }
            })
    }

    pub fn load_and_register_for<P: PipelineTextureLoader + 'static>(
        &mut self,
        loader: &P,
    ) -> Result<TextureId<P>, Validated<VulkanError>> {
        let texture_id = loader.prepare_texture(Arc::clone(&self.vulkan_image))?;
        self.register_texture_id(texture_id.clone());
        Ok(texture_id)
    }

    #[cfg(all(feature = "ui-egui", feature = "image"))]
    pub fn load_and_register_for_egui(
        &mut self,
        name: &'static str,
        ctx: &egui::Context,
        texture_options: egui::TextureOptions,
    ) {
        self.egui_texture = Some(ctx.load_texture(
            name,
            egui::ColorImage::from_rgba_unmultiplied(
                [self.width() as _, self.height() as _],
                self.memory_image.to_rgba8().as_flat_samples().as_slice(),
            ),
            texture_options,
        ));
    }

    pub fn register_texture_id<T>(&mut self, texture_id: TextureId<T>)
    where
        T: ?Sized + 'static,
        TextureInner<T>: Send + Sync,
    {
        self.texture_ids
            .push((TypeId::of::<T>(), texture_id.0 as Arc<_>));
    }

    #[inline]
    pub fn finalize(self) -> TextureView {
        Arc::new(self).create_view(None, None)
    }

    pub fn create_view(
        self: Arc<Self>,
        uv0: impl Into<Option<[f32; 2]>>,
        uv1: impl Into<Option<[f32; 2]>>,
    ) -> TextureView {
        TextureView {
            texture: self,
            uv0: uv0.into(),
            uv1: uv1.into(),
        }
    }
}
