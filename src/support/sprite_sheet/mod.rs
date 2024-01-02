use crate::engine::types::world2d::{Dim, Pos};
use egui::epaint::ahash::HashMap;
use std::borrow::Cow;
use std::ops::Index;

#[cfg(feature = "serde-xml-rs")]
pub mod xml_texture_atlas;

#[derive(Debug)]
pub struct SpriteSheet<T> {
    size: Dim<T>,
    sprites: Vec<Sprite<T>>,
    name_index: HashMap<Cow<'static, str>, usize>,
}

impl<T> SpriteSheet<T> {
    pub fn new(size: Dim<T>) -> Self {
        Self {
            size,
            sprites: Vec::default(),
            name_index: HashMap::default(),
        }
    }

    pub fn add<I, C>(&mut self, sprite: Sprite<T>, names: impl IntoIterator<Item = C, IntoIter = I>)
    where
        I: Iterator<Item = C>,
        C: Into<Cow<'static, str>>,
    {
        let index = self.sprites.len();
        self.sprites.push(sprite);
        for name in names {
            self.name_index.insert(name.into(), index);
        }
    }

    #[inline]
    pub fn names(&self) -> impl Iterator<Item = &Cow<'static, str>> {
        self.name_index.keys()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Sprite<T>> {
        self.sprites.iter()
    }
}

impl SpriteSheet<u32> {
    pub fn into_uv(self) -> SpriteSheet<f32> {
        let size = Dim::new(self.size.x as f32, self.size.y as f32);
        SpriteSheet {
            size,
            sprites: self
                .sprites
                .into_iter()
                .map(|sprite| Sprite {
                    pos: Pos::new(sprite.pos.x as f32 / size.x, sprite.pos.y as f32 / size.y),
                    dim: Dim::new(sprite.dim.x as f32 / size.x, sprite.dim.y as f32 / size.y),
                })
                .collect(),
            name_index: self.name_index,
        }
    }
}

impl<T> Index<usize> for SpriteSheet<T> {
    type Output = Sprite<T>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.sprites[index]
    }
}

impl<T> Index<&str> for SpriteSheet<T> {
    type Output = Sprite<T>;

    #[inline]
    fn index(&self, name: &str) -> &Self::Output {
        &self.sprites[self.name_index[&Cow::Borrowed(name)]]
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Sprite<T> {
    pub pos: Pos<T>,
    pub dim: Dim<T>,
}
