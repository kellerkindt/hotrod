use crate::engine::system::texture::TextureView;
use crate::engine::system::vulkan::textures::TextureId;
use crate::engine::system::vulkan::PipelineTextureLoader;
use crate::engine::types::world2d::Pos;
use std::ops::Index;

pub struct TileMapLoader {
    tile_size: (u32, u32),
}

impl Default for TileMapLoader {
    fn default() -> Self {
        Self {
            tile_size: Self::DEFAULT_TILE_SIZE,
        }
    }
}

impl TileMapLoader {
    pub const DEFAULT_TILE_SIZE: (u32, u32) = (64, 64);

    pub fn with_tile_size(mut self, width: u32, height: u32) -> Self {
        self.tile_size = (width, height);
        self
    }

    pub fn load_tiles_from_texture<P: PipelineTextureLoader + 'static>(
        &self,
        view: &TextureView,
    ) -> Option<TileMap2d<P>> {
        let texture_id = &view.texture().get_texture_id::<P>()?;
        let image_width = view.width();
        let image_height = view.height();

        let tile_size_x = self.tile_size.0;
        let tile_size_y = self.tile_size.1;
        let tiles_x = (image_width / tile_size_x as f32) as u32;
        let tiles_y = (image_height / tile_size_y as f32) as u32;

        Some(TileMap2d {
            tiles: (0..tiles_y)
                .flat_map(|y| {
                    (0..tiles_x).map(move |x| {
                        let pos_x0 = x * tile_size_x;
                        let pos_y0 = y * tile_size_y;
                        let pos_x1 = (x + 1) * tile_size_x;
                        let pos_y1 = (y + 1) * tile_size_y;

                        let subview = view.subview(
                            [pos_x0 as f32 / image_width, pos_y0 as f32 / image_height],
                            [pos_x1 as f32 / image_width, pos_y1 as f32 / image_height],
                        );

                        TileSprite {
                            texture: texture_id.clone(),
                            uv0: subview.uv0_or_default().into(),
                            uv1: subview.uv1_or_default().into(),
                            view: subview,
                        }
                    })
                })
                .collect::<Vec<_>>(),
            width: tiles_x as u16,
            height: tiles_y as u16,
        })
    }
}

pub struct TileSprite<P> {
    pub texture: TextureId<P>,
    pub view: TextureView,
    pub uv0: Pos<f32>,
    pub uv1: Pos<f32>,
}

impl<P> Clone for TileSprite<P> {
    fn clone(&self) -> Self {
        Self {
            texture: self.texture.clone(),
            view: self.view.clone(),
            uv0: self.uv0,
            uv1: self.uv1,
        }
    }
}

pub struct TileMap2d<P> {
    tiles: Vec<TileSprite<P>>,
    width: u16,
    height: u16,
}

impl<P> TileMap2d<P> {
    #[inline]
    pub fn width(&self) -> u16 {
        self.width
    }

    #[inline]
    pub fn height(&self) -> u16 {
        self.height
    }

    #[inline]
    pub fn get_tile(&self, x: u16, y: u16) -> Option<&TileSprite<P>> {
        self.tiles.get(usize::from((y * self.width) + x))
    }
}

impl<T: Into<u16>, P> Index<(T, T)> for TileMap2d<P> {
    type Output = TileSprite<P>;

    #[inline]
    fn index(&self, (x, y): (T, T)) -> &Self::Output {
        self.get_tile(x.into(), y.into()).unwrap()
    }
}
