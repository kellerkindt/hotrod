use crate::engine::types::world2d::{Dim, Pos};
use crate::support::sprite_sheet::{Sprite, SpriteSheet};

pub struct SpriteSheetGridGenerator;

impl SpriteSheetGridGenerator {
    pub fn generate(width: u32, height: u32, sprite_size: u32) -> SpriteSheet<u32> {
        let mut sprite_sheet = SpriteSheet::new(Dim::new(width, height));

        for y in 0..(height / sprite_size) {
            for x in 0..(width / sprite_size) {
                sprite_sheet.add(
                    Sprite {
                        pos: Pos::new(x * sprite_size, y * sprite_size),
                        dim: Dim::new(sprite_size, sprite_size),
                    },
                    [format!("{x}_{y}")],
                );
            }
        }

        sprite_sheet
    }
}
