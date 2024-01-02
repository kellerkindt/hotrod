use crate::engine::types::world2d::{Dim, Pos};
use crate::support::sprite_sheet::{Sprite, SpriteSheet};
use serde_derive::Deserialize;

pub struct XmlTextureAtlas;

impl XmlTextureAtlas {
    pub fn load_from_str(
        content: &str,
        width: u32,
        height: u32,
    ) -> Result<SpriteSheet<f32>, serde_xml_rs::Error> {
        let atlas = dbg!(serde_xml_rs::from_str::<TextureAtlas>(content)?);
        let mut sprite_sheet = SpriteSheet::new(Dim::new(width, height));
        for texture in atlas.sub_textures {
            sprite_sheet.add(
                Sprite {
                    pos: Pos::new(texture.x, texture.y),
                    dim: Dim::new(texture.width, texture.height),
                },
                [texture.name],
            )
        }
        Ok(sprite_sheet.into_uv())
    }
}

#[derive(Debug, Deserialize)]
struct TextureAtlas {
    #[serde(rename = "SubTexture")]
    pub sub_textures: Vec<SubTexture>,
}

#[derive(Debug, Deserialize)]
struct SubTexture {
    pub name: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
