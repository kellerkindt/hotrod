use egui::Color32;

/// RGBA f32 color information.
pub struct Color {
    pub rgba: [f32; 4],
}

impl Default for Color {
    #[inline]
    fn default() -> Self {
        Self::from([1.0, 1.0, 1.0, 1.0])
    }
}

impl From<[f32; 4]> for Color {
    #[inline]
    fn from(rgba: [f32; 4]) -> Self {
        Self { rgba }
    }
}

impl From<[u8; 4]> for Color {
    #[inline]
    fn from(value: [u8; 4]) -> Self {
        Self::from(value.map(|v| (f32::from(v) / f32::from(u8::MAX)).round()))
    }
}

impl From<Color32> for Color {
    #[inline]
    fn from(value: Color32) -> Self {
        Self::from([value.r(), value.g(), value.b(), value.a()])
    }
}

impl From<sdl2::pixels::Color> for Color {
    #[inline]
    fn from(value: sdl2::pixels::Color) -> Self {
        Self::from([value.r, value.g, value.b, value.a])
    }
}
