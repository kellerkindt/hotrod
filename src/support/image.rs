use std::borrow::Cow;

#[derive(Debug)]
pub struct RawRgbaImage {
    data: Cow<'static, [u8]>,
    width: u32,
    height: u32,
}

impl RawRgbaImage {
    pub fn new(data: impl Into<Cow<'static, [u8]>>, width: u32, height: u32) -> Self {
        Self {
            data: data.into(),
            width,
            height,
        }
    }

    #[inline]
    pub fn data(&self) -> &[u8] {
        self.data.as_ref()
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    pub fn destruct(self) -> (Cow<'static, [u8]>, u32, u32) {
        (self.data, self.width, self.height)
    }
}
