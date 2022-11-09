use std::ops::Deref;

use rgb::RGBA8;

pub trait Container: Deref<Target = [RGBA8]> + Sized {
    fn to_pixel_vec(self) -> Vec<u8> {
        bytemuck::cast_slice::<_, u8>(&self).to_vec()
    }
}

impl Container for Vec<RGBA8> {
    fn to_pixel_vec(self) -> Vec<u8> {
        bytemuck::cast_vec(self)
    }
}

impl Container for &[RGBA8] {}

/// Image struct manipulated by the library.
pub struct Image<C>
where
    C: Container,
{
    pub(crate) dimensions: (u32, u32),
    pub(crate) rgba: C,
}

impl<C> Image<C>
where
    C: Container,
{
    pub fn new(dimensions: (u32, u32), rgba: C) -> Self {
        Self { dimensions, rgba }
    }

    pub fn get_pixel(&self, x: u32, y: u32) -> &RGBA8 {
        let index = (x + y * self.dimensions.0) as usize;
        &self.rgba[index]
    }

    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    pub fn into_raw_pixels(self) -> Vec<u8> {
        self.rgba.to_pixel_vec()
    }
}

pub fn copied_pixel(dimensions: (u32, u32), rbga: &[u8]) -> Image<Vec<RGBA8>> {
    let mut pixels = Vec::with_capacity(dimensions.0 as usize * dimensions.1 as usize);
    pixels.extend_from_slice(bytemuck::cast_slice(rbga));
    Image {
        dimensions,
        rgba: pixels,
    }
}

pub fn borrowed_pixel(dimensions: (u32, u32), rbga: &[u8]) -> Image<&[RGBA8]> {
    Image {
        dimensions,
        rgba: bytemuck::cast_slice(rbga),
    }
}
