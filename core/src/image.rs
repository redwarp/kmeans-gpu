use std::ops::Deref;

pub trait Container: Deref<Target = [[u8; 4]]> + Sized {
    fn to_pixel_vec(self) -> Vec<u8> {
        bytemuck::cast_slice::<_, u8>(&self).to_vec()
    }
}

impl Container for Vec<[u8; 4]> {
    fn to_pixel_vec(self) -> Vec<u8> {
        bytemuck::cast_vec(self)
    }
}

impl Container for &[[u8; 4]] {}

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

    pub fn get_pixel(&self, x: u32, y: u32) -> &[u8; 4] {
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

pub fn copied_pixel(dimensions: (u32, u32), rbga: &[u8]) -> Image<Vec<[u8; 4]>> {
    let mut pixels = Vec::with_capacity(dimensions.0 as usize * dimensions.1 as usize);
    pixels.extend_from_slice(bytemuck::cast_slice(rbga));
    Image {
        dimensions,
        rgba: pixels,
    }
}

pub fn borrowed_pixel(dimensions: (u32, u32), rbga: &[u8]) -> Image<&[[u8; 4]]> {
    Image {
        dimensions,
        rgba: bytemuck::cast_slice(rbga),
    }
}
