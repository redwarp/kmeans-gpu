use anyhow::{anyhow, Result};
use palette::{IntoColor, Lab, Pixel, Srgba};
use rgb::ComponentSlice;
pub use rgb::RGBA8;
use std::sync::Arc;
use std::{fmt::Display, str::FromStr, time::Instant};
use wgpu::{
    Backends, Device, DeviceDescriptor, Features, Instance, PowerPreference, Queue,
    RequestAdapterOptionsBase,
};

use crate::image::{Container, Image};
use crate::structures::{CentroidsBuffer, InputTexture};

mod future;
mod modules;
mod octree;
mod operations;
#[cfg(test)]
mod shader_tests;
mod structures;
mod utils;

pub mod image;

pub struct ImageProcessor {
    device: Arc<Device>,
    queue: Arc<Queue>,
    query_time: bool,
}

impl ImageProcessor {
    /// Create a new ImageProcessor, initializing a [wgpu::Device] and [wgpu::Queue]
    /// to use in future operations.
    /// ```rust,no_run
    /// use pollster::FutureExt;
    /// use kmeans_color_gpu::ImageProcessor;
    ///
    /// let image_processor = ImageProcessor::new().block_on();
    /// ```
    pub async fn new() -> Result<Self> {
        let instance = Instance::new(Backends::all());
        let adapter = instance
            .request_adapter(&RequestAdapterOptionsBase {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Couldn't create the adapter"))?;

        let features = adapter.features();
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    features: features & (Features::TIMESTAMP_QUERY),
                    limits: Default::default(),
                },
                None,
            )
            .await?;
        let query_time = device.features().contains(Features::TIMESTAMP_QUERY);

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            query_time,
        })
    }

    pub async fn palette<C: Container>(
        &self,
        color_count: u32,
        image: &Image<C>,
        algo: Algorithm,
    ) -> Result<Vec<RGBA8>> {
        match algo {
            Algorithm::Kmeans => kmeans_palette(self, color_count, image).await,
            Algorithm::Octree => octree_palette(self, color_count, image).await,
        }
    }

    pub async fn find<C: Container>(
        &self,
        image: &Image<C>,
        colors: &[RGBA8],
        reduce_mode: &ReduceMode,
    ) -> Result<Image<Vec<RGBA8>>> {
        let input_texture = InputTexture::new(&self.device, &self.queue, image);
        let centroids_buffer =
            CentroidsBuffer::fixed_centroids(colors, &ColorSpace::Lab, &self.device);

        match reduce_mode {
            ReduceMode::Replace => operations::find_colors(
                &self.device,
                &self.queue,
                &input_texture,
                &ColorSpace::Lab,
                &centroids_buffer,
                self.query_time,
            ),
            ReduceMode::Dither => operations::dither_colors(
                &self.device,
                &self.queue,
                &input_texture,
                &ColorSpace::Lab,
                &centroids_buffer,
                self.query_time,
            ),
            ReduceMode::Meld => operations::meld_colors(
                &self.device,
                &self.queue,
                &input_texture,
                &ColorSpace::Lab,
                &centroids_buffer,
                self.query_time,
            ),
        }?
        .pull_image(&self.device, &self.queue)
        .await
    }

    pub async fn reduce<C: Container>(
        &self,
        color_count: u32,
        image: &Image<C>,
        algo: &Algorithm,
        reduce_mode: &ReduceMode,
    ) -> Result<Image<Vec<RGBA8>>> {
        let input_texture = InputTexture::new(&self.device, &self.queue, image);

        let centroids_buffer = match algo {
            Algorithm::Kmeans => operations::extract_palette_kmeans(
                &self.device,
                &self.queue,
                &input_texture,
                &ColorSpace::Lab,
                color_count,
                self.query_time,
            )?,
            Algorithm::Octree => {
                let palette = octree_palette(self, color_count, image).await?;
                CentroidsBuffer::fixed_centroids(&palette, &ColorSpace::Lab, &self.device)
            }
        };

        match reduce_mode {
            ReduceMode::Replace => operations::find_colors(
                &self.device,
                &self.queue,
                &input_texture,
                &ColorSpace::Lab,
                &centroids_buffer,
                self.query_time,
            ),
            ReduceMode::Dither => operations::dither_colors(
                &self.device,
                &self.queue,
                &input_texture,
                &ColorSpace::Lab,
                &centroids_buffer,
                self.query_time,
            ),
            ReduceMode::Meld => operations::meld_colors(
                &self.device,
                &self.queue,
                &input_texture,
                &ColorSpace::Lab,
                &centroids_buffer,
                self.query_time,
            ),
        }?
        .pull_image(&self.device, &self.queue)
        .await
    }
}

#[derive(Clone, Copy)]
pub enum ColorSpace {
    Lab,
    Rgb,
}

impl ColorSpace {
    pub fn from(str: &str) -> Option<ColorSpace> {
        match str {
            "lab" => Some(ColorSpace::Lab),
            "rgb" => Some(ColorSpace::Rgb),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ColorSpace::Lab => "lab",
            ColorSpace::Rgb => "rgb",
        }
    }

    pub fn convergence(&self) -> f32 {
        match self {
            ColorSpace::Lab => 1.0,
            ColorSpace::Rgb => 0.01,
        }
    }
}

impl FromStr for ColorSpace {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "lab" => Ok(ColorSpace::Lab),
            "rgb" => Ok(ColorSpace::Rgb),
            _ => Err(anyhow!("Unsupported color space {s}")),
        }
    }
}

impl Display for ColorSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Clone, Copy)]
pub enum Algorithm {
    Kmeans,
    Octree,
}

impl Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Algorithm::Kmeans => "kmeans",
                Algorithm::Octree => "octree",
            }
        )
    }
}

#[derive(Clone, Copy)]
pub enum ReduceMode {
    Replace,
    Dither,
    Meld,
}

impl Display for ReduceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ReduceMode::Replace => "replace",
                ReduceMode::Dither => "dither",
                ReduceMode::Meld => "meld",
            }
        )
    }
}

async fn kmeans_palette<C: Container>(
    image_processor: &ImageProcessor,
    color_count: u32,
    image: &Image<C>,
) -> Result<Vec<RGBA8>> {
    let input_texture = InputTexture::new(&image_processor.device, &image_processor.queue, image);

    let mut colors = operations::extract_palette_kmeans(
        &image_processor.device,
        &image_processor.queue,
        &input_texture,
        &ColorSpace::Lab,
        color_count,
        image_processor.query_time,
    )?
    .pull_values(
        &image_processor.device,
        &image_processor.queue,
        &ColorSpace::Lab,
    )
    .await?;

    colors.sort_unstable_by(|a, b| {
        let a: Lab = Srgba::from_raw(a.as_slice())
            .into_format::<_, f32>()
            .into_color();
        let b: Lab = Srgba::from_raw(b.as_slice())
            .into_format::<_, f32>()
            .into_color();
        a.l.partial_cmp(&b.l).unwrap()
    });
    Ok(colors)
}

async fn octree_palette<C: Container>(
    image_processor: &ImageProcessor,
    color_count: u32,
    image: &Image<C>,
) -> Result<Vec<RGBA8>> {
    const MAX_SIZE: u32 = 128;

    let (width, height) = image.dimensions;
    let resized = if width > MAX_SIZE || height > MAX_SIZE {
        let input_texture = InputTexture::new(
            &image_processor.device,
            &image_processor.queue,
            image,
        )
        .resized(MAX_SIZE, &image_processor.device, &image_processor.queue);
        Some(
            input_texture
                .pull_image(&image_processor.device, &image_processor.queue)
                .await?,
        )
    } else {
        None
    };

    let pixels: &[RGBA8] = if let Some(resized) = &resized {
        &resized.rgba
    } else {
        &image.rgba
    };

    let start = Instant::now();
    let mut colors = operations::extract_palette_octree(pixels, color_count)?;

    if image_processor.query_time {
        let duration = start.elapsed();
        println!("Time elapsed in octree() is: {duration:?}");
    }

    colors.sort_unstable_by(|a, b| {
        let a: Lab = Srgba::from_raw(a.as_slice())
            .into_format::<_, f32>()
            .into_color();
        let b: Lab = Srgba::from_raw(b.as_slice())
            .into_format::<_, f32>()
            .into_color();
        a.l.partial_cmp(&b.l).unwrap()
    });

    Ok(colors)
}
