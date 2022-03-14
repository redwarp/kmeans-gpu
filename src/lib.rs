use anyhow::Result;
use modules::{
    ChooseCentroidModule, ColorConverterModule, ColorReverterModule, ComputeBlock,
    FindCentroidModule, Module, SwapModule,
};
use palette::{IntoColor, Lab, Srgba};
use pollster::FutureExt;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::vec;
use utils::padded_bytes_per_row;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BufferAddress, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, DeviceDescriptor, Features, ImageDataLayout, Instance, MapMode,
    PowerPreference, QuerySetDescriptor, QueryType, RequestAdapterOptionsBase, TextureDimension,
    TextureFormat, TextureUsages,
};

use crate::modules::ConvertCentroidColorsModule;

mod modules;
mod utils;

pub struct Image {
    pub(crate) dimensions: (u32, u32),
    pub(crate) rgba: Vec<[u8; 4]>,
}

impl Image {
    pub fn new(dimensions: (u32, u32), rgba: Vec<[u8; 4]>) -> Self {
        Self { dimensions, rgba }
    }

    pub fn from_raw_pixels(dimensions: (u32, u32), rbga: &[u8]) -> Self {
        let mut pixels = Vec::with_capacity(dimensions.0 as usize * dimensions.1 as usize);
        pixels.extend_from_slice(bytemuck::cast_slice(rbga));
        Self {
            dimensions,
            rgba: pixels,
        }
    }

    pub fn get_pixel(&self, x: u32, y: u32) -> &[u8; 4] {
        let index = (x + y * self.dimensions.0) as usize;
        &self.rgba[index]
    }

    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    pub fn into_raw_pixels(self) -> Vec<u8> {
        self.rgba.into_iter().flatten().collect()
    }
}

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
            ColorSpace::Lab => 0.75,
            ColorSpace::Rgb => 0.01,
        }
    }
}

pub fn kmeans(k: u32, image: &Image, color_space: &ColorSpace) -> Result<Image> {
    let (width, height) = image.dimensions;

    let centroids = init_centroids(image, k, color_space);

    let instance = Instance::new(Backends::all());
    let adapter = instance
        .request_adapter(&RequestAdapterOptionsBase {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .block_on()
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
        .block_on()?;

    let query_set = if features.contains(Features::TIMESTAMP_QUERY) {
        Some(device.create_query_set(&QuerySetDescriptor {
            count: 2,
            ty: QueryType::Timestamp,
            label: None,
        }))
    } else {
        None
    };
    let query_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &[0; 16],
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    });

    let texture_size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let input_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("input texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    });

    queue.write_texture(
        input_texture.as_image_copy(),
        bytemuck::cast_slice(&image.rgba),
        ImageDataLayout {
            offset: 0,
            bytes_per_row: std::num::NonZeroU32::new(4 * width),
            rows_per_image: None,
        },
        texture_size,
    );

    let work_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("work texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
    });

    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
    });

    let centroid_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &centroids,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let color_index_size =
        (width * height) as BufferAddress * std::mem::size_of::<u32>() as BufferAddress;
    let color_index_buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size: color_index_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let color_converter_module = ColorConverterModule::new(
        &device,
        color_space,
        image.dimensions,
        &input_texture,
        &work_texture,
    );
    let color_reverter_module = ColorReverterModule::new(
        &device,
        color_space,
        image.dimensions,
        &work_texture,
        &output_texture,
    );
    let find_centroid_module = FindCentroidModule::new(
        &device,
        image.dimensions,
        &work_texture,
        &centroid_buffer,
        &color_index_buffer,
    );
    let mut choose_centroid_module = ChooseCentroidModule::new(
        &device,
        color_space,
        image.dimensions,
        k,
        &work_texture,
        &centroid_buffer,
        &color_index_buffer,
        &find_centroid_module,
    );

    let swap_module = SwapModule::new(
        &device,
        image.dimensions,
        &work_texture,
        &centroid_buffer,
        &color_index_buffer,
    );

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 0);
    }

    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Init pass"),
        });
        color_converter_module.dispatch(&mut compute_pass);

        find_centroid_module.dispatch(&mut compute_pass);
    }

    queue.submit(Some(encoder.finish()));

    choose_centroid_module.compute(&device, &queue);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Swap and fetch result pass"),
        });
        swap_module.dispatch(&mut compute_pass);
        color_reverter_module.dispatch(&mut compute_pass);
    }
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 1);
    }

    let padded_bytes_per_row = padded_bytes_per_row(width as u64 * 4);
    let unpadded_bytes_per_row = width as usize * 4;

    let output_buffer_size =
        padded_bytes_per_row as u64 * height as u64 * std::mem::size_of::<u8>() as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_buffer_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let centroid_size = centroids.len() as BufferAddress;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: centroid_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            aspect: wgpu::TextureAspect::All,
            texture: &output_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::ImageCopyBuffer {
            buffer: &output_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(padded_bytes_per_row as u32),
                rows_per_image: std::num::NonZeroU32::new(height),
            },
        },
        texture_size,
    );

    encoder.copy_buffer_to_buffer(&centroid_buffer, 0, &staging_buffer, 0, centroid_size);

    if let Some(query_set) = &query_set {
        encoder.resolve_query_set(query_set, 0..2, &query_buf, 0);
    }
    queue.submit(Some(encoder.finish()));

    let buffer_slice = output_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(MapMode::Read);

    let cent_buffer_slice = staging_buffer.slice(..);
    let cent_buffer_future = cent_buffer_slice.map_async(MapMode::Read);

    let query_slice = query_buf.slice(..);
    let query_future = query_slice.map_async(MapMode::Read);

    device.poll(wgpu::Maintain::Wait);

    if let Ok(()) = cent_buffer_future.block_on() {
        let data = cent_buffer_slice.get_mapped_range();

        for (index, k) in bytemuck::cast_slice::<u8, f32>(&data[16..])
            .chunks_exact(4)
            .enumerate()
        {
            println!("Centroid {index} = {k:?}")
        }
    }

    if query_future.block_on().is_ok() && features.contains(Features::TIMESTAMP_QUERY) {
        let ts_period = queue.get_timestamp_period();
        let ts_data_raw = &*query_slice.get_mapped_range();
        let ts_data: &[u64] = bytemuck::cast_slice(ts_data_raw);
        println!(
            "Compute shader elapsed: {:?}ms",
            (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
        );
    }

    match buffer_future.block_on() {
        Ok(()) => {
            let padded_data = buffer_slice.get_mapped_range();
            let mut pixels: Vec<u8> = vec![0; unpadded_bytes_per_row * height as usize];
            for (padded, pixels) in padded_data
                .chunks_exact(padded_bytes_per_row as usize)
                .zip(pixels.chunks_exact_mut(unpadded_bytes_per_row))
            {
                pixels.copy_from_slice(&padded[..unpadded_bytes_per_row]);
            }

            let result = Image::from_raw_pixels((width, height), &pixels);

            Ok(result)
        }
        Err(e) => Err(e.into()),
    }
}

pub fn palette(k: u32, image: &Image, color_space: &ColorSpace) -> Result<Vec<[u8; 4]>> {
    let (width, height) = image.dimensions;

    let centroids = init_centroids(image, k, color_space);

    let instance = Instance::new(Backends::all());
    let adapter = instance
        .request_adapter(&RequestAdapterOptionsBase {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .block_on()
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
        .block_on()?;

    let query_set = if features.contains(Features::TIMESTAMP_QUERY) {
        Some(device.create_query_set(&QuerySetDescriptor {
            count: 2,
            ty: QueryType::Timestamp,
            label: None,
        }))
    } else {
        None
    };
    let query_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &[0; 16],
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    });

    let texture_size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let input_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("input texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    });

    queue.write_texture(
        input_texture.as_image_copy(),
        bytemuck::cast_slice(&image.rgba),
        ImageDataLayout {
            offset: 0,
            bytes_per_row: std::num::NonZeroU32::new(4 * width),
            rows_per_image: None,
        },
        texture_size,
    );

    let work_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("work texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
    });

    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
    });

    let centroid_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &centroids,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let color_index_size =
        (width * height) as BufferAddress * std::mem::size_of::<u32>() as BufferAddress;
    let color_index_buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size: color_index_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let color_converter_module = ColorConverterModule::new(
        &device,
        color_space,
        image.dimensions,
        &input_texture,
        &work_texture,
    );
    let color_reverter_module = ColorReverterModule::new(
        &device,
        color_space,
        image.dimensions,
        &work_texture,
        &output_texture,
    );
    let find_centroid_module = FindCentroidModule::new(
        &device,
        image.dimensions,
        &work_texture,
        &centroid_buffer,
        &color_index_buffer,
    );
    let mut choose_centroid_module = ChooseCentroidModule::new(
        &device,
        color_space,
        image.dimensions,
        k,
        &work_texture,
        &centroid_buffer,
        &color_index_buffer,
        &find_centroid_module,
    );

    let convert_centroid_colors_module =
        ConvertCentroidColorsModule::new(&device, color_space, &centroid_buffer);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 0);
    }

    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Init pass"),
        });
        color_converter_module.dispatch(&mut compute_pass);

        find_centroid_module.dispatch(&mut compute_pass);
    }

    queue.submit(Some(encoder.finish()));

    choose_centroid_module.compute(&device, &queue);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Swap and fetch result pass"),
        });
        convert_centroid_colors_module.dispatch(&mut compute_pass);
        color_reverter_module.dispatch(&mut compute_pass);
    }
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 1);
    }

    let centroid_size = centroids.len() as BufferAddress;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: centroid_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&centroid_buffer, 0, &staging_buffer, 0, centroid_size);

    if let Some(query_set) = &query_set {
        encoder.resolve_query_set(query_set, 0..2, &query_buf, 0);
    }
    queue.submit(Some(encoder.finish()));

    let cent_buffer_slice = staging_buffer.slice(..);
    let cent_buffer_future = cent_buffer_slice.map_async(MapMode::Read);

    let query_slice = query_buf.slice(..);
    let query_future = query_slice.map_async(MapMode::Read);

    device.poll(wgpu::Maintain::Wait);

    if query_future.block_on().is_ok() && features.contains(Features::TIMESTAMP_QUERY) {
        let ts_period = queue.get_timestamp_period();
        let ts_data_raw = &*query_slice.get_mapped_range();
        let ts_data: &[u64] = bytemuck::cast_slice(ts_data_raw);
        println!(
            "Compute shader elapsed: {:?}ms",
            (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
        );
    }

    match cent_buffer_future.block_on() {
        Ok(()) => {
            let data = cent_buffer_slice.get_mapped_range();

            for (index, k) in bytemuck::cast_slice::<u8, f32>(&data[16..])
                .chunks_exact(4)
                .enumerate()
            {
                println!("Centroid {index} = {k:?}")
            }

            let colors: Vec<_> = bytemuck::cast_slice::<u8, f32>(&data[16..])
                .chunks_exact(4)
                .map(|color| {
                    [
                        (color[0] * 255.0) as u8,
                        (color[1] * 255.0) as u8,
                        (color[2] * 255.0) as u8,
                        (color[3] * 255.0) as u8,
                    ]
                })
                .collect();
            Ok(colors)
        }
        Err(e) => Err(e.into()),
    }
}

fn init_centroids(image: &Image, k: u32, color_space: &ColorSpace) -> Vec<u8> {
    let mut centroids: Vec<u8> = vec![];
    // Aligned 16, see https://www.w3.org/TR/WGSL/#address-space-layout-constraints
    centroids.extend_from_slice(bytemuck::cast_slice(&[k, 0, 0, 0]));

    let mut rng = StdRng::seed_from_u64(42);

    let (width, height) = image.dimensions;
    let total_px = width * height;
    let mut picked_indices = Vec::with_capacity(k as usize);

    for _ in 0..k {
        loop {
            let color_index = rng.gen_range(0..total_px);
            if !picked_indices.contains(&color_index) {
                picked_indices.push(color_index);
                break;
            }
        }
    }

    centroids.extend_from_slice(bytemuck::cast_slice(
        &picked_indices
            .into_iter()
            .flat_map(|color_index| {
                let x = color_index % width;
                let y = color_index / width;
                let pixel = image.get_pixel(x, y);
                match color_space {
                    ColorSpace::Lab => {
                        let lab: Lab = Srgba::new(pixel[0], pixel[1], pixel[2], pixel[3])
                            .into_format::<_, f32>()
                            .into_color();
                        [lab.l, lab.a, lab.b, 1.0]
                    }
                    ColorSpace::Rgb => pixel.map(|component| component as f32 / 255.0),
                }
            })
            .collect::<Vec<f32>>(),
    ));

    centroids
}
