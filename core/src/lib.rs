use anyhow::{anyhow, Result};
use modules::{
    ChooseCentroidLoopModule, ChooseCentroidModule, ColorConverterModule, ColorReverterModule,
    FindCentroidModule, MixColorsModule, Module, PlusPlusInitModule, SwapModule,
};
use palette::{IntoColor, Lab, Pixel, Srgb, Srgba};
use std::{fmt::Display, ops::Deref, str::FromStr, sync::mpsc::channel, vec};
use utils::padded_bytes_per_row;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferUsages,
    CommandEncoder, CommandEncoderDescriptor, ComputePassDescriptor, Device, DeviceDescriptor,
    Features, ImageDataLayout, Instance, MapMode, PowerPreference, QuerySetDescriptor, QueryType,
    Queue, RequestAdapterOptionsBase, ShaderStages, StorageTextureAccess, Texture,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureViewDimension,
};

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
pub enum MixMode {
    Dither,
    Meld,
}

impl MixMode {
    fn name(&self) -> &'static str {
        match self {
            MixMode::Dither => "dither",
            MixMode::Meld => "meld",
        }
    }
}

impl FromStr for MixMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "dither" => Ok(MixMode::Dither),
            "meld" => Ok(MixMode::Meld),
            _ => Err(anyhow!("Unsupported mix mode {s}")),
        }
    }
}

impl Display for MixMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

struct InputTexture(Texture);

impl InputTexture {
    fn new(device: &Device, queue: &Queue, image: &Image) -> Self {
        let (width, height) = image.dimensions;
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("input texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        });

        queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(&image.rgba),
            ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4 * width),
                rows_per_image: None,
            },
            texture_size,
        );

        Self(texture)
    }
}

impl Deref for InputTexture {
    type Target = Texture;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct WorkTexture(Texture);

impl WorkTexture {
    fn new(device: &Device, image: &Image) -> Self {
        let (width, height) = image.dimensions;
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("work texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        });

        Self(texture)
    }

    fn texture_2d_layout(binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }
    }

    fn texture_storage_layout(binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba32Float,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        }
    }
}

impl Deref for WorkTexture {
    type Target = Texture;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct ColorIndexTexture(Texture);

impl ColorIndexTexture {
    fn new(device: &Device, image: &Image) -> Self {
        let (width, height) = image.dimensions;
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Color index texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Uint,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        });

        Self(texture)
    }

    fn texture_2d_layout(binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Uint,
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }
    }
}

impl Deref for ColorIndexTexture {
    type Target = Texture;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct OutputTexture {
    texture: Texture,
    texture_size: wgpu::Extent3d,
}

impl OutputTexture {
    fn new(device: &Device, image: &Image) -> Self {
        let (width, height) = image.dimensions;
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("output texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
        });

        Self {
            texture,
            texture_size,
        }
    }

    fn output_buffer(&self, device: &Device, encoder: &mut CommandEncoder) -> OutputBuffer {
        let wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: _,
        } = self.texture_size;
        let padded_bytes_per_row = padded_bytes_per_row(width as u64 * 4) as usize;
        let unpadded_bytes_per_row = width as usize * 4;

        let output_buffer_size =
            padded_bytes_per_row as u64 * height as u64 * std::mem::size_of::<u8>() as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output_buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: self,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: std::num::NonZeroU32::new(padded_bytes_per_row as u32),
                    rows_per_image: std::num::NonZeroU32::new(height),
                },
            },
            self.texture_size,
        );

        OutputBuffer {
            buffer,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        }
    }

    fn pull_image(
        &self,
        (width, height): (u32, u32),
        device: &Device,
        queue: &Queue,
    ) -> Result<Image> {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

        let output_buffer = self.output_buffer(device, &mut encoder);

        queue.submit(Some(encoder.finish()));

        let buffer_slice = output_buffer.slice(..);
        let (buffer_sender, buffer_receiver) = channel();
        buffer_slice.map_async(MapMode::Read, move |v| {
            buffer_sender.send(v).expect("Couldn't send result");
        });

        device.poll(wgpu::Maintain::Wait);

        match buffer_receiver.recv() {
            Ok(Ok(())) => {
                let padded_data = buffer_slice.get_mapped_range();
                let mut pixels: Vec<u8> =
                    vec![0; output_buffer.unpadded_bytes_per_row * height as usize];
                for (padded, pixels) in padded_data
                    .chunks_exact(output_buffer.padded_bytes_per_row)
                    .zip(pixels.chunks_exact_mut(output_buffer.unpadded_bytes_per_row))
                {
                    pixels.copy_from_slice(&padded[..output_buffer.unpadded_bytes_per_row]);
                }

                let result = Image::from_raw_pixels((width, height), &pixels);

                Ok(result)
            }
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e.into()),
        }
    }
}

impl Deref for OutputTexture {
    type Target = Texture;

    fn deref(&self) -> &Self::Target {
        &self.texture
    }
}

struct OutputBuffer {
    buffer: Buffer,
    unpadded_bytes_per_row: usize,
    padded_bytes_per_row: usize,
}

impl Deref for OutputBuffer {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

struct CentroidsBuffer {
    copy_size: u64,
    buffer: Buffer,
}

impl CentroidsBuffer {
    fn empty_centroids(k: u32, device: &Device) -> Self {
        let mut centroids: Vec<u8> = Vec::with_capacity((k as usize + 1) * 4);
        // Aligned 16, see https://www.w3.org/TR/WGSL/#address-space-layout-constraints
        centroids.extend_from_slice(bytemuck::cast_slice(&[k, 0, 0, 0]));

        centroids.extend_from_slice(
            &(0..k)
                .flat_map(|_| [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                .collect::<Vec<u8>>(),
        );

        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &centroids,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let copy_size = centroids.len() as u64;

        Self { copy_size, buffer }
    }

    fn fixed_centroids(colors: &[[u8; 4]], color_space: &ColorSpace, device: &Device) -> Self {
        let mut centroids: Vec<u8> = Vec::with_capacity(16 * (colors.len() + 1));

        // Aligned 16, see https://www.w3.org/TR/WGSL/#address-space-layout-constraints
        centroids.extend_from_slice(bytemuck::cast_slice(&[colors.len() as u32, 0, 0, 0]));

        centroids.extend_from_slice(bytemuck::cast_slice(
            &colors
                .iter()
                .map(|c| match color_space {
                    ColorSpace::Lab => {
                        let lab: Lab = Srgb::new(c[0], c[1], c[2]).into_format().into_color();
                        [lab.l, lab.a, lab.b, 1.0]
                    }
                    ColorSpace::Rgb => {
                        let srgb: Srgb = Srgb::new(c[0], c[1], c[2]).into_format();
                        [srgb.red, srgb.green, srgb.blue, 1.0]
                    }
                })
                .collect::<Vec<[f32; 4]>>(),
        ));

        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &centroids,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let copy_size = centroids.len() as u64;

        Self { copy_size, buffer }
    }

    fn staging_buffer(&self, device: &Device, encoder: &mut CommandEncoder) -> Buffer {
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.copy_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, self.copy_size);

        staging_buffer
    }

    fn layout(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn pull_values(
        &self,
        device: &Device,
        queue: &Queue,
        color_space: &ColorSpace,
    ) -> Result<Vec<[u8; 4]>> {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

        let staging_buffer = self.staging_buffer(device, &mut encoder);

        queue.submit(Some(encoder.finish()));

        let cent_buffer_slice = staging_buffer.slice(..);
        let (cent_sender, cent_receiver) = channel();
        cent_buffer_slice.map_async(MapMode::Read, move |v| {
            cent_sender.send(v).expect("Couldn't send result");
        });

        device.poll(wgpu::Maintain::Wait);

        match cent_receiver.recv() {
            Ok(Ok(())) => {
                let data = cent_buffer_slice.get_mapped_range();

                let colors: Vec<_> = bytemuck::cast_slice::<u8, f32>(&data[16..])
                    .chunks_exact(4)
                    .map(|color| {
                        let raw: [u8; 4] = match color_space {
                            ColorSpace::Lab => IntoColor::<Srgba>::into_color(Lab::new(
                                color[0], color[1], color[2],
                            ))
                            .into_format()
                            .into_raw(),
                            ColorSpace::Rgb => Srgba::new(color[0], color[1], color[2], 1.0)
                                .into_format()
                                .into_raw(),
                        };
                        raw
                    })
                    .collect();
                Ok(colors)
            }
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e.into()),
        }
    }
}

impl Deref for CentroidsBuffer {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

pub async fn kmeans(k: u32, image: &Image, color_space: &ColorSpace) -> Result<Image> {
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

    let centroids_buffer = CentroidsBuffer::empty_centroids(k, &device);

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

    let input_texture = InputTexture::new(&device, &queue, image);
    let work_texture = WorkTexture::new(&device, image);
    let color_index_texture = ColorIndexTexture::new(&device, image);
    let output_texture = OutputTexture::new(&device, image);

    let plus_plus_init_module =
        PlusPlusInitModule::new(image.dimensions, k, &work_texture, &centroids_buffer);
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
        &centroids_buffer,
        &color_index_texture,
    );
    let choose_centroid_module = ChooseCentroidModule::new(
        &device,
        color_space,
        image.dimensions,
        k,
        &work_texture,
        &centroids_buffer,
        &color_index_texture,
        &find_centroid_module,
    );
    let swap_module = SwapModule::new(
        &device,
        image.dimensions,
        &work_texture,
        &centroids_buffer,
        &color_index_texture,
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
    }
    queue.submit(Some(encoder.finish()));

    plus_plus_init_module.compute(&device, &queue);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Init pass"),
        });
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
        encoder.resolve_query_set(query_set, 0..2, &query_buf, 0);
    }
    queue.submit(Some(encoder.finish()));

    let query_slice = query_buf.slice(..);
    let (query_sender, query_receiver) = channel();
    query_slice.map_async(MapMode::Read, move |v| {
        query_sender.send(v).expect("Couldn't send result");
    });

    device.poll(wgpu::Maintain::Wait);

    if features.contains(Features::TIMESTAMP_QUERY) {
        if let Ok(Ok(())) = query_receiver.recv() {
            let ts_period = queue.get_timestamp_period();
            let ts_data_raw = &*query_slice.get_mapped_range();
            let ts_data: &[u64] = bytemuck::cast_slice(ts_data_raw);
            println!(
                "Compute shader elapsed: {:?}ms",
                (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
            );
        }
    }

    output_texture.pull_image(image.dimensions, &device, &queue)
}

pub async fn palette(k: u32, image: &Image, color_space: &ColorSpace) -> Result<Vec<[u8; 4]>> {
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

    let centroids_buffer = CentroidsBuffer::empty_centroids(k, &device);

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

    let input_texture = InputTexture::new(&device, &queue, image);
    let work_texture = WorkTexture::new(&device, image);
    let color_index_texture = ColorIndexTexture::new(&device, image);
    let plus_plus_init_module =
        PlusPlusInitModule::new(image.dimensions, k, &work_texture, &centroids_buffer);
    let color_converter_module = ColorConverterModule::new(
        &device,
        color_space,
        image.dimensions,
        &input_texture,
        &work_texture,
    );
    let find_centroid_module = FindCentroidModule::new(
        &device,
        image.dimensions,
        &work_texture,
        &centroids_buffer,
        &color_index_texture,
    );
    let choose_centroid_module = ChooseCentroidModule::new(
        &device,
        color_space,
        image.dimensions,
        k,
        &work_texture,
        &centroids_buffer,
        &color_index_texture,
        &find_centroid_module,
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
    }
    queue.submit(Some(encoder.finish()));

    plus_plus_init_module.compute(&device, &queue);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Init pass"),
        });
        find_centroid_module.dispatch(&mut compute_pass);
    }

    queue.submit(Some(encoder.finish()));

    choose_centroid_module.compute(&device, &queue);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 1);
    }

    if features.contains(Features::TIMESTAMP_QUERY) {
        if let Some(query_set) = &query_set {
            encoder.resolve_query_set(query_set, 0..2, &query_buf, 0);
        }
        queue.submit(Some(encoder.finish()));

        let query_slice = query_buf.slice(..);
        let (query_sender, query_receiver) = channel();
        query_slice.map_async(MapMode::Read, move |v| {
            query_sender.send(v).expect("Couldn't send result");
        });

        device.poll(wgpu::Maintain::Wait);
        if let Ok(Ok(())) = query_receiver.recv() {
            let ts_period = queue.get_timestamp_period();
            let ts_data_raw = &*query_slice.get_mapped_range();
            let ts_data: &[u64] = bytemuck::cast_slice(ts_data_raw);
            println!(
                "Compute shader elapsed: {:?}ms",
                (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
            );
        }
    }

    let mut colors = centroids_buffer.pull_values(&device, &queue, color_space)?;
    colors.sort_unstable_by(|a, b| {
        let a: Lab = Srgba::from_raw(a).into_format::<_, f32>().into_color();
        let b: Lab = Srgba::from_raw(b).into_format::<_, f32>().into_color();
        a.l.partial_cmp(&b.l).unwrap()
    });
    Ok(colors)
}

pub async fn find(image: &Image, colors: &[[u8; 4]], color_space: &ColorSpace) -> Result<Image> {
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

    let centroids = CentroidsBuffer::fixed_centroids(colors, color_space, &device);

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

    let input_texture = InputTexture::new(&device, &queue, image);
    let work_texture = WorkTexture::new(&device, image);
    let color_index_texture = ColorIndexTexture::new(&device, image);
    let output_texture = OutputTexture::new(&device, image);

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
        &centroids,
        &color_index_texture,
    );

    let swap_module = SwapModule::new(
        &device,
        image.dimensions,
        &work_texture,
        &centroids,
        &color_index_texture,
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
        swap_module.dispatch(&mut compute_pass);
        color_reverter_module.dispatch(&mut compute_pass);
    }
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 1);
    }

    if let Some(query_set) = &query_set {
        encoder.resolve_query_set(query_set, 0..2, &query_buf, 0);
    }
    queue.submit(Some(encoder.finish()));

    let query_slice = query_buf.slice(..);
    let (query_sender, query_receiver) = channel();
    query_slice.map_async(MapMode::Read, move |v| {
        query_sender.send(v).expect("Couldn't send result");
    });

    device.poll(wgpu::Maintain::Wait);
    if features.contains(Features::TIMESTAMP_QUERY) {
        if let Ok(Ok(())) = query_receiver.recv() {
            let ts_period = queue.get_timestamp_period();
            let ts_data_raw = &*query_slice.get_mapped_range();
            let ts_data: &[u64] = bytemuck::cast_slice(ts_data_raw);
            println!(
                "Compute shader elapsed: {:?}ms",
                (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
            );
        }
    }

    output_texture.pull_image(image.dimensions, &device, &queue)
}

pub async fn mix(
    k: u32,
    image: &Image,
    color_space: &ColorSpace,
    mix_mode: &MixMode,
) -> Result<Image> {
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

    let centroids_buffer = CentroidsBuffer::empty_centroids(k, &device);

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

    let input_texture = InputTexture::new(&device, &queue, image);
    let work_texture = WorkTexture::new(&device, image);
    let dithered_texture = WorkTexture::new(&device, image);
    let color_index_texture = ColorIndexTexture::new(&device, image);
    let output_texture = OutputTexture::new(&device, image);

    let plus_plus_init_module =
        PlusPlusInitModule::new(image.dimensions, k, &work_texture, &centroids_buffer);
    let color_converter_module = ColorConverterModule::new(
        &device,
        color_space,
        image.dimensions,
        &input_texture,
        &work_texture,
    );
    let find_centroid_module = FindCentroidModule::new(
        &device,
        image.dimensions,
        &work_texture,
        &centroids_buffer,
        &color_index_texture,
    );
    let choose_centroid_module = ChooseCentroidModule::new(
        &device,
        color_space,
        image.dimensions,
        k,
        &work_texture,
        &centroids_buffer,
        &color_index_texture,
        &find_centroid_module,
    );
    let mix_colors_module = MixColorsModule::new(
        &device,
        image.dimensions,
        &work_texture,
        &dithered_texture,
        &color_index_texture,
        &centroids_buffer,
        mix_mode,
    );
    let color_reverter_module = ColorReverterModule::new(
        &device,
        color_space,
        image.dimensions,
        &dithered_texture,
        &output_texture,
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
    }
    queue.submit(Some(encoder.finish()));

    plus_plus_init_module.compute(&device, &queue);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Init pass"),
        });
        find_centroid_module.dispatch(&mut compute_pass);
    }

    queue.submit(Some(encoder.finish()));

    choose_centroid_module.compute(&device, &queue);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Swap and fetch result pass"),
        });
        mix_colors_module.dispatch(&mut compute_pass);
        color_reverter_module.dispatch(&mut compute_pass);
    }
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 1);
    }

    if let Some(query_set) = &query_set {
        encoder.resolve_query_set(query_set, 0..2, &query_buf, 0);
    }
    queue.submit(Some(encoder.finish()));

    let query_slice = query_buf.slice(..);
    let (query_sender, query_receiver) = channel();
    query_slice.map_async(MapMode::Read, move |v| {
        query_sender.send(v).expect("Couldn't send result");
    });

    device.poll(wgpu::Maintain::Wait);
    if features.contains(Features::TIMESTAMP_QUERY) {
        if let Ok(Ok(())) = query_receiver.recv() {
            let ts_period = queue.get_timestamp_period();
            let ts_data_raw = &*query_slice.get_mapped_range();
            let ts_data: &[u64] = bytemuck::cast_slice(ts_data_raw);
            println!(
                "Compute shader elapsed: {:?}ms",
                (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
            );
        }
    }

    output_texture.pull_image(image.dimensions, &device, &queue)
}

pub async fn debug_plus_plus_init(k: u32, image: &Image, color_space: &ColorSpace) -> Result<()> {
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

    let input_texture = InputTexture::new(&device, &queue, image);
    let work_texture = WorkTexture::new(&device, image);
    let color_converter_module = ColorConverterModule::new(
        &device,
        color_space,
        image.dimensions,
        &input_texture,
        &work_texture,
    );

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Init pass"),
        });
        color_converter_module.dispatch(&mut compute_pass);
    }
    queue.submit(Some(encoder.finish()));

    let mut results = vec![];

    let try_count = 100;
    for _ in 0..try_count {
        let centroids_buffer = CentroidsBuffer::empty_centroids(k, &device);

        let plus_plus_init_module =
            PlusPlusInitModule::new(image.dimensions, k, &work_texture, &centroids_buffer);
        plus_plus_init_module.compute(&device, &queue);

        let centroids = centroids_buffer.pull_values(&device, &queue, color_space)?;

        results.push(centroids);
    }

    results.sort();
    results.dedup();

    println!(
        "There are {count} unique results after init after {try_count} tries",
        count = results.len()
    );

    if results.len() == 1 {
        let colors = results[0]
            .iter()
            .map(|color| format!("#{:02X}{:02X}{:02X}", color[0], color[1], color[2]))
            .collect::<Vec<_>>()
            .join(",");

        println!("Colors: {colors}");
    }

    Ok(())
}

pub async fn debug_conversion(k: u32, image: &Image) -> Result<()> {
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

    let input_texture = InputTexture::new(&device, &queue, image);
    let work_texture = WorkTexture::new(&device, image);
    let color_converter_module = ColorConverterModule::new(
        &device,
        &ColorSpace::Lab,
        image.dimensions,
        &input_texture,
        &work_texture,
    );

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Init pass"),
        });
        color_converter_module.dispatch(&mut compute_pass);
    }
    queue.submit(Some(encoder.finish()));

    let mut results: Vec<Vec<[u8; 4]>> = vec![];

    let try_count = 1000;
    for _ in 0..try_count {
        let centroids_buffer = CentroidsBuffer::fixed_centroids(
            &[
                [42, 18, 19, 255],
                [253, 250, 56, 255],
                [49, 49, 245, 255],
                [255, 3, 1, 255],
                [83, 251, 250, 255],
                [241, 175, 235, 255],
                [0, 112, 10, 255],
                [201, 136, 75, 255],
                [30, 66, 157, 255],
                [122, 253, 113, 255],
                [59, 121, 143, 255],
                [169, 33, 65, 255],
                [230, 249, 180, 255],
                [89, 95, 53, 255],
                [228, 241, 253, 255],
                [143, 146, 0, 255],
                [254, 126, 12, 255],
                [17, 154, 121, 255],
                [78, 169, 252, 255],
                [152, 103, 115, 255],
                [103, 44, 3, 255],
                [2, 239, 176, 255],
                [0, 2, 55, 255],
                [255, 191, 54, 255],
                [255, 102, 84, 255],
                [253, 185, 178, 255],
                [25, 15, 152, 255],
                [79, 101, 249, 255],
                [159, 174, 147, 255],
                [0, 45, 33, 255],
                [113, 190, 104, 255],
                [160, 0, 0, 255],
            ],
            &ColorSpace::Lab,
            &device,
        );
        let color_index_texture = ColorIndexTexture::new(&device, image);

        let find_centroid_module = FindCentroidModule::new(
            &device,
            image.dimensions,
            &work_texture,
            &centroids_buffer,
            &color_index_texture,
        );
        let choose_centroid_module = ChooseCentroidLoopModule::new(
            &device,
            &ColorSpace::Lab,
            image.dimensions,
            k,
            &work_texture,
            &centroids_buffer,
            &color_index_texture,
        );

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Find centroid pass"),
            });
            find_centroid_module.dispatch(&mut compute_pass);
        }
        queue.submit(Some(encoder.finish()));

        choose_centroid_module.compute(&device, &queue);

        let centroids = centroids_buffer.pull_values(&device, &queue, &ColorSpace::Lab)?;

        results.push(centroids);
    }

    results.sort();
    results.dedup();

    println!(
        "There are {count} unique results after choosing centroids with {try_count} tries",
        count = results.len()
    );

    if results.len() == 1 {
        let colors = results[0]
            .iter()
            .map(|color| format!("#{:02X}{:02X}{:02X}", color[0], color[1], color[2]))
            .collect::<Vec<_>>()
            .join(",");

        println!("Colors: {colors}");
    }

    Ok(())
}
