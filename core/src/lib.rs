use anyhow::{anyhow, Result};
use image::{copied_pixel, Container, Image};
use octree::ColorTree;
use palette::{IntoColor, Lab, Pixel, Srgb, Srgba};
use std::{fmt::Display, ops::Deref, str::FromStr, sync::mpsc::channel, time::Instant, vec};
use utils::{compute_work_group_count, padded_bytes_per_row};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    AddressMode, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutEntry,
    BindingResource, BindingType, Buffer, BufferBindingType, BufferUsages, CommandEncoder,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, Device,
    DeviceDescriptor, Extent3d, Features, FilterMode, ImageDataLayout, Instance, MapMode,
    PowerPreference, Queue, RequestAdapterOptionsBase, ShaderSource, ShaderStages,
    StorageTextureAccess, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension,
};

mod modules;
mod octree;
mod operations;
mod utils;

pub mod image;

const MAX_IMAGE_DIMENSION: u32 = 256;

pub struct ImageProcessor {
    device: Device,
    queue: Queue,
    query_time: bool,
}

impl ImageProcessor {
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
            device,
            queue,
            query_time,
        })
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

struct InputTexture {
    texture: Texture,
    dimensions: (u32, u32),
}

impl InputTexture {
    fn new<C: Container>(device: &Device, queue: &Queue, image: &Image<C>) -> Self {
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

        Self {
            texture,
            dimensions: image.dimensions,
        }
    }

    fn shrunk(&self, device: &Device, queue: &Queue) -> Option<InputTexture> {
        let (width, height) = self.dimensions;
        if width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION {
            // Need resize.
            let new_width;
            let new_height;
            if width > height {
                new_width = MAX_IMAGE_DIMENSION;
                new_height =
                    ((height as f32 * MAX_IMAGE_DIMENSION as f32 / width as f32) as u32).max(1);
            } else {
                new_height = MAX_IMAGE_DIMENSION;
                new_width =
                    ((width as f32 * MAX_IMAGE_DIMENSION as f32 / height as f32) as u32).max(1);
            }

            let texture_size = wgpu::Extent3d {
                width: new_width,
                height: new_height,
                depth_or_array_layers: 1,
            };

            let updated_texture = device.create_texture(&TextureDescriptor {
                label: None,
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC
                    | TextureUsages::STORAGE_BINDING,
            });

            let resize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Resize shader"),
                source: ShaderSource::Wgsl(include_str!("shaders/resize.wgsl").into()),
            });

            let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Resize pipeline"),
                layout: None,
                module: &resize_shader,
                entry_point: "main",
            });

            let filter_mode = FilterMode::Linear;

            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: None,
                address_mode_u: AddressMode::ClampToEdge,
                address_mode_v: AddressMode::ClampToEdge,
                address_mode_w: AddressMode::ClampToEdge,
                mag_filter: filter_mode,
                min_filter: filter_mode,
                mipmap_filter: filter_mode,
                ..Default::default()
            });

            let compute_constants = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Compute constants"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&sampler),
                }],
            });

            let texture_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Texture bind group"),
                layout: &pipeline.get_bind_group_layout(1),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(
                            &self.texture.create_view(&TextureViewDescriptor::default()),
                        ),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(
                            &updated_texture.create_view(&TextureViewDescriptor::default()),
                        ),
                    },
                ],
            });

            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });

            {
                let (dispatch_with, dispatch_height) =
                    compute_work_group_count((texture_size.width, texture_size.height), (16, 16));
                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Resize pass"),
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &compute_constants, &[]);
                compute_pass.set_bind_group(1, &texture_bind_group, &[]);
                compute_pass.dispatch_workgroups(dispatch_with, dispatch_height, 1);
            }

            queue.submit(Some(encoder.finish()));

            Some(Self {
                texture: updated_texture,
                dimensions: (new_width, new_height),
            })
        } else {
            None
        }
    }

    fn resized(self, max_size: u32, device: &Device, queue: &Queue) -> InputTexture {
        let (width, height) = self.dimensions;

        let (new_width, new_height) = if width > height {
            (
                max_size,
                ((height as f32 * max_size as f32 / width as f32) as u32).max(1),
            )
        } else {
            (
                ((width as f32 * max_size as f32 / height as f32) as u32).max(1),
                max_size,
            )
        };

        let texture_size = wgpu::Extent3d {
            width: new_width,
            height: new_height,
            depth_or_array_layers: 1,
        };

        let updated_texture = device.create_texture(&TextureDescriptor {
            label: None,
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::STORAGE_BINDING,
        });

        let resize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Resize shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/resize.wgsl").into()),
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Resize pipeline"),
            layout: None,
            module: &resize_shader,
            entry_point: "main",
        });

        let filter_mode = FilterMode::Linear;

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: filter_mode,
            min_filter: filter_mode,
            mipmap_filter: filter_mode,
            ..Default::default()
        });

        let compute_constants = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Compute constants"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Sampler(&sampler),
            }],
        });

        let texture_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Texture bind group"),
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &self.texture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &updated_texture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

        {
            let (dispatch_with, dispatch_height) =
                compute_work_group_count((texture_size.width, texture_size.height), (16, 16));
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Resize pass"),
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &compute_constants, &[]);
            compute_pass.set_bind_group(1, &texture_bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch_with, dispatch_height, 1);
        }

        queue.submit(Some(encoder.finish()));
        Self {
            texture: updated_texture,
            dimensions: (new_width, new_height),
        }
    }

    fn output_buffer(&self, device: &Device, encoder: &mut CommandEncoder) -> OutputBuffer {
        let (width, height) = self.dimensions;
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
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        OutputBuffer {
            buffer,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        }
    }

    fn pull_image(&self, device: &Device, queue: &Queue) -> Result<Image<Vec<[u8; 4]>>> {
        let (width, height) = self.dimensions;
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

                let result = copied_pixel((width, height), &pixels);

                Ok(result)
            }
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e.into()),
        }
    }
}

impl Deref for InputTexture {
    type Target = Texture;

    fn deref(&self) -> &Self::Target {
        &self.texture
    }
}

struct WorkTexture(Texture);

impl WorkTexture {
    fn new(device: &Device, dimensions: (u32, u32)) -> Self {
        let (width, height) = dimensions;
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
    fn new(device: &Device, dimensions: (u32, u32)) -> Self {
        let (width, height) = dimensions;
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
    fn new(device: &Device, dimensions: (u32, u32)) -> Self {
        let (width, height) = dimensions;
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

    fn pull_image(&self, device: &Device, queue: &Queue) -> Result<Image<Vec<[u8; 4]>>> {
        let Extent3d {
            width,
            height,
            depth_or_array_layers: _,
        } = self.texture_size;
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

                let result = copied_pixel((width, height), &pixels);

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

pub fn palette<C: Container>(
    image_processor: &ImageProcessor,
    k: u32,
    image: &Image<C>,
) -> Result<Vec<[u8; 4]>> {
    let input_texture = InputTexture::new(&image_processor.device, &image_processor.queue, image);

    let mut colors = operations::extract_palette(
        &image_processor.device,
        &image_processor.queue,
        &input_texture,
        &ColorSpace::Lab,
        k,
        image_processor.query_time,
    )?
    .pull_values(
        &image_processor.device,
        &image_processor.queue,
        &ColorSpace::Lab,
    )?;

    colors.sort_unstable_by(|a, b| {
        let a: Lab = Srgba::from_raw(a).into_format::<_, f32>().into_color();
        let b: Lab = Srgba::from_raw(b).into_format::<_, f32>().into_color();
        a.l.partial_cmp(&b.l).unwrap()
    });
    Ok(colors)
}

pub fn find<C: Container>(
    image_processor: &ImageProcessor,
    image: &Image<C>,
    colors: &[[u8; 4]],
    reduce_mode: &ReduceMode,
) -> Result<Image<Vec<[u8; 4]>>> {
    let input_texture = InputTexture::new(&image_processor.device, &image_processor.queue, image);
    let centroids_buffer =
        CentroidsBuffer::fixed_centroids(colors, &ColorSpace::Lab, &image_processor.device);

    match reduce_mode {
        ReduceMode::Replace => operations::find_colors(
            &image_processor.device,
            &image_processor.queue,
            &input_texture,
            &ColorSpace::Lab,
            &centroids_buffer,
            image_processor.query_time,
        ),
        ReduceMode::Dither => operations::dither_colors(
            &image_processor.device,
            &image_processor.queue,
            &input_texture,
            &ColorSpace::Lab,
            &centroids_buffer,
            image_processor.query_time,
        ),
        ReduceMode::Meld => operations::meld_colors(
            &image_processor.device,
            &image_processor.queue,
            &input_texture,
            &ColorSpace::Lab,
            &centroids_buffer,
            image_processor.query_time,
        ),
    }?
    .pull_image(&image_processor.device, &image_processor.queue)
}

pub fn reduce<C: Container>(
    image_processor: &ImageProcessor,
    color_count: u32,
    image: &Image<C>,
    reduce_mode: &ReduceMode,
) -> Result<Image<Vec<[u8; 4]>>> {
    let input_texture = InputTexture::new(&image_processor.device, &image_processor.queue, image);

    let centroids_buffer = operations::extract_palette(
        &image_processor.device,
        &image_processor.queue,
        &input_texture,
        &ColorSpace::Lab,
        color_count,
        image_processor.query_time,
    )?;

    match reduce_mode {
        ReduceMode::Replace => operations::find_colors(
            &image_processor.device,
            &image_processor.queue,
            &input_texture,
            &ColorSpace::Lab,
            &centroids_buffer,
            image_processor.query_time,
        ),
        ReduceMode::Dither => operations::dither_colors(
            &image_processor.device,
            &image_processor.queue,
            &input_texture,
            &ColorSpace::Lab,
            &centroids_buffer,
            image_processor.query_time,
        ),
        ReduceMode::Meld => operations::meld_colors(
            &image_processor.device,
            &image_processor.queue,
            &input_texture,
            &ColorSpace::Lab,
            &centroids_buffer,
            image_processor.query_time,
        ),
    }?
    .pull_image(&image_processor.device, &image_processor.queue)
}

pub fn octree_palette<C: Container>(
    image_processor: &ImageProcessor,
    color_count: u32,
    image: &Image<C>,
) -> Result<Vec<[u8; 4]>> {
    const MAX_SIZE: u32 = 128;

    let (width, height) = image.dimensions;
    let resized = if width > MAX_SIZE || height > MAX_SIZE {
        let input_texture = InputTexture::new(
            &image_processor.device,
            &image_processor.queue,
            image,
        )
        .resized(MAX_SIZE, &image_processor.device, &image_processor.queue);
        Some(input_texture.pull_image(&image_processor.device, &image_processor.queue)?)
    } else {
        None
    };

    let pixels: &[[u8; 4]] = if let Some(resized) = &resized {
        &resized.rgba
    } else {
        &image.rgba
    };

    let start = Instant::now();
    let mut tree = ColorTree::new();

    for pixel in pixels {
        tree.add_color(pixel);
    }

    let mut colors = tree.reduce(color_count as usize);
    let duration = start.elapsed();

    println!("Time elapsed in octree() is: {duration:?}");

    colors.sort_unstable_by(|a, b| {
        let a: Lab = Srgba::from_raw(a).into_format::<_, f32>().into_color();
        let b: Lab = Srgba::from_raw(b).into_format::<_, f32>().into_color();
        a.l.partial_cmp(&b.l).unwrap()
    });

    Ok(colors)
}
