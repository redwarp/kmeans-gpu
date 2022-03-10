use anyhow::Result;
use pollster::FutureExt;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{num::NonZeroU32, vec};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutEntry, BindingResource,
    BindingType, Buffer, BufferAddress, BufferBinding, BufferBindingType, BufferDescriptor,
    BufferUsages, ComputePipelineDescriptor, DeviceDescriptor, Features, ImageDataLayout, Instance,
    PipelineLayoutDescriptor, PowerPreference, QueryType, RequestAdapterOptionsBase, ShaderSource,
    ShaderStages, StorageTextureAccess, TextureDimension, TextureFormat, TextureUsages,
    TextureViewDescriptor, TextureViewDimension,
};

const WORKGROUP_SIZE: u32 = 256;
const N_SEQ: u32 = 24;

pub struct Image {
    pub(crate) dimensions: (u32, u32),
    pub(crate) rgba: Vec<[f32; 4]>,
}

impl Image {
    pub fn new(dimensions: (u32, u32), rgba: Vec<[f32; 4]>) -> Self {
        Self { dimensions, rgba }
    }

    pub fn from_raw_pixels(dimensions: (u32, u32), rbga: &[f32]) -> Self {
        let mut pixels = Vec::with_capacity(dimensions.0 as usize * dimensions.1 as usize);
        pixels.extend_from_slice(bytemuck::cast_slice(rbga));
        Self {
            dimensions,
            rgba: pixels,
        }
    }

    pub fn get_pixel(&self, x: u32, y: u32) -> &[f32; 4] {
        let index = (x + y * self.dimensions.0) as usize;
        &self.rgba[index]
    }

    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    pub fn into_raw_pixels(self) -> Vec<f32> {
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

    let centroids = init_centroids(image, k);

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
        Some(device.create_query_set(&wgpu::QuerySetDescriptor {
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
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    });

    queue.write_texture(
        input_texture.as_image_copy(),
        bytemuck::cast_slice(&image.rgba),
        ImageDataLayout {
            offset: 0,
            bytes_per_row: std::num::NonZeroU32::new(16 * width),
            rows_per_image: None,
        },
        texture_size,
    );

    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
    });

    let centroid_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &centroids,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });

    let index_size = width * height;
    let calculated_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice::<u32, u8>(&vec![k + 1; index_size as usize]),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let find_centroid_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("Find centroid shader"),
        source: ShaderSource::Wgsl(include_str!("shaders/find_centroid.wgsl").into()),
    });

    let find_centroid_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Find centroid bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let find_centroid_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Pipeline layout"),
        bind_group_layouts: &[&find_centroid_bind_group_layout],
        push_constant_ranges: &[],
    });

    let find_centroid_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Find centroid pipeline"),
        layout: Some(&find_centroid_pipeline_layout),
        module: &find_centroid_shader,
        entry_point: "main",
    });

    let find_centroid_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Find centroid bind group"),
        layout: &find_centroid_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&input_texture.create_view(
                    &TextureViewDescriptor {
                        label: None,
                        format: Some(TextureFormat::Rgba32Float),
                        aspect: wgpu::TextureAspect::All,
                        base_mip_level: 0,
                        mip_level_count: NonZeroU32::new(1),
                        dimension: Some(TextureViewDimension::D2),
                        ..Default::default()
                    },
                )),
            },
            BindGroupEntry {
                binding: 1,
                resource: centroid_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: calculated_buffer.as_entire_binding(),
            },
        ],
    });

    let choose_centroid_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("Choose centroid shader"),
        source: ShaderSource::Wgsl(include_str!("shaders/choose_centroid.wgsl").into()),
    });

    let choose_centroid_bind_group_0_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Choose centroid bind group 0 layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
    let choose_centroid_bind_group_0 = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &choose_centroid_bind_group_0_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: centroid_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: calculated_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(
                    &input_texture.create_view(&TextureViewDescriptor::default()),
                ),
            },
        ],
    });

    let mut choose_centroid_settings_content: Vec<u8> = Vec::new();
    choose_centroid_settings_content.extend_from_slice(bytemuck::cast_slice(&[N_SEQ]));
    choose_centroid_settings_content
        .extend_from_slice(bytemuck::cast_slice(&[color_space.convergence()]));
    let choose_centroid_settings_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &choose_centroid_settings_content,
        usage: BufferUsages::UNIFORM,
    });

    let (choose_centroid_dispatch_width, _) = compute_work_group_count(
        (texture_size.width * texture_size.height, 1),
        (WORKGROUP_SIZE * N_SEQ, 1),
    );
    let color_buffer_size = choose_centroid_dispatch_width * 8 * 4;
    let color_buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size: color_buffer_size as BufferAddress,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let state_buffer_size = choose_centroid_dispatch_width;
    let state_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice::<u32, u8>(&vec![0; state_buffer_size as usize]),
        usage: BufferUsages::STORAGE,
    });
    let convergence_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice::<u32, u8>(&vec![0; k as usize + 1]),
        usage: BufferUsages::STORAGE,
    });

    let choose_centroid_bind_group_1_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Choose centroid bind group 1 layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let choose_centroid_bind_group_1 = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &choose_centroid_bind_group_1_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: color_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: state_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: convergence_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: choose_centroid_settings_buffer.as_entire_binding(),
            },
        ],
    });

    let k_index_buffers: Vec<Buffer> = (0..k)
        .map(|k| {
            device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[k]),
                usage: BufferUsages::UNIFORM,
            })
        })
        .collect();

    let k_index_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Choose centroid bind group 2 layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
    let k_index_bind_groups: Vec<_> = (0..k)
        .map(|k| {
            device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &k_index_bind_group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &k_index_buffers[k as usize],
                        offset: 0,
                        size: None,
                    }),
                }],
            })
        })
        .collect();

    let choose_centroid_pipeline_layout =
        device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Choose centroid pipeline layout"),
            bind_group_layouts: &[
                &choose_centroid_bind_group_0_layout,
                &choose_centroid_bind_group_1_layout,
                &k_index_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
    let choose_centroid_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Choose centroid pipeline"),
        layout: Some(&choose_centroid_pipeline_layout),
        module: &choose_centroid_shader,
        entry_point: "main",
    });

    let swap_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("Swap colors shader"),
        source: ShaderSource::Wgsl(include_str!("shaders/swap.wgsl").into()),
    });

    let swap_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Swap bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

    let swap_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &swap_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: centroid_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: calculated_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(
                    &output_texture.create_view(&TextureViewDescriptor::default()),
                ),
            },
        ],
    });

    let swap_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Swap pipeline layout"),
        bind_group_layouts: &[&swap_bind_group_layout],
        push_constant_ranges: &[],
    });
    let swap_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Swap pipeline"),
        layout: Some(&swap_pipeline_layout),
        module: &swap_shader,
        entry_point: "main",
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 0);
    }

    let (dispatch_with, dispatch_height) =
        compute_work_group_count((texture_size.width, texture_size.height), (16, 16));
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Kmean pass"),
        });
        compute_pass.set_pipeline(&find_centroid_pipeline);
        compute_pass.set_bind_group(0, &find_centroid_bind_group, &[]);
        compute_pass.dispatch(dispatch_with, dispatch_height, 1);

        for _ in 0..20 {
            compute_pass.set_pipeline(&choose_centroid_pipeline);
            compute_pass.set_bind_group(0, &choose_centroid_bind_group_0, &[]);
            compute_pass.set_bind_group(1, &choose_centroid_bind_group_1, &[]);
            for i in 0..k {
                compute_pass.set_bind_group(2, &k_index_bind_groups[i as usize], &[]);
                compute_pass.dispatch(choose_centroid_dispatch_width, 1, 1);
            }

            compute_pass.set_pipeline(&find_centroid_pipeline);
            compute_pass.set_bind_group(0, &find_centroid_bind_group, &[]);
            compute_pass.dispatch(dispatch_with, dispatch_height, 1);
        }

        compute_pass.set_pipeline(&swap_pipeline);
        compute_pass.set_bind_group(0, &swap_bind_group, &[]);
        compute_pass.dispatch(dispatch_with, dispatch_height, 1);
    }
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 1);
    }

    let padded_bytes_per_row = padded_bytes_per_row(width);
    let unpadded_bytes_per_row = width as usize * 16;

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
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    let cent_buffer_slice = staging_buffer.slice(..);
    let cent_buffer_future = cent_buffer_slice.map_async(wgpu::MapMode::Read);

    let query_slice = query_buf.slice(..);
    let query_future = query_slice.map_async(wgpu::MapMode::Read);

    device.poll(wgpu::Maintain::Wait);

    if let Ok(()) = cent_buffer_future.block_on() {
        let data = cent_buffer_slice.get_mapped_range();

        for (index, k) in bytemuck::cast_slice::<u8, f32>(&data[4..])
            .chunks(4)
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
            let mut pixels: Vec<f32> = Vec::with_capacity(4 * width as usize * height as usize);
            for padded in padded_data.chunks_exact(padded_bytes_per_row) {
                pixels.extend_from_slice(bytemuck::cast_slice(&padded[..unpadded_bytes_per_row]));
            }

            let result = Image::from_raw_pixels((width, height), &pixels);

            Ok(result)
        }
        Err(e) => Err(e.into()),
    }
}

fn init_centroids(image: &Image, k: u32) -> Vec<u8> {
    let mut centroids: Vec<u8> = vec![];
    centroids.extend_from_slice(bytemuck::cast_slice(&[k]));

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
                image.get_pixel(x, y).clone()
            })
            .collect::<Vec<f32>>(),
    ));

    centroids
}

fn compute_work_group_count(
    (width, height): (u32, u32),
    (workgroup_width, workgroup_height): (u32, u32),
) -> (u32, u32) {
    let x = (width + workgroup_width - 1) / workgroup_width;
    let y = (height + workgroup_height - 1) / workgroup_height;

    (x, y)
}

/// Compute the next multiple of 256 for texture retrieval padding.
fn padded_bytes_per_row(width: u32) -> usize {
    let bytes_per_row = width as usize * 16;
    let padding = (256 - bytes_per_row % 256) % 256;
    bytes_per_row + padding
}
