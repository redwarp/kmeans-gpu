use anyhow::Result;
use image::{ImageBuffer, Rgba};
use pollster::FutureExt;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::vec;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferAddress, BufferBinding,
    BufferDescriptor, BufferUsages, Features, TextureViewDescriptor,
};

fn main() -> Result<()> {
    let image = image::load_from_memory(include_bytes!("landscape.jpg"))?.to_rgba8();

    let (width, height) = image.dimensions();

    let k = 16;

    let centroids = init_centroids(&image, k);

    println!("Setting up compute");

    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptionsBase {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .block_on()
        .ok_or(anyhow::anyhow!("Couldn't create the adapter"))?;

    let features = adapter.features();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
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
            ty: wgpu::QueryType::Timestamp,
            label: None,
        }))
    } else {
        None
    };
    let query_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &[0; 16],
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    });
    queue.write_texture(
        input_texture.as_image_copy(),
        bytemuck::cast_slice(image.as_raw()),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: std::num::NonZeroU32::new(4 * width),
            rows_per_image: None,
        },
        texture_size,
    );

    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
    });

    let centroid_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &centroids,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });

    let size = next_multiple_of(256 * 8, width * height) as usize;
    let calculated_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice::<u32, u8>(&vec![k + 1; size]),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let find_centroid_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("Find centroid shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/find_centroid.wgsl").into()),
    });

    let find_centroid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Find centroid pipeline"),
        layout: None,
        module: &find_centroid_shader,
        entry_point: "main",
    });

    let find_centroid_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &find_centroid_pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(
                    &input_texture.create_view(&TextureViewDescriptor::default()),
                ),
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
        label: Some("Find centroid shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/choose_centroid.wgsl").into()),
    });

    let choose_centroid_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Find centroid pipeline"),
            layout: None,
            module: &choose_centroid_shader,
            entry_point: "main",
        });

    let choose_centroid_bind_group_0 = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &choose_centroid_pipeline.get_bind_group_layout(0),
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

    let color_buffer_size = size / (256 * 8) * 8 * 4;
    let color_buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size: color_buffer_size as BufferAddress,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let state_buffer_size = size / (256 * 8);
    let state_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice::<u32, u8>(&vec![0; state_buffer_size]),
        usage: BufferUsages::STORAGE,
    });
    let choose_centroid_bind_group_1 = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &choose_centroid_pipeline.get_bind_group_layout(1),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: color_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: state_buffer.as_entire_binding(),
            },
        ],
    });

    let settings_buffer: Vec<Buffer> = (0..k)
        .map(|k| {
            device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[k]),
                usage: BufferUsages::UNIFORM,
            })
        })
        .collect();
    // let settings_buffer = device.create_buffer_init(&BufferInitDescriptor {
    //     label: None,
    //     contents: bytemuck::cast_slice(&(0..k).collect::<Vec<_>>()),
    //     usage: BufferUsages::STORAGE,
    // });

    let settings_bind_groups: Vec<_> = (0..k)
        .map(|k| {
            device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &choose_centroid_pipeline.get_bind_group_layout(2),
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &settings_buffer[k as usize],
                        offset: 0,
                        size: None,
                    }),
                }],
            })
        })
        .collect();

    let swap_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("Swap colors shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/swap.wgsl").into()),
    });

    let swap_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Swap pipeline"),
        layout: None,
        module: &swap_shader,
        entry_point: "main",
    });

    let swap_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &swap_pipeline.get_bind_group_layout(0),
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

        for _ in 0..30 {
            compute_pass.set_pipeline(&choose_centroid_pipeline);
            compute_pass.set_bind_group(0, &choose_centroid_bind_group_0, &[]);
            compute_pass.set_bind_group(1, &choose_centroid_bind_group_1, &[]);
            for i in 0..k {
                compute_pass.set_bind_group(2, &settings_bind_groups[i as usize], &[]);
                compute_pass.dispatch((size / (256 * 8)) as u32, 1, 1);
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
    let unpadded_bytes_per_row = width as usize * 4;

    let output_buffer_size =
        padded_bytes_per_row as u64 * height as u64 * std::mem::size_of::<u8>() as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let centroid_size = centroids.len() as BufferAddress;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: centroid_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
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

    // Note that we're not calling `.await` here.
    let cent_buffer_slice = staging_buffer.slice(..);
    // Gets the future representing when `staging_buffer` can be read from
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

    if query_future.block_on().is_ok() {
        if features.contains(Features::TIMESTAMP_QUERY) {
            let ts_period = queue.get_timestamp_period();
            let ts_data_raw = &*query_slice.get_mapped_range();
            let ts_data: &[u64] = bytemuck::cast_slice(ts_data_raw);
            println!(
                "Compute shader elapsed: {:?}ms",
                (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
            );
        }
    }

    if let Ok(()) = buffer_future.block_on() {
        println!("We mapped the data back");
        let padded_data = buffer_slice.get_mapped_range();

        let mut pixels: Vec<u8> = vec![0; unpadded_bytes_per_row * height as usize];
        for (padded, pixels) in padded_data
            .chunks_exact(padded_bytes_per_row)
            .zip(pixels.chunks_exact_mut(unpadded_bytes_per_row))
        {
            pixels.copy_from_slice(&padded[..unpadded_bytes_per_row]);
        }

        if let Some(output_image) =
            image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, &pixels[..])
        {
            output_image.save("kmean.png")?;
        }
    }

    Ok(())
}

fn init_centroids(image: &ImageBuffer<Rgba<u8>, Vec<u8>>, k: u32) -> Vec<u8> {
    let mut centroids: Vec<u8> = vec![];
    centroids.extend_from_slice(bytemuck::cast_slice(&[k]));

    let mut rng = StdRng::seed_from_u64(42);

    let (width, height) = image.dimensions();
    let total_px = width * height;
    for _ in 0..k {
        let color_index = rng.gen_range(0..total_px);
        let x = color_index % width;
        let y = color_index / width;
        let Rgba(pixel) = image.get_pixel(x, y);
        centroids.extend_from_slice(bytemuck::cast_slice(&[
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
            pixel[3] as f32 / 255.0,
        ]));
    }

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
    let bytes_per_row = width as usize * 4;
    let padding = (256 - bytes_per_row % 256) % 256;
    bytes_per_row + padding
}

fn next_multiple_of(multiple: u32, value: u32) -> u32 {
    let padding = (multiple - value % multiple) % multiple;
    value + padding
}
