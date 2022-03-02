use std::{mem::size_of, vec};

use anyhow::Result;
use image::{ImageBuffer, Rgb};
use pollster::FutureExt;
use rand::Rng;
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BufferAddress, BufferDescriptor,
    BufferUsages,
};

fn main() -> Result<()> {
    let image = image::load_from_memory(include_bytes!("landscape_small.jpg"))?.to_rgb8();

    let (width, height) = image.dimensions();

    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptionsBase {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .block_on()
        .ok_or(anyhow::anyhow!("Couldn't create the adapter"))?;
    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .block_on()?;

    let k = 7;

    let pixels = pixels(&image);
    let centroids = init_centroids(&image, k);

    let pixel_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&pixels),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let centroid_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&centroids),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });

    let size = (size_of::<u32>() * pixels.len()) as BufferAddress;
    let calculated_buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
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

    let choose_centroid_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("Find centroid shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/choose_centroid.wgsl").into()),
    });

    let choose_centroid_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Find centroid pipeline"),
            layout: None,
            module: &&choose_centroid_shader,
            entry_point: "main",
        });

    let find_centroid_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &find_centroid_pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: centroid_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: pixel_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: calculated_buffer.as_entire_binding(),
            },
        ],
    });

    let choose_centroid_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &choose_centroid_pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: centroid_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: pixel_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: calculated_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let find_centroid_dispatch_with = compute_work_group_count(pixels[0], 256);
        let choose_centroid_dispatch_with = compute_work_group_count(k, 16);
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Kmean pass"),
        });
        for _ in 0..1 {
            compute_pass.set_pipeline(&find_centroid_pipeline);
            compute_pass.set_bind_group(0, &find_centroid_bind_group, &[]);
            compute_pass.dispatch(find_centroid_dispatch_with, 1, 1);

            compute_pass.set_pipeline(&choose_centroid_pipeline);
            compute_pass.set_bind_group(0, &choose_centroid_bind_group, &[]);
            compute_pass.dispatch(choose_centroid_dispatch_with, 1, 1);
        }
    }

    encoder.copy_buffer_to_buffer(&calculated_buffer, 0, &staging_buffer, 0, size);

    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    device.poll(wgpu::Maintain::Wait);

    if let Ok(()) = buffer_future.block_on() {
        let data = buffer_slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        let out_pixels: Vec<u8> = result
            .into_iter()
            .flat_map(|color| {
                [
                    (color >> 16) as u8 & 0xff,
                    (color >> 8) as u8 & 0xff,
                    (color >> 0) as u8 & 0xff,
                ]
            })
            .collect();

        if let Some(output_image) =
            image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(width, height, &out_pixels[..])
        {
            output_image.save("kmean.png")?;
        }
    }

    Ok(())
}

fn pixels(image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Vec<u32> {
    let (width, height) = image.dimensions();
    let mut pixels = vec![0; 1 + (width * height) as usize];
    pixels[0] = width * height;

    let packed: Vec<u32> = image
        .pixels()
        .map(|Rgb(pixel)| (pixel[0] as u32) << 16 | (pixel[1] as u32) << 8 | pixel[2] as u32)
        .collect();

    pixels[1..].copy_from_slice(&packed);

    packed
}

fn init_centroids(image: &ImageBuffer<Rgb<u8>, Vec<u8>>, k: u32) -> Vec<u32> {
    let mut rng = rand::thread_rng();

    let (width, height) = image.dimensions();
    let total_px = width * height;
    let mut centroids = vec![0; 1 + k as usize];
    centroids[0] = k;
    for index in 1..=k {
        let color_index = rng.gen_range(0..total_px);
        let x = color_index % width;
        let y = color_index / width;
        let Rgb(pixel) = image.get_pixel(x, y);
        centroids[index as usize] =
            (pixel[0] as u32) << 16 | (pixel[1] as u32) << 8 | pixel[2] as u32;
    }

    centroids
}

fn compute_work_group_count(width: u32, workgroup_width: u32) -> u32 {
    let x = (width + workgroup_width - 1) / workgroup_width;
    x
}
