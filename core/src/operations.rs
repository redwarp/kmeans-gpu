use anyhow::Result;
use palette::{IntoColor, Lab, Pixel, Srgba};
use rgb::ComponentSlice;
use rgb::RGBA8;
use std::sync::mpsc::channel;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, Device, MapMode,
    QuerySetDescriptor, QueryType, Queue,
};

use crate::{
    modules::{
        ChooseCentroidModule, ColorConverterModule, ColorReverterModule, FindCentroidModule,
        MixColorsModule, MixMode, Module, PlusPlusInitModule, SwapModule,
    },
    octree::ColorTree,
    structures::{ColorIndexTexture, OutputTexture, WorkTexture},
    CentroidsBuffer, ColorSpace, InputTexture,
};

pub(crate) fn extract_palette_kmeans(
    device: &Device,
    queue: &Queue,
    input_texture: &InputTexture,
    color_space: &ColorSpace,
    k: u32,
    query_time: bool,
) -> Result<CentroidsBuffer> {
    let centroids_buffer = CentroidsBuffer::empty_centroids(k, device);

    let query_set = if query_time {
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

    let shrunk = input_texture.shrunk(device, queue);
    let input_texture = if let Some(shrunk) = &shrunk {
        shrunk
    } else {
        input_texture
    };
    let work_texture = WorkTexture::new(device, input_texture.dimensions);
    let color_index_texture = ColorIndexTexture::new(device, input_texture.dimensions);
    let plus_plus_init_module = PlusPlusInitModule::new(
        input_texture.dimensions,
        k,
        &work_texture,
        &centroids_buffer,
    );
    let color_converter_module = ColorConverterModule::new(
        device,
        color_space,
        input_texture.dimensions,
        input_texture,
        &work_texture,
    );
    let find_centroid_module = FindCentroidModule::new(
        device,
        input_texture.dimensions,
        &work_texture,
        &centroids_buffer,
        &color_index_texture,
    );
    let choose_centroid_module = ChooseCentroidModule::new(
        device,
        color_space,
        input_texture.dimensions,
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

    plus_plus_init_module.compute(device, queue);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Init pass"),
        });
        find_centroid_module.dispatch(&mut compute_pass);
    }

    queue.submit(Some(encoder.finish()));

    choose_centroid_module.compute(device, queue);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 1);
    }

    if query_time {
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
                "Compute shader elapsed: {:?}ms [palette]",
                (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
            );
        }
    }

    let mut colors = centroids_buffer.pull_values(device, queue, color_space)?;
    colors.sort_unstable_by(|a, b| {
        let a: Lab = Srgba::from_raw(a.as_slice())
            .into_format::<_, f32>()
            .into_color();
        let b: Lab = Srgba::from_raw(b.as_slice())
            .into_format::<_, f32>()
            .into_color();
        a.l.partial_cmp(&b.l).unwrap()
    });

    Ok(centroids_buffer)
}

pub(crate) fn extract_palette_octree(pixels: &[RGBA8], color_count: u32) -> Result<Vec<RGBA8>> {
    let mut tree = ColorTree::new();
    for pixel in pixels {
        tree.add_color(pixel);
    }

    Ok(tree.reduce(color_count as usize))
}

pub(crate) fn dither_colors(
    device: &Device,
    queue: &Queue,
    input_texture: &InputTexture,
    color_space: &ColorSpace,
    centroids_buffer: &CentroidsBuffer,
    query_time: bool,
) -> Result<OutputTexture> {
    let query_set = if query_time {
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

    let work_texture = WorkTexture::new(device, input_texture.dimensions);
    let dithered_texture = WorkTexture::new(device, input_texture.dimensions);
    let color_index_texture = ColorIndexTexture::new(device, input_texture.dimensions);
    let output_texture = OutputTexture::new(device, input_texture.dimensions);

    let color_converter_module = ColorConverterModule::new(
        device,
        color_space,
        input_texture.dimensions,
        input_texture,
        &work_texture,
    );
    let mix_colors_module = MixColorsModule::new(
        device,
        input_texture.dimensions,
        &work_texture,
        &dithered_texture,
        &color_index_texture,
        centroids_buffer,
        &MixMode::Dither,
    );
    let color_reverter_module = ColorReverterModule::new(
        device,
        color_space,
        input_texture.dimensions,
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
    if query_time {
        if let Ok(Ok(())) = query_receiver.recv() {
            let ts_period = queue.get_timestamp_period();
            let ts_data_raw = &*query_slice.get_mapped_range();
            let ts_data: &[u64] = bytemuck::cast_slice(ts_data_raw);
            println!(
                "Compute shader elapsed: {:?}ms [mix]",
                (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
            );
        }
    }

    Ok(output_texture)
}

pub(crate) fn meld_colors(
    device: &Device,
    queue: &Queue,
    input_texture: &InputTexture,
    color_space: &ColorSpace,
    centroids_buffer: &CentroidsBuffer,
    query_time: bool,
) -> Result<OutputTexture> {
    let query_set = if query_time {
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

    let work_texture = WorkTexture::new(device, input_texture.dimensions);
    let dithered_texture = WorkTexture::new(device, input_texture.dimensions);
    let color_index_texture = ColorIndexTexture::new(device, input_texture.dimensions);
    let output_texture = OutputTexture::new(device, input_texture.dimensions);

    let color_converter_module = ColorConverterModule::new(
        device,
        color_space,
        input_texture.dimensions,
        input_texture,
        &work_texture,
    );
    let mix_colors_module = MixColorsModule::new(
        device,
        input_texture.dimensions,
        &work_texture,
        &dithered_texture,
        &color_index_texture,
        centroids_buffer,
        &MixMode::Meld,
    );
    let color_reverter_module = ColorReverterModule::new(
        device,
        color_space,
        input_texture.dimensions,
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
    if query_time {
        if let Ok(Ok(())) = query_receiver.recv() {
            let ts_period = queue.get_timestamp_period();
            let ts_data_raw = &*query_slice.get_mapped_range();
            let ts_data: &[u64] = bytemuck::cast_slice(ts_data_raw);
            println!(
                "Compute shader elapsed: {:?}ms [mix]",
                (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
            );
        }
    }

    Ok(output_texture)
}

pub(crate) fn find_colors(
    device: &Device,
    queue: &Queue,
    input_texture: &InputTexture,
    color_space: &ColorSpace,
    centroids_buffer: &CentroidsBuffer,
    query_time: bool,
) -> Result<OutputTexture> {
    let query_set = if query_time {
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

    let work_texture = WorkTexture::new(device, input_texture.dimensions);
    let color_index_texture = ColorIndexTexture::new(device, input_texture.dimensions);
    let output_texture = OutputTexture::new(device, input_texture.dimensions);

    let color_converter_module = ColorConverterModule::new(
        device,
        color_space,
        input_texture.dimensions,
        input_texture,
        &work_texture,
    );
    let color_reverter_module = ColorReverterModule::new(
        device,
        color_space,
        input_texture.dimensions,
        &work_texture,
        &output_texture,
    );
    let find_centroid_module = FindCentroidModule::new(
        device,
        input_texture.dimensions,
        &work_texture,
        centroids_buffer,
        &color_index_texture,
    );

    let swap_module = SwapModule::new(
        device,
        input_texture.dimensions,
        &work_texture,
        centroids_buffer,
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
    if query_time {
        if let Ok(Ok(())) = query_receiver.recv() {
            let ts_period = queue.get_timestamp_period();
            let ts_data_raw = &*query_slice.get_mapped_range();
            let ts_data: &[u64] = bytemuck::cast_slice(ts_data_raw);
            println!(
                "Compute shader elapsed: {:?}ms [find]",
                (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
            );
        }
    }

    Ok(output_texture)
}
