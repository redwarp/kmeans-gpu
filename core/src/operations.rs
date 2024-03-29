use anyhow::Result;
use rgb::RGBA8;
use wgpu::{CommandEncoderDescriptor, ComputePassDescriptor, Device, Queue};

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
) -> Result<CentroidsBuffer> {
    let centroids_buffer = CentroidsBuffer::empty_centroids(k, device);

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
) -> Result<OutputTexture> {
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
    queue.submit(Some(encoder.finish()));

    Ok(output_texture)
}

pub(crate) fn meld_colors(
    device: &Device,
    queue: &Queue,
    input_texture: &InputTexture,
    color_space: &ColorSpace,
    centroids_buffer: &CentroidsBuffer,
) -> Result<OutputTexture> {
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
    queue.submit(Some(encoder.finish()));

    Ok(output_texture)
}

pub(crate) fn find_colors(
    device: &Device,
    queue: &Queue,
    input_texture: &InputTexture,
    color_space: &ColorSpace,
    centroids_buffer: &CentroidsBuffer,
) -> Result<OutputTexture> {
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

    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Init pass"),
        });
        color_converter_module.dispatch(&mut compute_pass);
        find_centroid_module.dispatch(&mut compute_pass);
        swap_module.dispatch(&mut compute_pass);
        color_reverter_module.dispatch(&mut compute_pass);
    }

    queue.submit(Some(encoder.finish()));

    Ok(output_texture)
}
