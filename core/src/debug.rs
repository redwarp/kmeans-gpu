use crate::{
    modules::{
        ChooseCentroidLoopModule, ColorConverterModule, FindCentroidModule, Module,
        PlusPlusInitModule,
    },
    CentroidsBuffer, ColorIndexTexture, ColorSpace, Image, InputTexture, WorkTexture,
};
use anyhow::Result;
use wgpu::{
    Backends, CommandEncoderDescriptor, ComputePassDescriptor, DeviceDescriptor, Features,
    Instance, PowerPreference, RequestAdapterOptionsBase,
};

pub async fn debug_plus_plus_init(
    k: u32,
    image: &Image,
    color_space: &ColorSpace,
) -> Result<Vec<[u8; 4]>> {
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
        Ok(results.remove(0))
    } else {
        Err(anyhow::anyhow!(
            "There are {count} unique results after init after {try_count} tries",
            count = results.len()
        ))
    }
}

pub async fn debug_conversion(k: u32, image: &Image, starting_centroids: &[[u8; 4]]) -> Result<()> {
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
    for i in 0..try_count {
        let centroids_buffer =
            CentroidsBuffer::fixed_centroids(starting_centroids, &ColorSpace::Lab, &device);
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

        if i == 0 {
            println!(
                "Dispatch size for choose: {size}",
                size = choose_centroid_module.dispatch_size
            );
        }

        for _ in 0..1 {
            choose_centroid_module.compute(&device, &queue);
            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Find centroid pass"),
                });
                find_centroid_module.dispatch(&mut compute_pass);
            }
            queue.submit(Some(encoder.finish()));
        }

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
