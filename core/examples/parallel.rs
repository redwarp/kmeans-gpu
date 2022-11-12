use std::{fs::File, path::Path, sync::Arc, thread, time::Instant};

use gif::{Frame, Repeat};
use kmeans_color_gpu::{image::copied_pixel, Algorithm, ImageProcessor};
use pollster::FutureExt;

fn main() {
    let start = Instant::now();
    let parent = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("gfx");
    let path = parent.join("turtles.png");

    let image = image::open(&path).unwrap().to_rgba8();
    let dimensions = image.dimensions();

    let image = Arc::new(copied_pixel(
        dimensions,
        bytemuck::cast_slice(image.as_raw()),
    ));

    let image_processor = Arc::new(ImageProcessor::new().block_on().unwrap());

    let width = dimensions.0 as u16;
    let height = dimensions.1 as u16;

    let output_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("gfx/turtles.gif");
    let mut gif = File::create(&output_path).unwrap();
    let mut gif_encoder = gif::Encoder::new(&mut gif, width, height, &[]).unwrap();
    gif_encoder.set_repeat(Repeat::Infinite).unwrap();

    let handles: Vec<_> = (2..16)
        .map(|color_count| {
            let image_processor = image_processor.clone();
            let image = image.clone();
            thread::spawn(move || {
                image_processor
                    .reduce(
                        color_count,
                        &image,
                        &Algorithm::Kmeans,
                        &kmeans_color_gpu::ReduceMode::Replace,
                    )
                    .block_on()
            })
        })
        .collect();

    for handle in handles {
        let reduced = handle.join().unwrap().unwrap();

        let mut frame = Frame::from_rgba(width, height, &mut reduced.into_raw_pixels());
        frame.delay = 100;

        gif_encoder.write_frame(&frame).unwrap();
    }

    let duration = start.elapsed();

    println!("Time elapsed in creating gif is: {:?}", duration);
}
