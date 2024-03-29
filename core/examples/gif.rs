use std::{fs::File, path::Path, time::Instant};

use gif::{Frame, Repeat};
use kmeans_color_gpu::{image::Image, Algorithm, ImageProcessor, ReduceMode};
use pollster::FutureExt;

fn main() {
    let start = Instant::now();
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("gfx/turtles.png");

    let image = image::open(&path).unwrap().to_rgba8();
    let dimensions = image.dimensions();

    let image = Image::new(dimensions, bytemuck::cast_slice(image.as_raw()));

    let image_processor = ImageProcessor::new().block_on().unwrap();

    let width = dimensions.0 as u16;
    let height = dimensions.1 as u16;

    let output_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("gfx/turtles.gif");
    let mut gif = File::create(&output_path).unwrap();
    let mut gif_encoder = gif::Encoder::new(&mut gif, width, height, &[]).unwrap();
    gif_encoder.set_repeat(Repeat::Infinite).unwrap();

    for c in 2..16 {
        let reduced = image_processor
            .reduce(c, &image, &Algorithm::Kmeans, &ReduceMode::Replace)
            .block_on()
            .unwrap();

        let mut frame = Frame::from_rgba(width, height, &mut reduced.into_raw_pixels());
        frame.delay = 100;

        gif_encoder.write_frame(&frame).unwrap();
    }

    let duration = start.elapsed();

    println!("Time elapsed in creating gif is: {:?}", duration);
}
