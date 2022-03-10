use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use clap::{command, Arg};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba};
use k_means_gpu::{kmeans, Image};

fn main() -> Result<()> {
    let matches = command!()
        .arg(
            Arg::new("k")
                .short('k')
                .help("K value, aka the number of colors we want to extract")
                .takes_value(true)
                .default_value("8")
                .validator(|input| input.parse::<u32>()),
        )
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .help("Input file, required")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Optional output file")
                .required(false)
                .takes_value(true)
                .validator(|input| {
                    if input.ends_with(".png") || input.ends_with(".jpg") {
                        Ok(())
                    } else {
                        Err(String::from("Filters only support png or jpg files"))
                    }
                }),
        )
        .arg(
            Arg::new("extension")
                .long("ext")
                .help("Optional extension to use if an output file is not specified")
                .takes_value(true)
                .default_value("png")
                .possible_values(["png", "jpg"]),
        )
        .get_matches();

    let k = matches.value_of("k").expect("Has default value").parse()?;
    let input = matches.value_of("input").expect("Required argument");
    let extension = matches.value_of("extension");
    let image = image::open(input)?.to_rgb8();
    let dimensions = image.dimensions();
    let pixels = image.into_raw()[..]
        .chunks_exact(3)
        .map(|rgb| {
            [
                rgb[0] as f32 / 255.0,
                rgb[1] as f32 / 255.0,
                rgb[1] as f32 / 255.0,
                1.0,
            ]
        })
        .collect::<Vec<[f32; 4]>>();
    let image = Image::new(dimensions, pixels);

    let result = kmeans(k, &image)?;
    let (width, height) = result.dimensions();
    let pixels: Vec<_> = result.into_raw_pixels()[..]
        .chunks_exact(4)
        .map(|rgba| {
            [
                (rgba[0] * 255.0) as u8,
                (rgba[1] * 255.0) as u8,
                (rgba[2] * 255.0) as u8,
            ]
        })
        .flatten()
        .collect();
    if let Some(output_image) = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, pixels) {
        let output_file = output_file(None, input, extension, k)?;
        output_image.save(output_file)?;
    }

    Ok(())
}

fn output_file(
    output: Option<&str>,
    input: &str,
    extension: Option<&str>,
    k: u32,
) -> Result<PathBuf> {
    if let Some(output) = output {
        Ok(Path::new(output).to_owned())
    } else {
        let path = Path::new(input);
        let parent = path.parent();
        let stem = path
            .file_stem()
            .expect("Expecting .jpg or .png files")
            .to_string_lossy();
        let extension = if let Some(extension) = extension {
            Cow::Borrowed(extension)
        } else {
            path.extension()
                .expect("Expecting .jpg or .png files")
                .to_string_lossy()
        };

        let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
        let millis = format!("{}{:03}", now.as_secs(), now.subsec_millis());

        let filename = format!("{stem}-{millis}-{k}.{extension}");
        let output_path = if let Some(parent) = parent {
            parent.join(filename)
        } else {
            Path::new(&filename).to_path_buf()
        };

        Ok(output_path)
    }
}
