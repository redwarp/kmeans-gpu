use palette::{IntoColor, Lab, Pixel, Srgb};
use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use clap::{command, Arg};
use image::{ImageBuffer, Rgb};
use k_means_gpu::{kmeans, ColorSpace, Image};

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
        .arg(
            Arg::new("colorspace")
                .long("colorspace")
                .short('c')
                .takes_value(true)
                .possible_values(["rgb", "lab"])
                .default_value("lab"),
        )
        .get_matches();

    let k = matches.value_of("k").expect("Has default value").parse()?;
    let input = matches.value_of("input").expect("Required argument");
    let extension = matches.value_of("extension");
    let color_space = ColorSpace::from(matches.value_of("colorspace").expect("Default value"))
        .expect("Values restricted to supported ones");

    let image = image::open(input)?.to_rgb8();
    let dimensions = image.dimensions();
    let pixels = match color_space {
        ColorSpace::Lab => Srgb::from_raw_slice(&image.into_raw())
            .into_iter()
            .map(|rgb| {
                let lab = IntoColor::<Lab>::into_color(rgb.into_format::<f32>());
                [lab.l, lab.a, lab.b, 1.0]
            })
            .collect(),
        ColorSpace::Rgb => image.into_raw()[..]
            .chunks_exact(3)
            .map(|rgb| {
                [
                    rgb[0] as f32 / 255.0,
                    rgb[1] as f32 / 255.0,
                    rgb[2] as f32 / 255.0,
                    1.0,
                ]
            })
            .collect::<Vec<[f32; 4]>>(),
    };

    let image = Image::new(dimensions, pixels);

    let result = kmeans(k, &image, &color_space)?;
    let (width, height) = result.dimensions();
    let rgb: Vec<u8> = match color_space {
        ColorSpace::Lab => result.into_raw_pixels()[..]
            .chunks_exact(4)
            .map(|lab| {
                IntoColor::<Srgb>::into_color(Lab::new(lab[0], lab[1], lab[2]))
                    .into_format()
                    .into_raw::<[u8; 3]>()
            })
            .flatten()
            .collect(),
        ColorSpace::Rgb => result.into_raw_pixels()[..]
            .chunks_exact(4)
            .map(|rgba| {
                [
                    (rgba[0] * 255.0) as u8,
                    (rgba[1] * 255.0) as u8,
                    (rgba[2] * 255.0) as u8,
                ]
            })
            .flatten()
            .collect(),
    };

    if let Some(output_image) = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, rgb) {
        let output_file = output_file(None, input, extension, k, &color_space)?;
        output_image.save(output_file)?;
    }

    Ok(())
}

fn output_file(
    output: Option<&str>,
    input: &str,
    extension: Option<&str>,
    k: u32,
    color_space: &ColorSpace,
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

        let filename = format!(
            "{stem}-{cs}-k{k}-{millis}.{extension}",
            cs = color_space.name()
        );
        let output_path = if let Some(parent) = parent {
            parent.join(filename)
        } else {
            Path::new(&filename).to_path_buf()
        };

        Ok(output_path)
    }
}
