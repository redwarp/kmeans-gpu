use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use clap::{command, Arg, ArgMatches, Command};
use image::{ImageBuffer, Rgba};
use k_means_gpu::{kmeans, palette, ColorSpace, Image};
use palette::Srgb;

fn main() -> Result<()> {
    let matches = command!()
    .subcommand(Command::new("kmeans")
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
                .short('e')
                .help("Optional extension to use if an output file is not specified")
                .takes_value(true)
                .default_value("png")
                .possible_values(["png", "jpg"]),
        )
        .arg(
            Arg::new("colorspace")
                .long("colorspace")
                .short('c')
                .help("The colorspace to use when clustering colors. Lab gives more natural colors")
                .takes_value(true)
                .possible_values(["rgb", "lab"])
                .default_value("lab"),
        ))
        .subcommand(
            Command::new("palette").about("Output the palette calculated with k-means")
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
                    Arg::new("colorspace")
                        .long("colorspace")
                        .short('c')
                        .help("The colorspace to use when clustering colors. Lab gives more natural colors")
                        .takes_value(true)
                        .possible_values(["rgb", "lab"])
                        .default_value("lab"),
                ),
        )
        .subcommand_required(true)
        .get_matches();

    match matches.subcommand() {
        Some(("palette", sub_matches)) => palette_subcommand(sub_matches)?,
        Some(("kmeans", sub_matches)) => kmeans_subcommand(sub_matches)?,
        _ => {}
    };

    Ok(())
}

fn kmeans_subcommand(matches: &ArgMatches) -> Result<()> {
    let k = matches.value_of("k").expect("Has default value").parse()?;
    let input = matches.value_of("input").expect("Required argument");
    let extension = matches.value_of("extension");
    let color_space = ColorSpace::from(matches.value_of("colorspace").expect("Default value"))
        .expect("Values restricted to supported ones");

    let image = Image::open(input)?;

    let result = kmeans(k, &image, &color_space)?;
    let (width, height) = result.dimensions();

    if let Some(output_image) =
        ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, result.into_raw_pixels())
    {
        let output_file = output_file(None, input, extension, k, &color_space)?;
        output_image.save(output_file)?;
    }

    Ok(())
}

fn palette_subcommand(matches: &ArgMatches) -> Result<()> {
    let k: u32 = matches.value_of("k").expect("Has default value").parse()?;
    let input = matches.value_of("input").expect("Required argument");
    let color_space = ColorSpace::from(matches.value_of("colorspace").expect("Default value"))
        .expect("Values restricted to supported ones");

    let image = Image::open(input)?;

    let result = palette(k, &image, &color_space)?;

    let colors = result
        .into_iter()
        .map(|color| format!("#{:02X}{:02X}{:02X}", color[0], color[1], color[2]))
        .collect::<Vec<_>>()
        .join(", ");

    println!("Palette: {colors}");

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

trait Openable: Sized {
    fn open<P>(path: P) -> Result<Self>
    where
        P: AsRef<Path>;
}

impl Openable for Image {
    fn open<P>(path: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let image = image::open(path)?.to_rgba8();
        let dimensions = image.dimensions();
        let image = Image::from_raw_pixels(dimensions, &image.into_raw());

        Ok(image)
    }
}
