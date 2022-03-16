use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Ok, Result};
use args::{Cli, Commands, Extension};
use clap::Parser;
use image::{ImageBuffer, Rgba};
use k_means_gpu::{find, kmeans, palette, ColorSpace, Image};

mod args;

fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match cli.commands {
        Commands::Kmeans {
            k,
            input,
            output,
            extension,
            color_space,
        } => kmeans_subcommand(k, input, output, extension, color_space),
        Commands::Palette {
            k,
            input,
            output,
            color_space,
        } => palette_subcommand(k, input, output, color_space),
        Commands::Find {
            input,
            output,
            replacement,
            color_space,
        } => find_subcommand(input, output, replacement, color_space),
    }?;

    Ok(())
}

fn kmeans_subcommand(
    k: u32,
    input: PathBuf,
    output: Option<PathBuf>,
    extension: Option<Extension>,
    color_space: ColorSpace,
) -> Result<()> {
    let image = Image::open(&input)?;

    let result = kmeans(k, &image, &color_space)?;
    let (width, height) = result.dimensions();

    if let Some(output_image) =
        ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, result.into_raw_pixels())
    {
        let output_file = output_file(k, &output, &input, &extension, &color_space)?;
        output_image.save(output_file)?;
    }

    Ok(())
}

fn palette_subcommand(
    k: u32,
    input: PathBuf,
    output: Option<PathBuf>,
    color_space: ColorSpace,
) -> Result<()> {
    let image = Image::open(&input)?;

    let result = palette(k, &image, &color_space)?;

    let path = palette_file(k, &input, &output, &color_space)?;
    save_palette(path, &result)?;

    let colors = result
        .into_iter()
        .map(|color| format!("#{:02X}{:02X}{:02X}", color[0], color[1], color[2]))
        .collect::<Vec<_>>()
        .join(",");

    println!("Palette: {colors}");

    Ok(())
}

fn find_subcommand(
    input: PathBuf,
    output: Option<PathBuf>,
    replacement: String,
    color_space: ColorSpace,
) -> Result<()> {
    let colors = parse_colors(&replacement)?;

    let image = Image::open(&input)?;

    let result = find(&image, &colors, &color_space)?;

    let (width, height) = result.dimensions();

    if let Some(output_image) =
        ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, result.into_raw_pixels())
    {
        let output_file = find_file(&output, &input, &None, &color_space)?;
        output_image.save(output_file)?;
    }

    Ok(())
}

fn output_file(
    k: u32,
    output: &Option<PathBuf>,
    input: &Path,
    extension: &Option<Extension>,
    color_space: &ColorSpace,
) -> Result<PathBuf> {
    if let Some(output) = output {
        Ok(output.clone())
    } else {
        let parent = input.parent();
        let stem = input
            .file_stem()
            .expect("Expecting .jpg or .png files")
            .to_string_lossy();
        let extension = if let Some(extension) = extension {
            Cow::Borrowed(extension.value())
        } else {
            input
                .extension()
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

fn palette_file(
    k: u32,
    input: &Path,
    output: &Option<PathBuf>,
    color_space: &ColorSpace,
) -> Result<PathBuf> {
    if let Some(output) = output {
        return Ok(output.clone());
    }

    let path = Path::new(input);
    let parent = path.parent();
    let stem = path
        .file_stem()
        .expect("Expecting .jpg or .png files")
        .to_string_lossy();
    let extension = "png";

    let filename = format!(
        "{stem}-palette-{cs}-k{k}.{extension}",
        cs = color_space.name()
    );
    let output_path = if let Some(parent) = parent {
        parent.join(filename)
    } else {
        Path::new(&filename).to_path_buf()
    };

    Ok(output_path)
}

fn find_file(
    output: &Option<PathBuf>,
    input: &Path,
    extension: &Option<Extension>,
    color_space: &ColorSpace,
) -> Result<PathBuf> {
    if let Some(output) = output {
        Ok(output.clone())
    } else {
        let parent = input.parent();
        let stem = input
            .file_stem()
            .expect("Expecting .jpg or .png files")
            .to_string_lossy();
        let extension = if let Some(extension) = extension {
            Cow::Borrowed(extension.value())
        } else {
            input
                .extension()
                .expect("Expecting .jpg or .png files")
                .to_string_lossy()
        };

        let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
        let millis = format!("{}{:03}", now.as_secs(), now.subsec_millis());

        let filename = format!(
            "{stem}-find-{cs}-{millis}.{extension}",
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

fn save_palette<P>(path: P, palette: &[[u8; 4]]) -> Result<()>
where
    P: AsRef<Path>,
{
    let height = 40;
    let width = palette.len() as u32 * height;

    let mut image_buffer: image::RgbaImage = image::ImageBuffer::new(width, height);

    for (x, _, pixel) in image_buffer.enumerate_pixels_mut() {
        let index = (x / height) as usize;
        let color = palette[index];

        *pixel = image::Rgba(color);
    }

    image_buffer.save(path)?;

    Ok(())
}

fn parse_colors(colors: &str) -> Result<Vec<[u8; 4]>> {
    colors
        .split(',')
        .map(|color_string| {
            let r = u8::from_str_radix(&color_string[1..3], 16)?;
            let g = u8::from_str_radix(&color_string[3..5], 16)?;
            let b = u8::from_str_radix(&color_string[5..7], 16)?;

            Ok([r, g, b, 255])
        })
        .collect::<Result<Vec<_>>>()
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_parse_colors() {
        let colors = String::from("#ffffff,#000000");

        let parsed = parse_colors(&colors);

        assert!(parsed.is_ok());

        let parsed = parsed.unwrap();
        assert_eq!(2, parsed.len());

        let expected: Vec<[u8; 4]> = vec![[255, 255, 255, 255], [0, 0, 0, 255]];
        assert_eq!(parsed, expected);
    }
}
