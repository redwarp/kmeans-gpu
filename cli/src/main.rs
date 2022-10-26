use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Ok, Result};
use args::{Cli, Commands, Extension};
use clap::Parser;
use image::{ImageBuffer, Rgba, RgbaImage};
use k_means_gpu::{find, image::Image, palette, reduce, ImageProcessor, ReduceMode};
use pollster::FutureExt;

mod args;

fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match cli.commands {
        Commands::Palette {
            color_count,
            input,
            output,
        } => palette_subcommand(color_count, input, output).block_on(),
        Commands::Find {
            input,
            output,
            replacement,
            mode,
        } => find_subcommand(input, output, replacement, mode.into()).block_on(),
        Commands::Reduce {
            color_count,
            input,
            output,
            mode,
        } => reduce_subcommand(color_count, input, output, mode.into()).block_on(),
    }?;

    Ok(())
}

async fn palette_subcommand(
    color_count: u32,
    input: PathBuf,
    output: Option<PathBuf>,
) -> Result<()> {
    let image = image::open(&input)?.to_rgba8();
    let image = to_lib_image(&image);

    let image_processor = ImageProcessor::new().await?;
    let result = palette(&image_processor, color_count, &image)?;

    let path = palette_file(color_count, &input, &output)?;
    save_palette(path, &result)?;

    let colors = result
        .into_iter()
        .map(|color| format!("#{:02X}{:02X}{:02X}", color[0], color[1], color[2]))
        .collect::<Vec<_>>()
        .join(",");

    println!("Palette: {colors}");

    Ok(())
}

async fn find_subcommand(
    input: PathBuf,
    output: Option<PathBuf>,
    replacement: String,
    reduce_mode: ReduceMode,
) -> Result<()> {
    let colors = parse_colors(&replacement)?;

    let image = image::open(&input)?.to_rgba8();
    let image = to_lib_image(&image);

    let image_processor = ImageProcessor::new().await?;
    let result = find(&image_processor, &image, &colors, &reduce_mode)?;

    let (width, height) = result.dimensions();

    if let Some(output_image) =
        ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, result.into_raw_pixels())
    {
        let output_file = find_file(&output, &input, &None)?;
        output_image.save(output_file)?;
    }

    Ok(())
}

async fn reduce_subcommand(
    color_count: u32,
    input: PathBuf,
    output: Option<PathBuf>,
    reduce_mode: ReduceMode,
) -> Result<()> {
    let image = image::open(&input)?.to_rgba8();
    let image = to_lib_image(&image);

    let image_processor = ImageProcessor::new().await?;
    let result = reduce(&image_processor, color_count, &image, &reduce_mode)?;

    let (width, height) = result.dimensions();

    if let Some(output_image) =
        ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, result.into_raw_pixels())
    {
        let output_file = output_file(
            color_count,
            &format!("reduce-{reduce_mode}"),
            &output,
            &input,
            &None,
        )?;
        output_image.save(output_file)?;
    }

    Ok(())
}

fn output_file(
    color_count: u32,
    function: &str,
    output: &Option<PathBuf>,
    input: &Path,
    extension: &Option<Extension>,
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
            Cow::Borrowed(extension.name())
        } else {
            input
                .extension()
                .expect("Expecting .jpg or .png files")
                .to_string_lossy()
        };

        let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
        let millis = format!("{}{:03}", now.as_secs(), now.subsec_millis());

        let filename = format!("{stem}-{function}-c{color_count}-{millis}.{extension}");
        let output_path = if let Some(parent) = parent {
            parent.join(filename)
        } else {
            Path::new(&filename).to_path_buf()
        };

        Ok(output_path)
    }
}

fn palette_file(k: u32, input: &Path, output: &Option<PathBuf>) -> Result<PathBuf> {
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

    let filename = format!("{stem}-palette-c{k}.{extension}",);
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
            Cow::Borrowed(extension.name())
        } else {
            input
                .extension()
                .expect("Expecting .jpg or .png files")
                .to_string_lossy()
        };

        let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
        let millis = format!("{}{:03}", now.as_secs(), now.subsec_millis());

        let filename = format!("{stem}-find-{millis}.{extension}",);
        let output_path = if let Some(parent) = parent {
            parent.join(filename)
        } else {
            Path::new(&filename).to_path_buf()
        };

        Ok(output_path)
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

fn to_lib_image(image: &RgbaImage) -> Image<&[[u8; 4]]> {
    Image::new(image.dimensions(), bytemuck::cast_slice(image.as_raw()))
}
