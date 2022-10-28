use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Ok, Result};
use args::{Cli, Commands, Extension, Palette};
use clap::Parser;
use image::{ImageBuffer, Rgba, RgbaImage};
use k_means_gpu::{find, image::Image, palette, reduce, Algorithm, ImageProcessor, ReduceMode};
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
            algo,
            size,
        } => palette_subcommand2(color_count, input, output, algo.into(), size).block_on(),
        Commands::Find {
            input,
            output,
            palette,
            mode,
        } => find_subcommand(input, output, palette, mode.into()).block_on(),
        Commands::Reduce {
            color_count,
            input,
            output,
            algo,
            mode,
        } => reduce_subcommand(color_count, input, output, algo.into(), mode.into()).block_on(),
    }?;

    Ok(())
}

async fn palette_subcommand2(
    color_count: u32,
    input: PathBuf,
    output: Option<PathBuf>,
    algo: Algorithm,
    size: u32,
) -> Result<()> {
    let image = image::open(&input)?.to_rgba8();
    let image = to_lib_image(&image);

    let image_processor = ImageProcessor::new().await?;

    let result = palette(&image_processor, color_count, &image, algo)?;

    let path = palette_file(color_count, &input, &output, &algo, size)?;
    save_palette(path, &result, size)?;

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
    palette: Palette,
    reduce_mode: ReduceMode,
) -> Result<()> {
    let image = image::open(&input)?.to_rgba8();
    let image = to_lib_image(&image);

    let image_processor = ImageProcessor::new().await?;
    let result = find(&image_processor, &image, &palette.colors, &reduce_mode)?;

    let (width, height) = result.dimensions();

    if let Some(output_image) =
        ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, result.into_raw_pixels())
    {
        let output_file = find_file(&reduce_mode, &output, &input, &None)?;
        output_image.save(output_file)?;
    }

    Ok(())
}

async fn reduce_subcommand(
    color_count: u32,
    input: PathBuf,
    output: Option<PathBuf>,
    algo: Algorithm,
    reduce_mode: ReduceMode,
) -> Result<()> {
    let image = image::open(&input)?.to_rgba8();
    let image = to_lib_image(&image);

    let image_processor = ImageProcessor::new().await?;
    let result = reduce(&image_processor, color_count, &image, &algo, &reduce_mode)?;

    let (width, height) = result.dimensions();

    if let Some(output_image) =
        ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, result.into_raw_pixels())
    {
        let output_file = reduce_file(color_count, &algo, &reduce_mode, &output, &input, &None)?;
        output_image.save(output_file)?;
    }

    Ok(())
}

fn reduce_file(
    color_count: u32,
    algo: &Algorithm,
    reduce_mode: &ReduceMode,
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

        let filename = format!("{stem}-reduce-c{color_count}-{algo}-{reduce_mode}.{extension}");
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
    algo: &Algorithm,
    size: u32,
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

    let filename = format!("{stem}-palette-c{k}-{algo}-s{size}.{extension}",);
    let output_path = if let Some(parent) = parent {
        parent.join(filename)
    } else {
        Path::new(&filename).to_path_buf()
    };

    Ok(output_path)
}

fn find_file(
    reduce_mode: &ReduceMode,
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

        let filename = format!("{stem}-find-{reduce_mode}-{millis}.{extension}",);
        let output_path = if let Some(parent) = parent {
            parent.join(filename)
        } else {
            Path::new(&filename).to_path_buf()
        };

        Ok(output_path)
    }
}

fn save_palette<P>(path: P, palette: &[[u8; 4]], size: u32) -> Result<()>
where
    P: AsRef<Path>,
{
    let width = palette.len() as u32 * size;

    let mut image_buffer: image::RgbaImage = image::ImageBuffer::new(width, size);

    for (x, _, pixel) in image_buffer.enumerate_pixels_mut() {
        let index = (x / size) as usize;
        let color = palette[index];

        *pixel = image::Rgba(color);
    }

    image_buffer.save(path)?;

    Ok(())
}

fn to_lib_image(image: &RgbaImage) -> Image<&[[u8; 4]]> {
    Image::new(image.dimensions(), bytemuck::cast_slice(image.as_raw()))
}
