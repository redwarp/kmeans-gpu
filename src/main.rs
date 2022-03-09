use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use clap::{command, Arg};
use image::{ImageBuffer, Rgba};
use k_means_gpu::{kmeans, Image};

fn main() -> Result<()> {
    let matches = command!()
        .arg(
            Arg::new("k")
                .short('k')
                .takes_value(true)
                .default_value("8")
                .validator(|input| input.parse::<u32>()),
        )
        .arg(
            Arg::new("input")
                .short('i')
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::new("extension")
                .long("ext")
                .takes_value(true)
                .default_value("png"),
        )
        .get_matches();

    let k = matches.value_of("k").expect("Has default value").parse()?;
    let input = matches.value_of("input").expect("Required argument");
    let extension = matches.value_of("extension");
    let image = image::open(input)?.to_rgba8();
    let image = Image::new(image.dimensions(), image.into_raw().to_vec());

    let result = kmeans(k, &image)?;
    let (width, height) = result.dimensions();
    if let Some(output_image) =
        ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, result.raw_pixels())
    {
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
