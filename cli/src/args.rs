use std::fmt::Display;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::anyhow;
use anyhow::Result;
use clap::ValueEnum;
use clap::{Parser, Subcommand};
use regex::Regex;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
#[clap(propagate_version = true)]
pub struct Cli {
    #[clap(subcommand)]
    pub commands: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Quantized the image then output the reduced palette.
    Palette {
        /// Color count of the generated palette
        #[clap(short, long="colorcount", value_parser = validate_k)]
        color_count: u32,
        /// Input image file
        #[clap(short, long, value_parser = validate_filenames)]
        input: PathBuf,
        /// Optional output image file
        #[clap(short, long, value_parser)]
        output: Option<PathBuf>,
        /// Algorithm to use for palette reduction
        #[clap(value_enum, short, long, default_value_t=Algorithm::Kmeans)]
        algo: Algorithm,
        /// Each color will be represented by a square of <SIZE x SIZE>. Between 1 and 60
        #[clap(short, long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=60))]
        size: u32,
    },
    /// Find colors in image that are closest to the replacements, and swap them.
    Find {
        /// Input image file
        #[clap(short, long, value_parser = validate_filenames)]
        input: PathBuf,
        /// Optional output image file
        #[clap(short, long, value_parser)]
        output: Option<PathBuf>,
        /// List of RGB replacement colors formatted as "#RRGGBB,#RRGGBB" or path to a palette image
        #[clap(short, long, value_parser = validate_palette)]
        palette: Palette,
        /// Mix function to apply on the result
        #[clap(value_enum, short, long, default_value_t=ReduceMode::Replace)]
        mode: ReduceMode,
    },
    /// Quantized the image then replaces it's resulting color.
    Reduce {
        /// Color count of the generated palette
        #[clap(short, long="colorcount", value_parser = validate_k)]
        color_count: u32,
        /// Input image file
        #[clap(short, long, value_parser = validate_filenames)]
        input: PathBuf,
        /// Optional output image file
        #[clap(short, long, value_parser)]
        output: Option<PathBuf>,
        /// Algorithm to use for palette reduction
        #[clap(value_enum, short, long, default_value_t=Algorithm::Kmeans)]
        algo: Algorithm,
        /// Mix function to apply on the result
        #[clap(value_enum, short, long, default_value_t=ReduceMode::Replace)]
        mode: ReduceMode,
    },
}

#[derive(Debug, Clone)]
pub enum Extension {
    Png,
    Jpg,
}

impl Extension {
    pub fn name(&self) -> &'static str {
        match self {
            Extension::Png => "png",
            Extension::Jpg => "jpg",
        }
    }
}

impl FromStr for Extension {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "png" => Ok(Extension::Png),
            "jpg" => Ok(Extension::Jpg),
            _ => Err(anyhow!("Unsupported extension {s}")),
        }
    }
}

impl Display for Extension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Clone, Copy, ValueEnum)]
pub enum ReduceMode {
    Replace,
    Dither,
    Meld,
}

impl From<ReduceMode> for color_quantization_gpu::ReduceMode {
    fn from(reduce_mode: ReduceMode) -> Self {
        match reduce_mode {
            ReduceMode::Replace => color_quantization_gpu::ReduceMode::Replace,
            ReduceMode::Dither => color_quantization_gpu::ReduceMode::Dither,
            ReduceMode::Meld => color_quantization_gpu::ReduceMode::Meld,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
pub enum ColorSpace {
    Lab,
    Rgb,
}

impl From<ColorSpace> for color_quantization_gpu::ColorSpace {
    fn from(color_space: ColorSpace) -> Self {
        match color_space {
            ColorSpace::Lab => color_quantization_gpu::ColorSpace::Lab,
            ColorSpace::Rgb => color_quantization_gpu::ColorSpace::Rgb,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
pub enum Algorithm {
    Kmeans,
    Octree,
}

impl From<Algorithm> for color_quantization_gpu::Algorithm {
    fn from(algo: Algorithm) -> Self {
        match algo {
            Algorithm::Kmeans => color_quantization_gpu::Algorithm::Kmeans,
            Algorithm::Octree => color_quantization_gpu::Algorithm::Octree,
        }
    }
}

#[derive(Clone)]
pub struct Palette {
    pub colors: Vec<[u8; 4]>,
}

fn validate_k(s: &str) -> Result<u32> {
    match s.parse::<u32>() {
        Ok(k) => {
            if k >= 1 {
                Ok(k)
            } else {
                Err(anyhow!("k must be an integer higher than 0."))
            }
        }
        Err(_) => Err(anyhow!("k must be an integer higher than 0.")),
    }
}

fn validate_filenames(s: &str) -> Result<PathBuf> {
    if s.len() > 4 && (s.ends_with(".png") || s.ends_with(".jpg")) {
        Ok(PathBuf::from(s))
    } else {
        Err(anyhow!("Only support png or jpg files."))
    }
}

fn validate_palette(s: &str) -> Result<Palette> {
    let re = Regex::new(r"^#[0-9a-fA-F]{6}(?:,#[0-9a-fA-F]{6})*$").unwrap();
    if re.is_match(s) {
        parse_colors(s)
    } else {
        let path = PathBuf::from(s);
        if s.len() > 4 && (s.ends_with(".png") || s.ends_with(".jpg")) && path.exists() {
            parse_palette(&path)
        } else {
            Err(anyhow!(
                "The palette should be a path to an image file, or defined as \"#RRGGBB,#RRGGBB,#RRGGBB\""
            ))
        }
    }
}

fn parse_palette(path: &PathBuf) -> Result<Palette> {
    let image = image::open(&path)?.to_rgba8();
    let (width, height) = image.dimensions();
    let pixel_count = width as usize * height as usize;

    if pixel_count > 512 {
        return Err(anyhow!(
            "Trying to load a palette with more than 512 colors"
        ));
    }

    let mut colors: Vec<[u8; 4]> = bytemuck::cast_slice(image.as_raw()).to_vec();
    colors.sort();
    colors.dedup();

    if colors.len() < pixel_count {
        return Err(anyhow!("Trying to load a palette with recuring colors"));
    }
    Ok(Palette { colors })
}

fn parse_colors(colors: &str) -> Result<Palette> {
    let colors = colors
        .split(',')
        .map(|color_string| {
            let r = u8::from_str_radix(&color_string[1..3], 16)?;
            let g = u8::from_str_radix(&color_string[3..5], 16)?;
            let b = u8::from_str_radix(&color_string[5..7], 16)?;

            Ok([r, g, b, 255])
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Palette { colors })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_palette() {
        assert!(validate_palette("#ffffff,#000000").is_ok());
        assert!(validate_palette("#ffffff#000000").is_err());
        assert!(validate_palette("").is_err());
    }

    #[test]
    fn test_validate_k() {
        assert!(validate_k("1").is_ok());
        assert!(validate_k("150").is_ok());
        assert!(validate_k("abs").is_err());
        assert!(validate_k("0").is_err());
    }

    #[test]
    fn test_validate_filename() {
        assert!(validate_filenames("jog.png").is_ok());
        assert!(validate_filenames("jog.jpg").is_ok());
        assert!(validate_filenames("jog.pom").is_err());
        assert!(validate_filenames(".png").is_err());
    }

    #[test]
    fn test_parse_colors() {
        let colors = String::from("#ffffff,#000000");

        let parsed = parse_colors(&colors);

        assert!(parsed.is_ok());

        let parsed = parsed.unwrap();

        let Palette { colors } = parsed;

        assert_eq!(2, colors.len());

        let expected: Vec<[u8; 4]> = vec![[255, 255, 255, 255], [0, 0, 0, 255]];
        assert_eq!(colors, expected);
    }

    #[test]
    fn test_parse_palette() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let mut palette_path = PathBuf::from(manifest_dir);
        palette_path.push("../gfx/resurrect_64.png");

        let parsed = parse_palette(&palette_path);

        assert!(parsed.is_ok());

        let parsed = parsed.unwrap();

        assert_eq!(64, parsed.colors.len());
    }
}
