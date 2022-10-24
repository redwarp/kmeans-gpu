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
    /// Create an image quantized with kmeans
    Kmeans {
        /// K value, aka the number of colors we want to extract
        #[clap(short, value_parser = validate_k)]
        k: u32,
        /// Input file
        #[clap(short, long, value_parser = validate_filenames)]
        input: PathBuf,
        /// Optional output file
        #[clap(short, long, value_parser)]
        output: Option<PathBuf>,
        /// Optional extension to use if an output file is not specified
        #[clap(short, long)]
        extension: Option<Extension>,
        /// The colorspace to use when calculating colors. Lab gives more natural colors
        #[clap(value_enum, short, long="colorspace", default_value_t=ColorSpace::Lab)]
        color_space: ColorSpace,
    },
    /// Output the palette calculated with k-means
    Palette {
        /// K value, aka the number of colors we want to extract
        #[clap(short, value_parser = validate_k)]
        k: u32,
        /// Input file
        #[clap(short, long, value_parser = validate_filenames)]
        input: PathBuf,
        /// Optional output file
        #[clap(short, long, value_parser)]
        output: Option<PathBuf>,
        /// The colorspace to use when calculating colors. Lab gives more natural colors
        #[clap(value_enum, short, long="colorspace", default_value_t=ColorSpace::Lab)]
        color_space: ColorSpace,
    },
    /// Find colors in image that are closest to the replacements, and swap them
    Find {
        /// Input file
        #[clap(short, long, value_parser = validate_filenames)]
        input: PathBuf,
        /// Optional output file
        #[clap(short, long, value_parser)]
        output: Option<PathBuf>,
        /// List of RGB replacement colors formatted as "#RRGGBB,#RRGGBB"
        #[clap(short, long, value_parser = validate_replacement)]
        replacement: String,
        /// The colorspace to use when calculating colors. Lab gives more natural colors
        #[clap(value_enum, short, long="colorspace", default_value_t=ColorSpace::Lab)]
        color_space: ColorSpace,
    },
    /// Quantized the image with kmeans, then mix it's resulting color.
    Mix {
        /// K value, aka the number of colors we want to extract
        #[clap(short, value_parser = validate_k_for_dithering)]
        k: u32,
        /// Input file
        #[clap(short, long, value_parser = validate_filenames)]
        input: PathBuf,
        /// Optional output file
        #[clap(short, long, value_parser)]
        output: Option<PathBuf>,
        /// Optional extension to use if an output file is not specified
        #[clap(short, long)]
        extension: Option<Extension>,
        /// The colorspace to use when calculating colors. Lab gives more natural colors
        #[clap(value_enum, short, long="colorspace", default_value_t=ColorSpace::Lab)]
        color_space: ColorSpace,
        /// Mix function to apply on the result
        #[clap(value_enum, short, long="mixmode", default_value_t=MixMode::Dither)]
        mix_mode: MixMode,
    },
}

#[derive(Debug, Clone)]
pub enum Extension {
    Png,
    Jpg,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum MixMode {
    Dither,
    Meld,
}

impl From<MixMode> for k_means_gpu::MixMode {
    fn from(mix_mode: MixMode) -> Self {
        match mix_mode {
            MixMode::Dither => k_means_gpu::MixMode::Dither,
            MixMode::Meld => k_means_gpu::MixMode::Meld,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
pub enum ColorSpace {
    Lab,
    Rgb,
}

impl From<ColorSpace> for k_means_gpu::ColorSpace {
    fn from(color_space: ColorSpace) -> Self {
        match color_space {
            ColorSpace::Lab => k_means_gpu::ColorSpace::Lab,
            ColorSpace::Rgb => k_means_gpu::ColorSpace::Rgb,
        }
    }
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

fn validate_k_for_dithering(s: &str) -> Result<u32> {
    match s.parse::<u32>() {
        Ok(k) => {
            if k >= 2 {
                Ok(k)
            } else {
                Err(anyhow!("k must be an integer, minimum 2."))
            }
        }
        Err(_) => Err(anyhow!("k must be an integer, minimum 2.")),
    }
}

fn validate_filenames(s: &str) -> Result<PathBuf> {
    if s.len() > 4 && s.ends_with(".png") || s.ends_with(".jpg") {
        Ok(PathBuf::from(s))
    } else {
        Err(anyhow!("Only support png or jpg files."))
    }
}

fn validate_replacement(s: &str) -> Result<String> {
    let re = Regex::new(r"^#[0-9a-fA-F]{6}(?:,#[0-9a-fA-F]{6})*$").unwrap();
    if re.is_match(s) {
        Ok(s.to_owned())
    } else {
        Err(anyhow!(
            "Replacement colors should be chained like #ffaa12,#fe7845,#aabbff"
        ))
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_validate_replacement() {
        assert!(validate_replacement("#ffffff,#000000").is_ok());
        assert!(validate_replacement("#ffffff#000000").is_err());
        assert!(validate_replacement("").is_err());
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
}
