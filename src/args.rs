use std::fmt::Display;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::anyhow;
use anyhow::Result;
use clap::{Parser, Subcommand};
use k_means_gpu::ColorSpace;
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
        #[clap(short, validator = validate_k)]
        k: u32,
        /// Input file
        #[clap(short, long, validator = validate_filenames, parse(from_os_str))]
        input: PathBuf,
        /// Optional output file
        #[clap(short, long, parse(from_os_str))]
        output: Option<PathBuf>,
        /// Optional extension to use if an output file is not specified
        #[clap(short, long)]
        extension: Option<Extension>,
        /// The colorspace to use when calculating colors. Lab gives more natural colors
        #[clap(short, long="colorspace", default_value_t=ColorSpace::Lab)]
        color_space: ColorSpace,
    },
    /// Output the palette calculated with k-means
    Palette {
        /// K value, aka the number of colors we want to extract
        #[clap(short, validator = validate_k)]
        k: u32,
        /// Input file
        #[clap(short, long, validator = validate_filenames, parse(from_os_str))]
        input: PathBuf,
        /// Optional output file
        #[clap(short, long, parse(from_os_str))]
        output: Option<PathBuf>,
        /// The colorspace to use when calculating colors. Lab gives more natural colors
        #[clap(short, long="colorspace", default_value_t=ColorSpace::Lab)]
        color_space: ColorSpace,
    },
    /// Find colors in image that are closest to the replacements, and swap them
    Find {
        /// Input file
        #[clap(short, long, validator = validate_filenames, parse(from_os_str))]
        input: PathBuf,
        /// Optional output file
        #[clap(short, long, parse(from_os_str))]
        output: Option<PathBuf>,
        /// List of RGB replacement colors, as #RRGGBB,#RRGGBB
        #[clap(short, long, validator = validate_replacement)]
        replacement: String,
        /// The colorspace to use when calculating colors. Lab gives more natural colors
        #[clap(short, long="colorspace", default_value_t=ColorSpace::Lab)]
        color_space: ColorSpace,
    },
    /// Quantized the image with kmeans, then dithers it with the reduced colors.
    Dither {
        /// K value, aka the number of colors we want to extract
        #[clap(short, validator = validate_k_for_dithering)]
        k: u32,
        /// Input file
        #[clap(short, long, validator = validate_filenames, parse(from_os_str))]
        input: PathBuf,
        /// Optional output file
        #[clap(short, long, parse(from_os_str))]
        output: Option<PathBuf>,
        /// Optional extension to use if an output file is not specified
        #[clap(short, long)]
        extension: Option<Extension>,
        /// The colorspace to use when calculating colors. Lab gives more natural colors
        #[clap(short, long="colorspace", default_value_t=ColorSpace::Lab)]
        color_space: ColorSpace,
    },
}

#[derive(Debug)]
pub enum Extension {
    Png,
    Jpg,
}

impl Extension {
    pub fn value(&self) -> &'static str {
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
        write!(f, "{}", self.value())
    }
}

fn validate_k(s: &str) -> Result<()> {
    match s.parse::<u32>() {
        Ok(k) => {
            if k >= 1 {
                Ok(())
            } else {
                Err(anyhow!("k must be an integer higher than 0."))
            }
        }
        Err(_) => Err(anyhow!("k must be an integer higher than 0.")),
    }
}

fn validate_k_for_dithering(s: &str) -> Result<()> {
    match s.parse::<u32>() {
        Ok(k) => {
            if k >= 2 {
                Ok(())
            } else {
                Err(anyhow!("k must be an integer, minimum 2."))
            }
        }
        Err(_) => Err(anyhow!("k must be an integer, minimum 2.")),
    }
}

fn validate_filenames(s: &str) -> Result<()> {
    if s.len() > 4 && s.ends_with(".png") || s.ends_with(".jpg") {
        Ok(())
    } else {
        Err(anyhow!("Only support png or jpg files."))
    }
}

fn validate_replacement(s: &str) -> Result<()> {
    let re = Regex::new(r"^#[0-9a-fA-F]{6}(?:,#[0-9a-fA-F]{6})*$").unwrap();
    if re.is_match(s) {
        Ok(())
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
