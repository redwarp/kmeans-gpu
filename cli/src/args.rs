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
        /// Input file
        #[clap(short, long, value_parser = validate_filenames)]
        input: PathBuf,
        /// Optional output file
        #[clap(short, long, value_parser)]
        output: Option<PathBuf>,
    },
    /// Find colors in image that are closest to the replacements, and swap them.
    Find {
        /// Input file
        #[clap(short, long, value_parser = validate_filenames)]
        input: PathBuf,
        /// Optional output file
        #[clap(short, long, value_parser)]
        output: Option<PathBuf>,
        /// List of RGB replacement colors formatted as "#RRGGBB,#RRGGBB"
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
        /// Input file
        #[clap(short, long, value_parser = validate_filenames)]
        input: PathBuf,
        /// Optional output file
        #[clap(short, long, value_parser)]
        output: Option<PathBuf>,
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

#[derive(Clone, Copy, ValueEnum)]
pub enum ReduceMode {
    Replace,
    Dither,
    Meld,
}

impl From<ReduceMode> for k_means_gpu::ReduceMode {
    fn from(reduce_mode: ReduceMode) -> Self {
        match reduce_mode {
            ReduceMode::Replace => k_means_gpu::ReduceMode::Replace,
            ReduceMode::Dither => k_means_gpu::ReduceMode::Dither,
            ReduceMode::Meld => k_means_gpu::ReduceMode::Meld,
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
    if s.len() > 4 && s.ends_with(".png") || s.ends_with(".jpg") {
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
        Err(anyhow!(
            "The palette should be define as \"#ffaa12,#fe7845,#aabbff\""
        ))
    }
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
    // Note this useful idiom: importing names from outer (for mod tests) scope.
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
}
