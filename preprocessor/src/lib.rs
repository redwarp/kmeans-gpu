use regex::Regex;
use std::{
    borrow::Cow,
    ffi::OsStr,
    fs, io,
    path::{Path, PathBuf},
};

struct PreProcessor {
    shaders: PathBuf,
    re: Regex,
}

impl PreProcessor {
    fn new(shaders: impl Into<PathBuf>) -> Self {
        let re: Regex = Regex::new(r"^(// \#include ([a-z0-9\-_./]+))$").expect("Verified regex");

        Self {
            shaders: shaders.into(),
            re,
        }
    }

    pub fn preprocess_shaders(&self, out_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
        let wgsl_extension = OsStr::new("wgsl").to_owned();

        let mut output_files = vec![];
        let mut processed = vec![];
        for wgsl_file in list_files(&self.shaders)?.into_iter().filter(|path| {
            if let Some(extension) = path.extension() {
                if extension == wgsl_extension {
                    return true;
                }
            }
            false
        }) {
            output_files.push(self.process_file(&wgsl_file, out_dir)?);
            processed.push(wgsl_file);
        }

        Ok(output_files)
    }

    fn process_file(&self, wgsl_file: &Path, into: &Path) -> anyhow::Result<PathBuf> {
        let path_diff =
            pathdiff::diff_paths(wgsl_file, &self.shaders).expect("Shader contained folder");

        let into = into.join(&path_diff);
        if let Some(parent) = into.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut inflated = vec![];

        let content = self.expanded_content(wgsl_file, &mut inflated)?;

        fs::write(&into, &content)?;

        Ok(into)
    }

    fn expanded_content(
        &self,
        wgsl_file: &Path,
        inflated: &mut Vec<String>,
    ) -> anyhow::Result<String> {
        let content = fs::read_to_string(wgsl_file)?.trim().to_owned();

        let lines: anyhow::Result<Vec<_>> = content
            .lines()
            .map(|line| {
                if let Some(captures) = self.re.captures(line) {
                    let include_filename = &captures[2];
                    let to_include = wgsl_file
                        .parent()
                        .expect("Should have parents?")
                        .join(include_filename);

                    if !to_include.exists() {
                        return Err(anyhow::anyhow!("File {include_filename} doesn't exist."));
                    }

                    let to_include = to_include.canonicalize()?;

                    let path_as_string = to_include.to_string_lossy().into_owned();

                    if inflated.contains(&path_as_string) {
                        return Err(anyhow::anyhow!(
                            "Recursion: file {include_filename} already inflated."
                        ));
                    }

                    inflated.push(path_as_string);

                    Ok(Cow::Owned(self.expanded_content(&to_include, inflated)?))
                } else {
                    Ok(Cow::Borrowed(line))
                }
            })
            .collect();

        Ok(lines?.join("\n") + "\n")
    }
}

/// Preprocess all shaders from the shaders directory, and outputting the inflatted shaders
/// in the out directory.
/// Preprocessing shaders will inflate `// #include <file>.wgsl`, allowing reusing of functions.
pub fn preprocess_shaders(shaders: &Path, out_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let preprocessor = PreProcessor::new(shaders);
    preprocessor.preprocess_shaders(out_dir)
}

fn list_files(folder: &Path) -> io::Result<Vec<PathBuf>> {
    fn recursive_list_files(folder: &Path, into: &mut Vec<PathBuf>) -> io::Result<()> {
        if folder.is_dir() {
            let paths = fs::read_dir(folder)?;

            for read in paths {
                let path = read?.path();
                if path.is_dir() {
                    recursive_list_files(&path, into)?;
                } else {
                    into.push(path);
                }
            }
        }
        Ok(())
    }

    let mut files = vec![];
    recursive_list_files(folder, &mut files)?;

    Ok(files)
}
