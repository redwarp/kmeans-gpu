use std::{fs, path::PathBuf};

use naga::valid::{Capabilities, ValidationFlags, Validator};

pub fn validate_shaders(shaders: &[PathBuf]) -> anyhow::Result<()> {
    for shader_path in shaders {
        let shader = String::from_utf8(fs::read(shader_path)?)?;
        let path = shader_path.to_string_lossy();

        let module = match naga::front::wgsl::parse_str(&shader) {
            Ok(module) => module,
            Err(e) => {
                e.emit_to_stderr_with_path(&shader, &*path);
                continue;
            }
        };

        let validation_caps =
            Capabilities::all() & !(Capabilities::CLIP_DISTANCES | Capabilities::CULL_DISTANCE);

        let filename = shader_path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or("input");

        match Validator::new(ValidationFlags::all(), validation_caps).validate(&module) {
            Ok(_) => println!("Shader {filename} is valid"),
            Err(e) => {
                eprintln!("Shader {filename} invalid");
                e.emit_to_stderr_with_path(&shader, &path);
            }
        }
    }

    Ok(())
}
