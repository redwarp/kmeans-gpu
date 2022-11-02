use std::{fs, path::PathBuf};

pub fn validate_shaders(shaders: &[PathBuf]) -> anyhow::Result<()> {
    for shader_path in shaders {
        let shader = String::from_utf8(fs::read(shader_path)?)?;

        let result = naga::front::wgsl::parse_str(&shader);

        match result {
            Ok(_) => println!(
                "Shader {path} is valid",
                path = shader_path.to_string_lossy()
            ),
            Err(e) => {
                let path = shader_path.to_string_lossy();
                e.emit_to_stderr_with_path(&shader, &path);
            }
        }
    }

    Ok(())
}
