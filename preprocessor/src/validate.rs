use std::{error::Error, fs, path::PathBuf};

use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    files::SimpleFile,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use naga::{valid::ValidationFlags, WithSpan};

pub fn validate_shaders(shaders: &[PathBuf]) -> anyhow::Result<()> {
    for shader_path in shaders {
        let shader = String::from_utf8(fs::read(shader_path)?)?;

        let result = naga::front::wgsl::parse_str(&shader);

        let module = match result {
            Ok(v) => Some(v),
            Err(e) => {
                let path = shader_path.to_string_lossy();
                e.emit_to_stderr_with_path(&shader, &path);
                None
            }
        };

        if let Some(module) = module {
            let validation_caps = naga::valid::Capabilities::all()
                & !(naga::valid::Capabilities::CLIP_DISTANCE
                    | naga::valid::Capabilities::CULL_DISTANCE);

            let validation_flags = ValidationFlags::all();

            let filename = shader_path
                .file_name()
                .and_then(std::ffi::OsStr::to_str)
                .unwrap_or("input");

            match naga::valid::Validator::new(validation_flags, validation_caps).validate(&module) {
                Ok(_) => println!("Shader {filename} is valid"),
                Err(e) => {
                    eprintln!("Shader {filename} invalid");
                    emit_annotated_error(&e, filename, &shader);
                }
            }
        }
    }

    Ok(())
}

pub fn emit_annotated_error<E: Error>(ann_err: &WithSpan<E>, filename: &str, source: &str) {
    let files = SimpleFile::new(filename, source);
    let config = codespan_reporting::term::Config::default();
    let writer = StandardStream::stderr(ColorChoice::Auto);

    let diagnostic = Diagnostic::error().with_labels(
        ann_err
            .spans()
            .map(|(span, desc)| {
                Label::primary((), span.to_range().unwrap()).with_message(desc.to_owned())
            })
            .collect(),
    );

    term::emit(&mut writer.lock(), &config, &files, &diagnostic).expect("cannot write error");
}
