use std::{fs, path::Path};

#[cfg(feature = "validate")]
pub mod validate;

fn main() {
    println!("Hello, world!");

    let shaders = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("core/shaders");

    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("output");
    fs::create_dir_all(&out_dir).unwrap();

    #[cfg(not(feature = "validate"))]
    preprocessor::preprocess_shaders(&shaders, &out_dir).unwrap();

    #[cfg(feature = "validate")]
    generate_and_validate(&shaders, &out_dir)
}

#[cfg(feature = "validate")]
fn generate_and_validate(shaders: &Path, out_dir: &Path) {
    use kmeans_color_gpu_preprocessor::preprocess_shaders;

    let generated_shaders = preprocess_shaders(&shaders, &out_dir).unwrap();

    validate::validate_shaders(&generated_shaders).unwrap();
}
