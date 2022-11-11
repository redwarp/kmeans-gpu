use std::{env, path::Path};

use kmeans_color_gpu_preprocessor::preprocess_shaders;

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir).join("shaders");

    let shaders = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("core/shaders");

    preprocess_shaders(&shaders, &out_dir).unwrap();

    println!(
        "cargo:rerun-if-changed={path}",
        path = shaders.to_string_lossy()
    );
}
