use std::{fs, path::Path};

fn main() {
    println!("Hello, world!");

    let shaders = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("core/shaders");

    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("processed_shaders");
    fs::create_dir_all(&out_dir).unwrap();

    preprocessor::preprocess_shaders(&shaders, &out_dir).unwrap();

    // https://stackoverflow.com/questions/35711044/how-can-i-specify-binary-only-dependencies
    // Check this, could use a naga feature to also validate the generated shaders
}
