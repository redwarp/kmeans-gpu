struct Tuple {
    a: vec3<f32>,
    b: vec3<f32>,
};

@group(0) @binding(0) var<uniform> input: Tuple;
@group(0) @binding(1) var<storage, read_write> output: f32;

// #include ../functions/delta_e.wgsl

@compute
@workgroup_size(1)
fn run_distance_cie94(){
    output = distance_cie94(input.a, input.b);
}

@compute
@workgroup_size(1)
fn run_distance_cie2000(){
    output = distance_cie2000(input.a, input.b);
}
