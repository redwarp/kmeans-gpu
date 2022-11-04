@group(0) @binding(0) var<uniform> input: array<vec4<f32>,2>;
@group(0) @binding(1) var<storage, read_write> output: f32;

// #include ../functions/delta_e.wgsl

@compute
@workgroup_size(1)
fn run_distance_cie94(){
    output = distance_cie94(input[0].rgb, input[1].rgb);
}

@compute
@workgroup_size(1)
fn run_distance_cie2000(){
    output = distance_cie2000(input[0].rgb, input[1].rgb);
}


@compute
@workgroup_size(1)
fn run_pow(){
    output = pow(input[0].x, input[0].y);
}