@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var output_texture : texture_storage_2d<rgba32float, write>;

@compute
@workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>,
) {
    let dimensions = textureDimensions(output_texture);
    let coords = vec2<i32>(global_id.xy);

    if(coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let texel = textureLoad(input_texture, coords, 0);
    textureStore(output_texture, coords, texel);
}