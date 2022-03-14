[[group(0), binding(0)]] var input_texture : texture_2d<f32>;
[[group(0), binding(1)]] var output_texture : texture_storage_2d<rgba16float, write>;

fn rgb_to_xyz(rgb: vec4<f32>) -> vec4<f32> {
    var r = rgb.r;
    var g = rgb.g;
    var b = rgb.b;

    if (r > 0.04045) {
        r = pow(((r + 0.055) / 1.055), 2.4);
    } else {
        r =  r / 12.92;
    }
    if (g > 0.04045) {
        g = pow(((g + 0.055) / 1.055), 2.4);
    } else {
        g =  g / 12.92;
    }
    if (b > 0.04045) {
        b = pow(((b + 0.055) / 1.055), 2.4);
    } else {
        b =  b / 12.92;
    }
    r = r * 100.0;
    g = g * 100.0;
    b = b * 100.0;

    var x = r * 0.4124 + g * 0.3576 + b * 0.1805;
    var y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    var z = r * 0.0193 + g * 0.1192 + b * 0.9505;
    return vec4<f32>(x, y, z, 1.0);
}

fn xyz_to_lab(xyz: vec4<f32>) -> vec4<f32> {
    var x = xyz.x / 95.047;
    var y = xyz.y / 100.000;
    var z = xyz.z / 108.883;

    if (x > 0.008856) {
        x = pow(x, 1.0/3.0);
    } else {
        x = (7.787 * x) + (16.0 / 116.0);
    }
    if (y > 0.008856) {
        y = pow(y, 1.0/3.0);
    } else {
        y = (7.787 * y) + (16.0 / 116.0);
    }
    if (z > 0.008856) {
        z = pow(z, 1.0/3.0);
    } else {
        z = (7.787 * z) + (16.0 / 116.0);
    }
    let l = (116.0 * y) - 16.0;
    let a = 500.0 * (x - y);
    let b = 200.0 * (y - z);

    return vec4<f32>(l, a, b, 1.0);
}

[[stage(compute), workgroup_size(16, 16)]]
fn main(
    [[builtin(global_invocation_id)]] global_id : vec3<u32>,
) {
    let dimensions = textureDimensions(output_texture);
    let coords = vec2<i32>(global_id.xy);

    if(coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let lab = xyz_to_lab(rgb_to_xyz(textureLoad(input_texture, coords, 0)));
    textureStore(output_texture, coords, lab);
}