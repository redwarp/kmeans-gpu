struct Centroids {
    count: u32;
    data: array<vec4<f32>>;
};

[[group(0), binding(0)]] var<storage, read_write> centroids: Centroids;

fn xyz_to_rgb(xyz: vec4<f32>) -> vec4<f32> {
    var x = xyz.x / 100.0;
    var y = xyz.y / 100.0;
    var z = xyz.z / 100.0;

    var r = x *  3.2406 + y * -1.5372 + z * -0.4986;
    var g = x * -0.9689 + y *  1.8758 + z *  0.0415;
    var b = x *  0.0557 + y * -0.2040 + z *  1.0570;

    if (r > 0.0031308) {
        r = 1.055 * pow(r,(1.0 / 2.4)) - 0.055;
    } else {
        r = 12.92 * r;
    }
    if (g > 0.0031308) {
        g = 1.055 * pow(g,(1.0 / 2.4)) - 0.055;
    } else {
        g = 12.92 * g;
    }
    if (b > 0.0031308) {
        b = 1.055 * pow(b,(1.0 / 2.4)) - 0.055;
    } else {
        b = 12.92 * b;
    }

    return vec4<f32>(r, g, b, 1.0);
}

fn lab_to_xyz(lab: vec4<f32>) -> vec4<f32> {
    var y = (lab.r + 16.0) / 116.0;
    var x = lab.g / 500.0 + y;
    var z = y - lab.b / 200.0;

    if (pow(x,3.0) > 0.008856) {
        x = pow(x, 3.0);
    } else {
        x = (x - 16.0 / 116.0) / 7.787;
    }
    if (pow(y,3.0) > 0.008856) {
        y = pow(y, 3.0);
    } else {
        y = (y - 16.0 / 116.0) / 7.787;
    }
    if (pow(z,3.0) > 0.008856) {
        z = pow(z, 3.0);
    } else {
        z = (z - 16.0 / 116.0) / 7.787;
    }

    x = x * 95.047;
    y = y * 100.000;
    z = z * 108.883;

    return vec4<f32>(x, y, z, 1.0);
}

[[stage(compute), workgroup_size(1)]]
fn main() {
    for(var k: u32 = 0u; k < centroids.count; k = k +1u) {
        let color = xyz_to_rgb(lab_to_xyz(centroids.data[k]));
        centroids.data[k] = color;
    }
}