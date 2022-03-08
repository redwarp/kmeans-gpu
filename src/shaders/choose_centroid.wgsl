struct Centroids {
    count: u32;
    data: array<u32>;
};

struct Indices {
    data: array<u32>;
};

struct ColorBuffer {
    data: array<atomic<u32>>;
};

struct StateBuffer {
    state: array<atomic<u32>>;
};

struct Settings {
    k: u32;
};

[[group(0), binding(0)]] var<storage, read_write> centroids: Centroids;
[[group(0), binding(1)]] var<storage, read> calculated: Indices;
[[group(0), binding(2)]] var pixels: texture_2d<u32>;
[[group(1), binding(0)]] var<storage, read_write> prefix_buffer: ColorBuffer;
[[group(1), binding(1)]] var<storage, read_write> flag_buffer: StateBuffer;
[[group(2), binding(0)]] var<uniform> settings: Settings;


let workgroup_size: u32 = 256u;
let workgroup_size_3: u32 = 768u;
let N_SEQ: u32 = 8u;

var<workgroup> scratch: array<vec4<u32>, workgroup_size_3>;
var<workgroup> shared_prefix: vec4<u32>;
var<workgroup> shared_flag: u32;

let FLAG_NOT_READY = 0u;
let FLAG_AGGREGATE_READY = 1u;
let FLAG_PREFIX_READY = 2u;

fn coords(global_x: u32, dimensions: vec2<i32>) -> vec2<i32> {
    return vec2<i32>(vec2<u32>(global_x % u32(dimensions.x), global_x / u32(dimensions.x)));
}

fn last_group_idx() -> u32 {
    return arrayLength(&flag_buffer.state) - 1u;
}

fn in_bounds(global_x: u32, dimensions: vec2<i32>) -> bool {
    let x = global_x % u32(dimensions.x);
    let y = global_x / u32(dimensions.x);
    return x < u32(dimensions.x) && y < u32(dimensions.y);
}

fn match_centroid(k: u32, global_x: u32) -> bool {
    return calculated.data[global_x] == k;
}

[[stage(compute), workgroup_size(256)]]
fn main(
    [[builtin(local_invocation_id)]] local_id : vec3<u32>,
    [[builtin(workgroup_id)]] workgroup_id : vec3<u32>,
    [[builtin(global_invocation_id)]] global_id : vec3<u32>,
) {
    let dimensions = textureDimensions(pixels);
    let xyz = calculated.data[0];
    let global_x = global_id.x;
   
    let k = settings.k;
    scratch[local_id.x] = vec4<u32>(0u);
    if (local_id.x == workgroup_size - 1u) {
        atomicStore(&flag_buffer.state[workgroup_id.x], FLAG_NOT_READY);
    }
    workgroupBarrier();
    storageBarrier();

    var local: vec4<u32> = vec4<u32>(0u);
    for (var i: u32 = 0u; i < N_SEQ; i = i + 1u) {
        if (match_centroid(k, global_x * N_SEQ + i) && in_bounds(global_x * N_SEQ + i, dimensions)) {
            local = local + vec4<u32>(textureLoad(pixels, coords(global_x * N_SEQ + i, dimensions), 0).rgb, 1u);
        }
        workgroupBarrier();
    }

    // var local: vec4<u32> = vec4<u32>(textureLoad(pixels, coords(global_x * N_SEQ, dimensions), 0).rgb, u32(in_bounds(global_x * N_SEQ, dimensions))) * u32(match_centroid(k, global_x));
    // for (var i: u32 = 1u; i < N_SEQ; i = i + 1u) {
    //     local = local + vec4<u32>(textureLoad(pixels, coords(global_x * N_SEQ + i, dimensions), 0).rgb, u32(in_bounds(global_x * N_SEQ + i, dimensions))) * u32(match_centroid(k, global_x));
    // }
    scratch[local_id.x] = local;
    workgroupBarrier();
    
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        workgroupBarrier();
        if (local_id.x >= (1u << i)) {
            local = local + scratch[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        scratch[local_id.x] = local;
    }
    
    var exclusive_prefix = vec4<u32>(0u);
    var flag = FLAG_AGGREGATE_READY;
    
    if (local_id.x == workgroup_size - 1u) {
        atomicStore(&prefix_buffer.data[workgroup_id.x * 8u + 4u], local.r);
        atomicStore(&prefix_buffer.data[workgroup_id.x * 8u + 5u], local.g);
        atomicStore(&prefix_buffer.data[workgroup_id.x * 8u + 6u], local.b);
        atomicStore(&prefix_buffer.data[workgroup_id.x * 8u + 7u], local.a);
        if (workgroup_id.x == 0u) {
            // Special case, group 0 will not need to sum prefix.
            atomicStore(&prefix_buffer.data[workgroup_id.x * 8u + 0u], local.r);
            atomicStore(&prefix_buffer.data[workgroup_id.x * 8u + 1u], local.g);
            atomicStore(&prefix_buffer.data[workgroup_id.x * 8u + 2u], local.b);
            atomicStore(&prefix_buffer.data[workgroup_id.x * 8u + 3u], local.a);
            flag = FLAG_PREFIX_READY;
        }
    }

    storageBarrier();
    if (local_id.x == workgroup_size - 1u) {
        atomicStore(&flag_buffer.state[workgroup_id.x], flag);
    }

    if(workgroup_id.x != 0u) {
        // decoupled loop-back
        var loop_back_ix = workgroup_id.x - 1u;
        loop {
            if(local_id.x == workgroup_size - 1u) {
                shared_flag = atomicLoad(&flag_buffer.state[loop_back_ix]);
            }
            workgroupBarrier();
            flag = shared_flag;
            storageBarrier();

            if (flag == FLAG_PREFIX_READY) {
                if (local_id.x == workgroup_size - 1u) {
                    let r = atomicLoad(&prefix_buffer.data[loop_back_ix * 8u + 0u]);
                    let g = atomicLoad(&prefix_buffer.data[loop_back_ix * 8u + 1u]);
                    let b = atomicLoad(&prefix_buffer.data[loop_back_ix * 8u + 2u]);
                    let a = atomicLoad(&prefix_buffer.data[loop_back_ix * 8u + 3u]);
                    let their_prefix = vec4<u32>(r, g, b, a);
                    // let their_prefix = prefix_buffer.data[loop_back_ix * 2u + 0u];
                    exclusive_prefix = exclusive_prefix + their_prefix;
                }
                break;
            } else if (flag == FLAG_AGGREGATE_READY) {                
                if (local_id.x == workgroup_size - 1u) {                    
                    let r = atomicLoad(&prefix_buffer.data[loop_back_ix * 8u + 4u]);
                    let g = atomicLoad(&prefix_buffer.data[loop_back_ix * 8u + 5u]);
                    let b = atomicLoad(&prefix_buffer.data[loop_back_ix * 8u + 6u]);
                    let a = atomicLoad(&prefix_buffer.data[loop_back_ix * 8u + 7u]);
                    let their_aggregate = vec4<u32>(r, g, b, a);
                    // let their_aggregate = prefix_buffer.data[loop_back_ix * 2u + 1u];
                    exclusive_prefix = their_aggregate + exclusive_prefix;
                }
                loop_back_ix = loop_back_ix - 1u;
            }
            // else spin
        }

        // compute inclusive prefix
        storageBarrier();
        if (local_id.x == workgroup_size - 1u) {
            let inclusive_prefix = exclusive_prefix + local;
            shared_prefix = exclusive_prefix;
            
            atomicStore(&prefix_buffer.data[workgroup_id.x * 8u + 0u], inclusive_prefix.r);
            atomicStore(&prefix_buffer.data[workgroup_id.x * 8u + 1u], inclusive_prefix.g);
            atomicStore(&prefix_buffer.data[workgroup_id.x * 8u + 2u], inclusive_prefix.b);
            atomicStore(&prefix_buffer.data[workgroup_id.x * 8u + 3u], inclusive_prefix.a);
        }
        storageBarrier();
        if (local_id.x == workgroup_size - 1u) {
            atomicStore(&flag_buffer.state[workgroup_id.x], FLAG_PREFIX_READY);
        }
    }

    var prefix = vec4<u32>(0u);
    workgroupBarrier();
    if(workgroup_id.x != 0u){
        prefix = shared_prefix;
    }

    if (workgroup_id.x == last_group_idx() & local_id.x == workgroup_size - 1u) {
        let sum = prefix + scratch[local_id.x];
        if(sum.a > 0u) {
            centroids.data[k * 4u + 0u] = sum.r / sum.a;
            centroids.data[k * 4u + 1u] = sum.g / sum.a;
            centroids.data[k * 4u + 2u] = sum.b / sum.a;
            centroids.data[k * 4u + 3u] = 255u;
            // centroids.data[k * 4u + 0u] = sum.r;
            // centroids.data[k * 4u + 1u] = sum.g;
            // centroids.data[k * 4u + 2u] = sum.b;
            // centroids.data[k * 4u + 3u] = sum.a;
        }
    }
    storageBarrier();
}
