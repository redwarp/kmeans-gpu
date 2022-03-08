struct Centroids {
    count: u32;
    data: array<f32>;
};

struct Indices {
    data: array<u32>;
};

struct ColorBuffer {
    data: array<vec4<f32>>;
};

struct StateBuffer {
    state: array<atomic<u32>>;
};

[[group(0), binding(0)]] var<storage, read_write> centroids: Centroids;
[[group(0), binding(1)]] var<storage, read> calculated: Indices;
[[group(0), binding(2)]] var pixels: texture_2d<f32>;
[[group(1), binding(0)]] var<storage, read_write> prefix_buffer: ColorBuffer;
[[group(1), binding(3)]] var<storage, read_write> flag_buffer: StateBuffer;


let workgroup_size: u32 = 256u;
let N_SEQ: u32 = 8u;

var<workgroup> scratch: array<vec4<f32>, workgroup_size>;
var<workgroup> shared_prefix: vec4<f32>;
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

[[stage(compute), workgroup_size(256)]]
fn main(
    [[builtin(local_invocation_id)]] local_id : vec3<u32>,
    [[builtin(workgroup_id)]] workgroup_id : vec3<u32>,
    [[builtin(global_invocation_id)]] global_id : vec3<u32>,
) {
    let dimensions = textureDimensions(pixels);
    let global_x = global_id.x;
    var local: vec4<f32> = textureLoad(pixels, coords(global_x, dimensions), 0);

    for (var i: u32 = 1u; i < N_SEQ; i = i + 1u) {
        local = local + textureLoad(pixels, coords(global_x * N_SEQ + i, dimensions), 0);
    }
    scratch[local_id.x] = local;
    
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        workgroupBarrier();
        if (local_id.x >= (1u << i)) {
            local = local + scratch[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        scratch[local_id.x] = local;
    }
    
    
    var exclusive_prefix = vec4<f32>(0.0);
    var flag = FLAG_AGGREGATE_READY;
    
    if (local_id.x == workgroup_size - 1u) {
        prefix_buffer.data[workgroup_id.x * 2u + 1u] = local;
        if (workgroup_id.x == 0u) {
            // Special case, group 0 will not need to sum prefix.
            prefix_buffer.data[workgroup_id.x * 2u + 0u] = local;
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
                    let their_prefix = prefix_buffer.data[loop_back_ix * 2u + 0u];
                    exclusive_prefix = exclusive_prefix + their_prefix;
                }
                break;
            } else if (flag == FLAG_AGGREGATE_READY) {                
                if (local_id.x == workgroup_size - 1u) {
                    let their_aggregate = prefix_buffer.data[loop_back_ix * 2u + 1u];
                    exclusive_prefix = their_aggregate + exclusive_prefix;
                }
                loop_back_ix = loop_back_ix - 1u;
            }
            // else spin
        }

        // compute inclusive prefix
        if (local_id.x == workgroup_size - 1u) {
            let inclusive_prefix = exclusive_prefix + local;
            shared_prefix = exclusive_prefix;
            prefix_buffer.data[workgroup_id.x * 2u + 0u] = inclusive_prefix;
        }
        storageBarrier();
        if (local_id.x == workgroup_size - 1u) {
            atomicStore(&flag_buffer.state[workgroup_id.x], FLAG_PREFIX_READY);
        }
    }

    var prefix = vec4<f32>(0.0);
    workgroupBarrier();
    if(workgroup_id.x != 0u){
        prefix = shared_prefix;
    }

    var old = vec4<f32>(0.0);
    if (local_id.x > 0u) {
        old = scratch[local_id.x - 1u];
    }

    if (workgroup_id.x == last_group_idx() & local_id.x == workgroup_size - 1u) {
        let sum = prefix + old + local;
        centroids.data[0] = sum.r;
        centroids.data[1] = sum.g;
        centroids.data[2] = sum.b;
    }
}
