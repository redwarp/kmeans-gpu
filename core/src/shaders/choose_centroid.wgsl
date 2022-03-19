struct Centroids {
    count: u32;
    // Aligned 16. See https://www.w3.org/TR/WGSL/#address-space-layout-constraints
    data: array<vec4<f32>>;
};

struct AtomicBuffer {
    data: array<atomic<u32>>;
};

struct KIndex {
    k: u32;
};

struct Settings {
    n_seq: u32;
    convergence: f32;
};

[[group(0), binding(0)]] var<storage, read_write> centroids: Centroids;
[[group(0), binding(1)]] var color_indices: texture_2d<u32>;
[[group(0), binding(2)]] var pixels: texture_2d<f32>;
[[group(0), binding(3)]] var<storage, read_write> part_id_buffer : AtomicBuffer;
[[group(1), binding(0)]] var<storage, read_write> prefix_buffer: AtomicBuffer;
[[group(1), binding(1)]] var<storage, read_write> flag_buffer: AtomicBuffer;
[[group(1), binding(2)]] var<storage, read_write> convergence: AtomicBuffer;
[[group(1), binding(3)]] var<uniform> settings: Settings;
[[group(2), binding(0)]] var<uniform> k_index: KIndex;

let workgroup_size: u32 = 256u;

var<workgroup> scratch: array<vec4<f32>, workgroup_size>;
var<workgroup> shared_flag: u32;
var<workgroup> part_id: u32;

let FLAG_NOT_READY = 0u;
let FLAG_AGGREGATE_READY = 1u;
let FLAG_PREFIX_READY = 2u;

fn coords(global_x: u32, dimensions: vec2<i32>) -> vec2<i32> {
    return vec2<i32>(vec2<u32>(global_x % u32(dimensions.x), global_x / u32(dimensions.x)));
}

fn last_group_idx() -> u32 {
    return arrayLength(&flag_buffer.data) - 1u;
}

fn in_bounds(global_x: u32, dimensions: vec2<i32>) -> bool {
    let x = global_x % u32(dimensions.x);
    let y = global_x / u32(dimensions.x);
    return x < u32(dimensions.x) && y < u32(dimensions.y);
}

fn match_centroid(k: u32, global_x: u32, width: u32) -> bool {
    let x = global_x % width;
    let y = global_x / width;
    let coords = vec2<i32>(i32(x), i32(y));
    return k == textureLoad(color_indices, coords, 0).r;
}

fn atomicStorePrefixVec(index: u32, value: vec4<f32>) {
    atomicStore(&prefix_buffer.data[index + 0u], bitcast<u32>(value.r));
    atomicStore(&prefix_buffer.data[index + 1u], bitcast<u32>(value.g));
    atomicStore(&prefix_buffer.data[index + 2u], bitcast<u32>(value.b));
    atomicStore(&prefix_buffer.data[index + 3u], bitcast<u32>(value.a));
}

fn atomicLoadPrefixVec(index: u32) -> vec4<f32> {
    let r = bitcast<f32>(atomicLoad(&prefix_buffer.data[index + 0u]));
    let g = bitcast<f32>(atomicLoad(&prefix_buffer.data[index + 1u]));
    let b = bitcast<f32>(atomicLoad(&prefix_buffer.data[index + 2u]));
    let a = bitcast<f32>(atomicLoad(&prefix_buffer.data[index + 3u]));
    return vec4<f32>(r, g, b, a);
}

[[stage(compute), workgroup_size(256)]]
fn main(
    [[builtin(local_invocation_id)]] local_id : vec3<u32>,
) {
    if (local_id.x == 0u) {
        part_id = atomicAdd(&part_id_buffer.data[0], 1u);
    }
    workgroupBarrier();
    let workgroup_x = part_id;
    
    if (local_id.x == 0u) {
        atomicStore(&flag_buffer.data[workgroup_x], FLAG_NOT_READY);
    }
    storageBarrier();

    let k = k_index.k;
    let N_SEQ = settings.n_seq;

    let dimensions = textureDimensions(pixels);
    let width = u32(dimensions.x);
    let global_x = workgroup_x * workgroup_size + local_id.x;

    var local: vec4<f32> = vec4<f32>(0.0);
    for (var i: u32 = 0u; i < N_SEQ; i = i + 1u) {
        let index = global_x * N_SEQ + i;
        if (in_bounds(index, dimensions) && match_centroid(k, index, width)) {
            local = local + vec4<f32>(textureLoad(pixels, coords(index, dimensions), 0).rgb, 1.0);
        }
    }

    scratch[local_id.x] = local;
    workgroupBarrier();
    
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        if (local_id.x >= (1u << i)) {
            local = local + scratch[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        scratch[local_id.x] = local;
        workgroupBarrier();
    }
    
    var exclusive_prefix = vec4<f32>(0.0);
    var flag = FLAG_AGGREGATE_READY;
    
    if (local_id.x == workgroup_size - 1u) {
        atomicStorePrefixVec(workgroup_x * 8u + 4u, local);
        if (workgroup_x == 0u) {
            // Special case, group 0 will not need to sum prefix.
            atomicStorePrefixVec(workgroup_x * 8u + 0u, local);
            flag = FLAG_PREFIX_READY;
        }
    }
    storageBarrier();

    if (local_id.x == workgroup_size - 1u) {
        atomicStore(&flag_buffer.data[workgroup_x], flag);
    }    
    storageBarrier();

    if(workgroup_x != 0u) {
        // decoupled loop-back
        var loop_back_ix = workgroup_x - 1u;
        loop {
            if(local_id.x == workgroup_size - 1u) {
                shared_flag = atomicLoad(&flag_buffer.data[loop_back_ix]);
            }
            workgroupBarrier();
            flag = shared_flag;

            storageBarrier();
            if (flag == FLAG_PREFIX_READY) {
                if (local_id.x == workgroup_size - 1u) {
                    let their_prefix = atomicLoadPrefixVec(loop_back_ix * 8u);
                    exclusive_prefix = exclusive_prefix + their_prefix;
                }
                break;
            } else if (flag == FLAG_AGGREGATE_READY) {                
                if (local_id.x == workgroup_size - 1u) {                    
                    let their_aggregate = atomicLoadPrefixVec(loop_back_ix * 8u + 4u);
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
            
            atomicStorePrefixVec(workgroup_x * 8u + 0u, inclusive_prefix);
            atomicStore(&flag_buffer.data[workgroup_x], FLAG_PREFIX_READY);
        }
        workgroupBarrier();
        storageBarrier();
    }
}

[[stage(compute), workgroup_size(1)]]
fn pick() {
    let dimensions = textureDimensions(pixels);
    let sum = atomicLoadPrefixVec(last_group_idx() * 8u + 0u);
    let k = k_index.k;
    if(sum.a > 0.0) {
        let new_centroid = vec4<f32>(sum.rgb / sum.a, 1.0);
        let previous_centroid = centroids.data[k];

        centroids.data[k] = new_centroid;

        atomicStore(&convergence.data[k], u32(distance(new_centroid, previous_centroid) < settings.convergence));
    } else {
        atomicStore(&convergence.data[k], 0u);
    }

    if (k == centroids.count - 1u) {
        var converge = atomicExchange(&convergence.data[0u], 0u);
        for (var i = 1u; i < centroids.count; i = i + 1u) {
            converge = converge + atomicExchange(&convergence.data[i], 0u);
        }
        atomicStore(&convergence.data[centroids.count], converge);
    }
    
    // Reset part ids for next centroid.
    atomicStore(&part_id_buffer.data[0], 0u);
}