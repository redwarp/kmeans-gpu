pub(crate) fn compute_work_group_count(
    (width, height): (u32, u32),
    (workgroup_width, workgroup_height): (u32, u32),
) -> (u32, u32) {
    let x = width.div_ceil(workgroup_width);
    let y = height.div_ceil(workgroup_height);

    (x, y)
}

/// Compute the next multiple of 256 for texture retrieval padding.
pub(crate) fn padded_bytes_per_row(bytes_per_row: u64) -> u64 {
    let padding = (256 - bytes_per_row % 256) % 256;
    bytes_per_row + padding
}
