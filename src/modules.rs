use std::num::NonZeroU32;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType, ComputePass,
    ComputePipeline, ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderSource,
    ShaderStages, StorageTextureAccess, Texture, TextureFormat, TextureViewDescriptor,
    TextureViewDimension,
};

use crate::{utils::compute_work_group_count, ColorSpace, Image};

pub(crate) trait Module {
    fn dispatch<'a>(&'a self, compute_pass: &mut ComputePass<'a>);
}

pub(crate) struct ColorConverterModule {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    dispatch_size: (u32, u32),
}

impl ColorConverterModule {
    pub fn new(
        device: &Device,
        color_space: &ColorSpace,
        image: &Image,
        input_texture: &Texture,
        work_texture: &Texture,
    ) -> Self {
        let convert_color_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Convert color shader"),
            source: ShaderSource::Wgsl(
                match color_space {
                    ColorSpace::Lab => include_str!("shaders/converters/rgb_to_lab.wgsl"),
                    ColorSpace::Rgb => include_str!("shaders/converters/rgb8u_to_rgb32f.wgsl"),
                }
                .into(),
            ),
        });

        let convert_color_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Convert color bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba32Float,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let convert_color_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Convert color layout"),
                bind_group_layouts: &[&convert_color_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Convert color pipeline"),
            layout: Some(&convert_color_pipeline_layout),
            module: &convert_color_shader,
            entry_point: "main",
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Convert color bind group"),
            layout: &convert_color_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&input_texture.create_view(
                        &TextureViewDescriptor {
                            label: None,
                            format: Some(TextureFormat::Rgba8Unorm),
                            aspect: wgpu::TextureAspect::All,
                            base_mip_level: 0,
                            mip_level_count: NonZeroU32::new(1),
                            dimension: Some(TextureViewDimension::D2),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&work_texture.create_view(
                        &TextureViewDescriptor {
                            label: None,
                            format: Some(TextureFormat::Rgba32Float),
                            aspect: wgpu::TextureAspect::All,
                            base_mip_level: 0,
                            mip_level_count: NonZeroU32::new(1),
                            dimension: Some(TextureViewDimension::D2),
                            ..Default::default()
                        },
                    )),
                },
            ],
        });

        let dispatch_size = compute_work_group_count(image.dimensions, (16, 16));

        Self {
            pipeline,
            bind_group,
            dispatch_size,
        }
    }
}

impl Module for ColorConverterModule {
    fn dispatch<'a>(&'a self, compute_pass: &mut ComputePass<'a>) {
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.dispatch(self.dispatch_size.0, self.dispatch_size.1, 1);
    }
}

pub(crate) struct ColorReverterModule {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    dispatch_size: (u32, u32),
}

impl ColorReverterModule {
    pub fn new(
        device: &Device,
        color_space: &ColorSpace,
        image: &Image,
        work_texture: &Texture,
        output_texture: &Texture,
    ) -> Self {
        let revert_color_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Revert color shader"),
            source: ShaderSource::Wgsl(
                match color_space {
                    ColorSpace::Lab => include_str!("shaders/converters/lab_to_rgb.wgsl"),
                    ColorSpace::Rgb => include_str!("shaders/converters/rgb32f_to_rgb8u.wgsl"),
                }
                .into(),
            ),
        });

        let revert_color_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Revert color bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba8Unorm,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let revert_color_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Revert color layout"),
                bind_group_layouts: &[&revert_color_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Revert color pipeline"),
            layout: Some(&revert_color_pipeline_layout),
            module: &revert_color_shader,
            entry_point: "main",
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Revert color bind group"),
            layout: &revert_color_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&work_texture.create_view(
                        &TextureViewDescriptor {
                            label: None,
                            format: Some(TextureFormat::Rgba32Float),
                            aspect: wgpu::TextureAspect::All,
                            base_mip_level: 0,
                            mip_level_count: NonZeroU32::new(1),
                            dimension: Some(TextureViewDimension::D2),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&output_texture.create_view(
                        &TextureViewDescriptor {
                            label: None,
                            format: Some(TextureFormat::Rgba8Unorm),
                            aspect: wgpu::TextureAspect::All,
                            base_mip_level: 0,
                            mip_level_count: NonZeroU32::new(1),
                            dimension: Some(TextureViewDimension::D2),
                            ..Default::default()
                        },
                    )),
                },
            ],
        });

        let dispatch_size = compute_work_group_count(image.dimensions, (16, 16));

        Self {
            pipeline,
            bind_group,
            dispatch_size,
        }
    }
}

impl Module for ColorReverterModule {
    fn dispatch<'a>(&'a self, compute_pass: &mut ComputePass<'a>) {
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.dispatch(self.dispatch_size.0, self.dispatch_size.1, 1);
    }
}

pub(crate) struct SwapModule {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    dispatch_size: (u32, u32),
}

impl SwapModule {
    pub fn new(
        device: &Device,
        image: &Image,
        work_texture: &Texture,
        centroid_buffer: &Buffer,
        color_index_buffer: &Buffer,
    ) -> Self {
        let swap_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Swap colors shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/swap.wgsl").into()),
        });

        let swap_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Swap bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let swap_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &swap_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: centroid_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: color_index_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &work_texture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
            ],
        });

        let swap_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Swap pipeline layout"),
            bind_group_layouts: &[&swap_bind_group_layout],
            push_constant_ranges: &[],
        });
        let swap_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Swap pipeline"),
            layout: Some(&swap_pipeline_layout),
            module: &swap_shader,
            entry_point: "main",
        });

        let dispatch_size = compute_work_group_count(image.dimensions, (16, 16));

        Self {
            pipeline: swap_pipeline,
            bind_group: swap_bind_group,
            dispatch_size,
        }
    }
}

impl Module for SwapModule {
    fn dispatch<'a>(&'a self, compute_pass: &mut ComputePass<'a>) {
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.dispatch(self.dispatch_size.0, self.dispatch_size.1, 1);
    }
}

pub(crate) struct FindCentroidModule {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    dispatch_size: (u32, u32),
}

impl FindCentroidModule {
    pub fn new(
        device: &Device,
        image: &Image,
        work_texture: &Texture,
        centroid_buffer: &Buffer,
        color_index_buffer: &Buffer,
    ) -> Self {
        let find_centroid_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Find centroid shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/find_centroid.wgsl").into()),
        });

        let find_centroid_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Find centroid bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let find_centroid_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Pipeline layout"),
                bind_group_layouts: &[&find_centroid_bind_group_layout],
                push_constant_ranges: &[],
            });

        let find_centroid_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Find centroid pipeline"),
            layout: Some(&find_centroid_pipeline_layout),
            module: &find_centroid_shader,
            entry_point: "main",
        });

        let find_centroid_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Find centroid bind group"),
            layout: &find_centroid_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&work_texture.create_view(
                        &TextureViewDescriptor {
                            label: None,
                            format: Some(TextureFormat::Rgba32Float),
                            aspect: wgpu::TextureAspect::All,
                            base_mip_level: 0,
                            mip_level_count: NonZeroU32::new(1),
                            dimension: Some(TextureViewDimension::D2),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: centroid_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: color_index_buffer.as_entire_binding(),
                },
            ],
        });

        let dispatch_size = compute_work_group_count(image.dimensions, (16, 16));

        Self {
            pipeline: find_centroid_pipeline,
            bind_group: find_centroid_bind_group,
            dispatch_size,
        }
    }
}

impl Module for FindCentroidModule {
    fn dispatch<'a>(&'a self, compute_pass: &mut ComputePass<'a>) {
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.dispatch(self.dispatch_size.0, self.dispatch_size.1, 1);
    }
}
