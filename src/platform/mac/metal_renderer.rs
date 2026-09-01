use super::metal_atlas::MetalAtlas;
use crate::{
    AtlasTextureId, Background, Bounds, ContentMask, DevicePixels, FramePresentationSample,
    MonochromeSprite, PaintSurface, Path, Point, PolychromeSprite, PrimitiveBatch, Quad,
    RasterCacheHandle, RasterCacheStats, RasterCompositorPresentationSample, RasterTile,
    RasterTileKey, RasterTileLookup, RasterTileRevision, ScaledPixels, Scene, Shadow, Size,
    Surface, Underline, point, size,
};
use anyhow::Result;
use block::ConcreteBlock;
#[cfg(feature = "frame-trace")]
use block::RcBlock;
use cocoa::{
    base::{NO, YES, id, nil},
    foundation::{NSPoint, NSRect, NSSize, NSUInteger},
    quartzcore::{AutoresizingMask, current_media_time},
};

use core_foundation::base::TCFType;
use core_graphics::geometry::CGAffineTransform;
use core_video::{
    metal_texture::CVMetalTextureGetTexture, metal_texture_cache::CVMetalTextureCache,
    pixel_buffer::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
};
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    CAMetalLayer, CommandQueue, MTLBlitOption, MTLOrigin, MTLPixelFormat, MTLResourceOptions,
    MTLSize, NSRange, RenderPassColorAttachmentDescriptorRef,
};
use objc::{self, class, msg_send, sel, sel_impl};
use parking_lot::Mutex;

#[cfg(feature = "frame-trace")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    ffi::c_void,
    mem, ptr,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

#[cfg(feature = "frame-trace")]
type FrameTraceCommandBufferHandler = RcBlock<(&'static metal::CommandBufferRef,), ()>;

// Exported to metal
pub(crate) type PointF = crate::Point<f32>;

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;
const DISPLAY_LINK_TARGET_LEAD_SECONDS: f64 = 0.001;
const MAX_DISPLAY_LINK_TARGET_LEAD: f64 = 0.050;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MetalPresentationMode {
    Asap,
    DisplayLinkTargetEarly,
}

fn parse_metal_presentation_mode(value: Option<&str>) -> MetalPresentationMode {
    match value {
        Some("display-link-target-early") => MetalPresentationMode::DisplayLinkTargetEarly,
        Some("asap") | None => MetalPresentationMode::Asap,
        Some(value) => {
            log::error!(
                "unknown GPUI_METAL_PRESENTATION_MODE value {value:?}; falling back to asap"
            );
            MetalPresentationMode::Asap
        }
    }
}

fn metal_presentation_mode() -> MetalPresentationMode {
    static MODE: OnceLock<MetalPresentationMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        parse_metal_presentation_mode(
            std::env::var("GPUI_METAL_PRESENTATION_MODE")
                .ok()
                .as_deref(),
        )
    })
}

fn mach_host_time_seconds(host_time: u64) -> Option<f64> {
    use mach2::mach_time::{mach_timebase_info, mach_timebase_info_data_t};

    static TIMEBASE: OnceLock<Option<mach_timebase_info_data_t>> = OnceLock::new();
    let timebase = TIMEBASE
        .get_or_init(|| {
            let mut timebase = mach_timebase_info_data_t { numer: 0, denom: 0 };
            // SAFETY: timebase is a valid out pointer for mach_timebase_info.
            let result = unsafe { mach_timebase_info(&mut timebase) };
            (result == 0 && timebase.denom != 0).then_some(timebase)
        })
        .as_ref()?;
    Some(host_time as f64 * f64::from(timebase.numer) / f64::from(timebase.denom) / 1_000_000_000.)
}

fn valid_display_link_target_seconds(target_host_time: u64, now_host_time: u64) -> Option<f64> {
    let target = mach_host_time_seconds(target_host_time)?;
    let now = mach_host_time_seconds(now_host_time)?;
    valid_target_seconds(target, now)
}

fn valid_target_seconds(target: f64, now: f64) -> Option<f64> {
    let target_lead = target - now;
    let presentation_time = target - DISPLAY_LINK_TARGET_LEAD_SECONDS;
    (target_lead > 0. && target_lead <= MAX_DISPLAY_LINK_TARGET_LEAD && presentation_time > now)
        .then_some(presentation_time)
}

pub type Context = Arc<Mutex<InstanceBufferPool>>;
pub type Renderer = MetalRenderer;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Viewport {
    pub size: Size<DevicePixels>,
    pub origin: Point<ScaledPixels>,
}

struct CachedRasterAllocation {
    texture: metal::Texture,
    bytes: usize,
}

struct CachedRasterTexture {
    allocation: Arc<CachedRasterAllocation>,
    source_origin: Point<DevicePixels>,
    source_size: Size<DevicePixels>,
    gutter: u32,
    last_used_frame: u64,
}

type CachedRasterTextureKey = (u64, u64, u64);

fn cached_raster_texture_key(
    cache_id: u64,
    tile_key: u64,
    revision: u64,
) -> CachedRasterTextureKey {
    (cache_id, tile_key, revision)
}

struct RasterNamespace {
    owner: std::sync::Weak<crate::raster_cache::RasterCacheIdentity>,
    stats: RasterCacheStats,
    comparisons: Arc<Mutex<RasterComparisonStats>>,
}

struct RasterCompositorLayer {
    container: id,
    layer: metal::MetalLayer,
    handle: crate::RasterCompositorTransformHandle,
    captured_transform: crate::RasterCompositorTransform,
    clip_bounds: Bounds<ScaledPixels>,
    raster_bounds: Bounds<ScaledPixels>,
    last_applied: Option<AppliedRasterCompositorTransform>,
    last_presented_revision: Option<u64>,
}

#[derive(Clone, Copy)]
struct AppliedRasterCompositorTransform {
    revision: u64,
    position: NSPoint,
    ratio: f32,
    updated_at: Instant,
}

impl Drop for RasterCompositorLayer {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.container, removeFromSuperlayer];
            let _: () = msg_send![self.container, release];
        }
    }
}

fn new_raster_compositor_layer(
    device: &metal::DeviceRef,
    root_layer: &metal::MetalLayerRef,
    handle: crate::RasterCompositorTransformHandle,
    captured_transform: crate::RasterCompositorTransform,
    clip_bounds: Bounds<ScaledPixels>,
    raster_bounds: Bounds<ScaledPixels>,
) -> RasterCompositorLayer {
    let layer = metal::MetalLayer::new();
    layer.set_device(device);
    layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    layer.set_opaque(false);
    layer.set_framebuffer_only(true);
    layer.set_maximum_drawable_count(3);
    let container: id = unsafe { msg_send![class!(CALayer), new] };
    unsafe {
        let _: () = msg_send![container, setMasksToBounds: YES];
        let _: () = msg_send![container, setGeometryFlipped: YES];
        let _: () = msg_send![layer.as_ref(), setGeometryFlipped: YES];
        let _: () = msg_send![layer.as_ref(), setAnchorPoint: NSPoint { x: 0., y: 0. }];
        let _: () = msg_send![container, addSublayer: layer.as_ref()];
        let _: () = msg_send![root_layer, addSublayer: container];
    }
    let compositor = RasterCompositorLayer {
        container,
        layer,
        handle,
        captured_transform,
        clip_bounds,
        raster_bounds,
        last_applied: None,
        last_presented_revision: None,
    };
    configure_raster_compositor_geometry(&compositor, root_layer);
    compositor
}

fn configure_raster_compositor_geometry(
    compositor: &RasterCompositorLayer,
    root_layer: &metal::MetalLayerRef,
) {
    let contents_scale: f64 = unsafe { msg_send![root_layer, contentsScale] };
    let contents_scale = contents_scale.max(1.);
    let root_bounds: NSRect = unsafe { msg_send![root_layer, bounds] };
    let root_flipped: cocoa::base::BOOL = unsafe { msg_send![root_layer, isGeometryFlipped] };
    let clip_x = compositor.clip_bounds.origin.x.0 as f64 / contents_scale;
    let clip_y_top = compositor.clip_bounds.origin.y.0 as f64 / contents_scale;
    let clip_width = compositor.clip_bounds.size.width.0 as f64 / contents_scale;
    let clip_height = compositor.clip_bounds.size.height.0 as f64 / contents_scale;
    let clip_y = if root_flipped == YES {
        clip_y_top
    } else {
        root_bounds.size.height - clip_y_top - clip_height
    };
    let raster_width = compositor.raster_bounds.size.width.0 as f64 / contents_scale;
    let raster_height = compositor.raster_bounds.size.height.0 as f64 / contents_scale;
    unsafe {
        let _: () = msg_send![compositor.container, setFrame: NSRect {
            origin: NSPoint { x: clip_x, y: clip_y },
            size: NSSize { width: clip_width, height: clip_height },
        }];
        let _: () = msg_send![compositor.layer.as_ref(), setBounds: NSRect {
            origin: NSPoint { x: 0., y: 0. },
            size: NSSize { width: raster_width, height: raster_height },
        }];
        let _: () = msg_send![compositor.layer.as_ref(), setContentsScale: contents_scale];
    }
    unsafe {
        let _: () = msg_send![compositor.layer.as_ref(), setDrawableSize: NSSize {
            width: compositor.raster_bounds.size.width.0.ceil() as f64,
            height: compositor.raster_bounds.size.height.0.ceil() as f64,
        }];
    }
}

fn observe_raster_compositor_presentation(
    compositor: &mut RasterCompositorLayer,
    samples: &Arc<Mutex<Vec<RasterCompositorPresentationSample>>>,
) {
    let Some(applied) = compositor.last_applied else {
        return;
    };
    if compositor.last_presented_revision == Some(applied.revision) {
        return;
    }
    let presentation: id = unsafe { msg_send![compositor.layer.as_ref(), presentationLayer] };
    if presentation == nil {
        return;
    }
    let position: NSPoint = unsafe { msg_send![presentation, position] };
    let transform: CGAffineTransform = unsafe { msg_send![presentation, affineTransform] };
    if (position.x - applied.position.x).abs() > 0.25
        || (position.y - applied.position.y).abs() > 0.25
        || (transform.a - f64::from(applied.ratio)).abs() > 0.0005
        || (transform.d - f64::from(applied.ratio)).abs() > 0.0005
    {
        return;
    }
    compositor.last_presented_revision = Some(applied.revision);
    samples.lock().push(RasterCompositorPresentationSample {
        compositor_id: compositor.handle.id(),
        revision: applied.revision,
        updated_at: applied.updated_at,
        presented_at: Instant::now(),
    });
}

fn apply_raster_compositor_transform(
    compositor: &mut RasterCompositorLayer,
    contents_scale: f32,
    samples: &Arc<Mutex<Vec<RasterCompositorPresentationSample>>>,
    flush_transaction: bool,
) {
    observe_raster_compositor_presentation(compositor, samples);
    let (revision, current) = compositor.handle.snapshot();
    if compositor
        .last_applied
        .is_some_and(|applied| applied.revision == revision)
    {
        return;
    }
    let ratio = current.scale / compositor.captured_transform.scale;
    let raster_origin_x = compositor.raster_bounds.origin.x.0 / contents_scale;
    let raster_origin_y = compositor.raster_bounds.origin.y.0 / contents_scale;
    let clip_origin_x = compositor.clip_bounds.origin.x.0 / contents_scale;
    let clip_origin_y = compositor.clip_bounds.origin.y.0 / contents_scale;
    let position = NSPoint {
        x: (ratio * (raster_origin_x - compositor.captured_transform.translation.x.0)
            + current.translation.x.0
            - clip_origin_x) as f64,
        y: (ratio * (raster_origin_y - compositor.captured_transform.translation.y.0)
            + current.translation.y.0
            - clip_origin_y) as f64,
    };
    let raster_size = NSSize {
        width: compositor.raster_bounds.size.width.0 as f64 / f64::from(contents_scale),
        height: compositor.raster_bounds.size.height.0 as f64 / f64::from(contents_scale),
    };
    let clip_size = NSSize {
        width: compositor.clip_bounds.size.width.0 as f64 / f64::from(contents_scale),
        height: compositor.clip_bounds.size.height.0 as f64 / f64::from(contents_scale),
    };
    if !raster_compositor_surface_covers_clip(position, ratio, raster_size, clip_size) {
        // Keep the last complete frame until the ordinary scene pass captures a new surface.
        // Applying an out-of-coverage transform would expose transparent pixels at the edge.
        return;
    }
    let transform = CGAffineTransform::new(ratio as f64, 0., 0., ratio as f64, 0., 0.);
    unsafe {
        let _: () = msg_send![class!(CATransaction), begin];
        let _: () = msg_send![class!(CATransaction), setDisableActions: YES];
        let _: () = msg_send![compositor.layer.as_ref(), setAffineTransform: transform];
        let _: () = msg_send![compositor.layer.as_ref(), setPosition: position];
        let _: () = msg_send![class!(CATransaction), commit];
        if flush_transaction {
            // Input-triggered camera transforms must reach the render server before the next
            // display deadline. A committed transaction may otherwise remain queued until the
            // following run-loop boundary, adding one avoidable presentation interval.
            let _: () = msg_send![class!(CATransaction), flush];
        }
    }
    compositor.last_applied = Some(AppliedRasterCompositorTransform {
        revision,
        position,
        ratio,
        updated_at: compositor
            .handle
            .updated_at(revision)
            .unwrap_or_else(Instant::now),
    });
}

fn raster_compositor_surface_covers_clip(
    position: NSPoint,
    ratio: f32,
    raster_size: NSSize,
    clip_size: NSSize,
) -> bool {
    const COVERAGE_EPSILON: f64 = 0.01;
    if !ratio.is_finite() || ratio <= 0. {
        return false;
    }
    let ratio = f64::from(ratio);
    position.x <= COVERAGE_EPSILON
        && position.y <= COVERAGE_EPSILON
        && position.x + raster_size.width * ratio + COVERAGE_EPSILON >= clip_size.width
        && position.y + raster_size.height * ratio + COVERAGE_EPSILON >= clip_size.height
}

#[derive(Default)]
struct RasterComparisonStats {
    samples: u64,
    min_ssim_ppb: u32,
    p99_channel_error: u8,
    max_channel_error: u8,
}

impl RasterNamespace {
    fn new(cache: &RasterCacheHandle) -> Self {
        Self {
            owner: cache.weak_identity(),
            stats: RasterCacheStats::default(),
            comparisons: Arc::new(Mutex::new(RasterComparisonStats::default())),
        }
    }
}

pub unsafe fn new_renderer(
    context: self::Context,
    _native_window: *mut c_void,
    _native_view: *mut c_void,
    _bounds: crate::Size<f32>,
    _transparent: bool,
) -> Renderer {
    MetalRenderer::new(context)
}

pub(crate) struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<metal::Buffer>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

pub(crate) struct InstanceBuffer {
    metal_buffer: metal::Buffer,
    size: usize,
}

struct DeferredRender {
    command_buffer: metal::CommandBuffer,
    instance_buffer: InstanceBuffer,
}

enum PreparedRasterBatchTexture {
    AlreadyReady,
    Allocated(metal::Texture),
}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    pub(crate) fn acquire(&mut self, device: &metal::Device) -> InstanceBuffer {
        let buffer = self.buffers.pop().unwrap_or_else(|| {
            device.new_buffer(
                self.buffer_size as u64,
                MTLResourceOptions::StorageModeManaged,
            )
        });
        InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        }
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer)
        }
    }
}

pub(crate) struct MetalRenderer {
    device: metal::Device,
    layer: metal::MetalLayer,
    presents_with_transaction: bool,
    presentation_mode: MetalPresentationMode,
    display_link_target_host_time: Option<u64>,
    command_queue: CommandQueue,
    paths_rasterization_pipeline_state: metal::RenderPipelineState,
    path_sprites_pipeline_state: metal::RenderPipelineState,
    shadows_pipeline_state: metal::RenderPipelineState,
    quads_pipeline_state: metal::RenderPipelineState,
    underlines_pipeline_state: metal::RenderPipelineState,
    monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    surfaces_pipeline_state: metal::RenderPipelineState,
    raster_tiles_pipeline_state: metal::RenderPipelineState,
    unit_vertices: metal::Buffer,
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: core_video::metal_texture_cache::CVMetalTextureCache,
    path_intermediate_texture: Option<metal::Texture>,
    path_intermediate_msaa_texture: Option<metal::Texture>,
    path_sample_count: u32,
    // Keep multiple content revisions for the same logical tile alive. A retained compositor
    // frame can still reference the previous revision while a deferred command buffer builds
    // its replacement. Keying only by `(cache, tile)` removed the visible revision before the
    // replacement was ready and produced transparent strips and mixed frames during hover,
    // selection, pan and zoom.
    raster_textures: HashMap<CachedRasterTextureKey, CachedRasterTexture>,
    raster_namespaces: HashMap<u64, RasterNamespace>,
    raster_compositor_layers: HashMap<u64, RasterCompositorLayer>,
    raster_compositor_deferred_renders: Vec<DeferredRender>,
    frame_index: u64,
    presented_frame_samples: Arc<Mutex<Vec<FramePresentationSample>>>,
    raster_compositor_presentation_samples: Arc<Mutex<Vec<RasterCompositorPresentationSample>>>,
    #[cfg(feature = "frame-trace")]
    frame_trace_renderer_instance_id: u64,
    #[cfg(feature = "frame-trace")]
    frame_trace_presentation_queue_depth: Arc<AtomicU64>,
    #[cfg(feature = "frame-trace")]
    frame_trace_scheduled_handler: FrameTraceCommandBufferHandler,
    #[cfg(feature = "frame-trace")]
    frame_trace_completed_handler: FrameTraceCommandBufferHandler,
    #[cfg(feature = "frame-trace")]
    frame_trace_last_logical_frame_id: u64,
}

#[repr(C)]
pub struct PathRasterizationVertex {
    pub xy_position: Point<ScaledPixels>,
    pub st_position: Point<f32>,
}

#[derive(Clone)]
#[repr(C)]
pub struct PathRasterizationStyle {
    pub color: Background,
    pub bounds: Bounds<ScaledPixels>,
}

impl MetalRenderer {
    pub fn new(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>) -> Self {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        let device = if let Some(d) = metal::Device::all()
            .into_iter()
            .min_by_key(|d| (d.is_removable(), !d.is_low_power()))
        {
            d
        } else {
            // For some reason `all()` can return an empty list, see https://github.com/zed-industries/zed/issues/37689
            // In that case, we fall back to the system default device.
            log::error!(
                "Unable to enumerate Metal devices; attempting to use system default device"
            );
            metal::Device::system_default().unwrap_or_else(|| {
                log::error!("unable to access a compatible graphics device");
                std::process::exit(1);
            })
        };

        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_opaque(false);
        // Two drawables bound presentation latency to one queued frame. The previous third
        // drawable improved throughput only when GPU work approached the refresh budget, while
        // adding a full frame of latency to interactive surfaces with short GPU workloads.
        layer.set_maximum_drawable_count(2);
        unsafe {
            let _: () = msg_send![&*layer, setAllowsNextDrawableTimeout: NO];
            let _: () = msg_send![&*layer, setNeedsDisplayOnBoundsChange: YES];
            let _: () = msg_send![
                &*layer,
                setAutoresizingMask: AutoresizingMask::WIDTH_SIZABLE
                    | AutoresizingMask::HEIGHT_SIZABLE
            ];
        }
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .new_library_with_source(&SHADERS_SOURCE_FILE, &metal::CompileOptions::new())
            .expect("error building metal library");
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .new_library_with_data(SHADERS_METALLIB)
            .expect("error building metal library");

        fn to_float2_bits(point: PointF) -> u64 {
            let mut output = point.y.to_bits() as u64;
            output <<= 32;
            output |= point.x.to_bits() as u64;
            output
        }

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        let unit_vertices = device.new_buffer_with_data(
            unit_vertices.as_ptr() as *const c_void,
            mem::size_of_val(&unit_vertices) as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        );
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let raster_tiles_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "raster_tiles",
            "raster_tile_vertex",
            "raster_tile_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );

        let command_queue = device.new_command_queue();
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone()));
        let core_video_texture_cache =
            CVMetalTextureCache::new(None, device.clone(), None).unwrap();

        #[cfg(feature = "frame-trace")]
        let frame_trace_presentation_queue_depth = Arc::new(AtomicU64::new(0));
        #[cfg(feature = "frame-trace")]
        let frame_trace_scheduled_handler = ConcreteBlock::new({
            let presentation_queue_depth = frame_trace_presentation_queue_depth.clone();
            move |command_buffer: &'static metal::CommandBufferRef| {
                if !crate::frame_trace::is_detailed_enabled() {
                    return;
                }
                let mut event = crate::frame_trace::FrameTraceEvent::now(
                    crate::frame_trace::FrameTraceEventKind::GpuScheduled,
                );
                event.command_buffer_id = command_buffer.as_ptr() as usize as u64;
                event.presentation_queue_depth = presentation_queue_depth.load(Ordering::Acquire);
                crate::frame_trace::record(event);
            }
        })
        .copy();
        #[cfg(feature = "frame-trace")]
        let frame_trace_completed_handler = ConcreteBlock::new({
            let presentation_queue_depth = frame_trace_presentation_queue_depth.clone();
            move |command_buffer: &'static metal::CommandBufferRef| {
                if !crate::frame_trace::is_detailed_enabled() {
                    return;
                }
                let callback_observed_ns = crate::frame_trace::monotonic_time_ns();
                let gpu_start_seconds: f64 = unsafe { msg_send![command_buffer, GPUStartTime] };
                let gpu_end_seconds: f64 = unsafe { msg_send![command_buffer, GPUEndTime] };
                let gpu_start_time_ns = crate::frame_trace::host_seconds_to_ns(gpu_start_seconds);
                let gpu_end_time_ns = crate::frame_trace::host_seconds_to_ns(gpu_end_seconds);
                let mut event = crate::frame_trace::FrameTraceEvent::now(
                    crate::frame_trace::FrameTraceEventKind::GpuCompleted,
                );
                event.timestamp_ns = if gpu_end_time_ns == 0 {
                    callback_observed_ns
                } else {
                    gpu_end_time_ns
                };
                event.callback_observed_ns = callback_observed_ns;
                event.command_buffer_id = command_buffer.as_ptr() as usize as u64;
                event.presentation_queue_depth = presentation_queue_depth.load(Ordering::Acquire);
                event.gpu_start_time_ns = gpu_start_time_ns;
                event.gpu_end_time_ns = gpu_end_time_ns;
                if gpu_start_time_ns == 0
                    || gpu_end_time_ns == 0
                    || gpu_end_time_ns < gpu_start_time_ns
                {
                    event.flags |= crate::frame_trace::FLAG_GPU_TIMESTAMPS_INVALID;
                }
                crate::frame_trace::record(event);
            }
        })
        .copy();

        Self {
            device,
            layer,
            presents_with_transaction: false,
            presentation_mode: metal_presentation_mode(),
            display_link_target_host_time: None,
            command_queue,
            paths_rasterization_pipeline_state,
            path_sprites_pipeline_state,
            shadows_pipeline_state,
            quads_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            surfaces_pipeline_state,
            raster_tiles_pipeline_state,
            unit_vertices,
            instance_buffer_pool,
            sprite_atlas,
            core_video_texture_cache,
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            path_sample_count: PATH_SAMPLE_COUNT,
            raster_textures: HashMap::new(),
            raster_namespaces: HashMap::new(),
            raster_compositor_layers: HashMap::new(),
            raster_compositor_deferred_renders: Vec::new(),
            frame_index: 0,
            presented_frame_samples: Arc::new(Mutex::new(Vec::new())),
            raster_compositor_presentation_samples: Arc::new(Mutex::new(Vec::new())),
            #[cfg(feature = "frame-trace")]
            frame_trace_renderer_instance_id: crate::frame_trace::next_renderer_instance_id(),
            #[cfg(feature = "frame-trace")]
            frame_trace_presentation_queue_depth,
            #[cfg(feature = "frame-trace")]
            frame_trace_scheduled_handler,
            #[cfg(feature = "frame-trace")]
            frame_trace_completed_handler,
            #[cfg(feature = "frame-trace")]
            frame_trace_last_logical_frame_id: 0,
        }
    }

    #[cfg(feature = "frame-trace")]
    fn frame_trace_event(
        &self,
        kind: crate::frame_trace::FrameTraceEventKind,
        scene: &Scene,
        drawable_id: u64,
        command_buffer_id: u64,
    ) -> crate::frame_trace::FrameTraceEvent {
        let mut event = crate::frame_trace::FrameTraceEvent::now(kind);
        event.input_sequence_id = scene.frame_trace_input_sequence_id;
        event.logical_frame_id = scene.frame_trace_logical_frame_id;
        event.renderer_instance_id = self.frame_trace_renderer_instance_id;
        event.renderer_frame_id = self.frame_index;
        event.drawable_id = drawable_id;
        event.command_buffer_id = command_buffer_id;
        event.target_display_time_ns = crate::frame_trace::latest_display_target_ns();
        event.display_tick_sequence = crate::frame_trace::latest_display_tick_sequence();
        event.presentation_queue_depth = self
            .frame_trace_presentation_queue_depth
            .load(Ordering::Acquire);
        if scene.frame_trace_logical_frame_id != 0
            && scene.frame_trace_logical_frame_id == self.frame_trace_last_logical_frame_id
        {
            event.flags |= crate::frame_trace::FLAG_REUSED_SCENE;
        }
        if event.target_display_time_ns == 0 {
            event.flags |= crate::frame_trace::FLAG_DISPLAY_TARGET_INVALID;
        }
        event
    }

    pub fn raster_tile_lookup(
        &mut self,
        cache: &RasterCacheHandle,
        key: RasterTileKey,
        revision: RasterTileRevision,
    ) -> RasterTileLookup {
        self.prune_released_raster_caches();
        let namespace = self
            .raster_namespaces
            .entry(cache.id())
            .or_insert_with(|| RasterNamespace::new(cache));
        if let Some(tile) = self.raster_textures.get_mut(&cached_raster_texture_key(
            cache.id(),
            key.value(),
            revision.value(),
        )) {
            tile.last_used_frame = self.frame_index;
            namespace.stats.hits += 1;
            return RasterTileLookup::Hit(cache.tile_hit(key, revision, tile.gutter));
        }
        namespace.stats.misses += 1;
        RasterTileLookup::Miss(cache.tile_miss(key, revision))
    }

    pub fn raster_cache_stats(&self, cache: &RasterCacheHandle) -> RasterCacheStats {
        self.raster_namespaces
            .get(&cache.id())
            .map(|namespace| {
                let mut stats = namespace.stats;
                let comparisons = namespace.comparisons.lock();
                stats.comparison_samples = comparisons.samples;
                stats.comparison_min_ssim_ppb = comparisons.min_ssim_ppb;
                stats.comparison_p99_channel_error = comparisons.p99_channel_error;
                stats.comparison_max_channel_error = comparisons.max_channel_error;
                stats
            })
            .unwrap_or_default()
    }

    pub fn release_raster_cache(&mut self, cache: &RasterCacheHandle) {
        self.raster_textures
            .retain(|(cache_id, _, _), _| *cache_id != cache.id());
        self.raster_namespaces.remove(&cache.id());
    }

    fn prune_released_raster_caches(&mut self) {
        let released = self
            .raster_namespaces
            .iter()
            .filter_map(|(cache_id, namespace)| {
                namespace.owner.upgrade().is_none().then_some(*cache_id)
            })
            .collect::<Vec<_>>();
        for cache_id in released {
            self.raster_textures
                .retain(|(texture_cache_id, _, _), _| *texture_cache_id != cache_id);
            self.raster_namespaces.remove(&cache_id);
        }
    }

    pub fn take_presented_frame_samples(&self) -> Vec<FramePresentationSample> {
        mem::take(&mut *self.presented_frame_samples.lock())
    }

    pub fn take_raster_compositor_presentation_samples(
        &self,
    ) -> Vec<RasterCompositorPresentationSample> {
        mem::take(&mut *self.raster_compositor_presentation_samples.lock())
    }

    fn commit_deferred_renders(&self, deferred_renders: Vec<DeferredRender>) {
        for deferred in deferred_renders {
            let instance_buffer_pool = self.instance_buffer_pool.clone();
            let instance_buffer = Cell::new(Some(deferred.instance_buffer));
            let release = ConcreteBlock::new(move |_| {
                if let Some(instance_buffer) = instance_buffer.take() {
                    instance_buffer_pool.lock().release(instance_buffer);
                }
            });
            let release = release.copy();
            deferred.command_buffer.add_completed_handler(&release);
            deferred.command_buffer.commit();
        }
    }

    pub fn layer(&self) -> &metal::MetalLayerRef {
        &self.layer
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer.as_ptr()
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        if presents_with_transaction {
            self.display_link_target_host_time = None;
        }
        self.layer
            .set_presents_with_transaction(presents_with_transaction);
    }

    pub fn set_display_link_target(&mut self, target_host_time: Option<u64>) {
        self.display_link_target_host_time = target_host_time;
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        let size = NSSize {
            width: size.width.0 as f64,
            height: size.height.0 as f64,
        };
        unsafe {
            let _: () = msg_send![
                self.layer(),
                setDrawableSize: size
            ];
        }
        let device_pixels_size = Size {
            width: DevicePixels(size.width as i32),
            height: DevicePixels(size.height as i32),
        };
        self.update_path_intermediate_textures(device_pixels_size);
    }

    fn update_path_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        // We are uncertain when this happens, but sometimes size can be 0 here. Most likely before
        // the layout pass on window creation. Zero-sized texture creation causes SIGABRT.
        // https://github.com/zed-industries/zed/issues/36229
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.path_intermediate_texture = None;
            self.path_intermediate_msaa_texture = None;
            return;
        }

        (
            self.path_intermediate_texture,
            self.path_intermediate_msaa_texture,
        ) = self.allocate_path_intermediate_textures(size);
    }

    pub fn update_transparency(&self, _transparent: bool) {
        // todo(mac)?
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    pub fn update_raster_compositor_transforms(&mut self) {
        let contents_scale: f64 = unsafe { msg_send![self.layer.as_ref(), contentsScale] };
        let samples = self.raster_compositor_presentation_samples.clone();
        for compositor in self.raster_compositor_layers.values_mut() {
            apply_raster_compositor_transform(
                compositor,
                contents_scale.max(1.) as f32,
                &samples,
                false,
            );
        }
    }

    pub fn latch_raster_compositor_transform(&mut self, compositor_id: u64) -> bool {
        let contents_scale: f64 = unsafe { msg_send![self.layer.as_ref(), contentsScale] };
        let samples = self.raster_compositor_presentation_samples.clone();
        let Some(compositor) = self.raster_compositor_layers.get_mut(&compositor_id) else {
            return false;
        };
        apply_raster_compositor_transform(
            compositor,
            contents_scale.max(1.) as f32,
            &samples,
            true,
        );
        true
    }

    fn synchronize_raster_compositors(&mut self, scene: &Scene) {
        let active_ids = scene
            .raster_compositor_surfaces
            .iter()
            .map(|surface| surface.handle.id())
            .collect::<HashSet<_>>();
        self.raster_compositor_layers
            .retain(|id, _| active_ids.contains(id));

        for surface in &scene.raster_compositor_surfaces {
            let id = surface.handle.id();
            if !self.raster_compositor_layers.contains_key(&id) {
                let compositor = new_raster_compositor_layer(
                    &self.device,
                    &self.layer,
                    surface.handle.clone(),
                    surface.captured_transform,
                    surface.clip_bounds,
                    surface.raster_bounds,
                );
                self.raster_compositor_layers.insert(id, compositor);
            }

            let layer = {
                let compositor = self.raster_compositor_layers.get_mut(&id).unwrap();
                if compositor.captured_transform != surface.captured_transform
                    || compositor.clip_bounds != surface.clip_bounds
                    || compositor.raster_bounds != surface.raster_bounds
                {
                    compositor.last_applied = None;
                }
                compositor.handle = surface.handle.clone();
                compositor.captured_transform = surface.captured_transform;
                compositor.clip_bounds = surface.clip_bounds;
                compositor.raster_bounds = surface.raster_bounds;
                configure_raster_compositor_geometry(compositor, &self.layer);
                compositor.layer.clone()
            };
            self.draw_scene_to_compositor(&surface.scene, &layer, surface.raster_bounds);
        }
        self.update_raster_compositor_transforms();
    }

    fn draw_scene_to_compositor(
        &mut self,
        scene: &Scene,
        layer: &metal::MetalLayerRef,
        raster_bounds: Bounds<ScaledPixels>,
    ) {
        let Some(drawable) = layer.next_drawable() else {
            return;
        };
        let viewport = Viewport {
            size: size(
                DevicePixels(raster_bounds.size.width.0.ceil() as i32),
                DevicePixels(raster_bounds.size.height.0.ceil() as i32),
            ),
            origin: raster_bounds.origin,
        };
        loop {
            let mut instance_buffer = self.instance_buffer_pool.lock().acquire(&self.device);
            match self.draw_primitives(scene, &mut instance_buffer, drawable, viewport) {
                Ok((command_buffer, deferred_renders)) => {
                    let instance_buffer_pool = self.instance_buffer_pool.clone();
                    let instance_buffer = Cell::new(Some(instance_buffer));
                    let release = ConcreteBlock::new(move |_| {
                        if let Some(instance_buffer) = instance_buffer.take() {
                            instance_buffer_pool.lock().release(instance_buffer);
                        }
                    });
                    let release = release.copy();
                    command_buffer.add_completed_handler(&release);
                    command_buffer.present_drawable(drawable);
                    command_buffer.commit();
                    self.raster_compositor_deferred_renders
                        .extend(deferred_renders);
                    break;
                }
                Err(error) => {
                    log::error!("failed to draw raster compositor surface: {error}");
                    let mut pool = self.instance_buffer_pool.lock();
                    let size = pool.buffer_size;
                    if size >= 256 * 1024 * 1024 {
                        break;
                    }
                    pool.reset(size * 2);
                }
            }
        }
    }

    pub fn draw(&mut self, scene: &Scene) {
        self.frame_index = self.frame_index.wrapping_add(1);
        let display_link_target_host_time = self.display_link_target_host_time.take();
        {
            const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;
            let required = required_instance_buffer_size(scene);
            if required > MAX_INSTANCE_BUFFER_SIZE {
                log::error!(
                    "scene instance data requires {} bytes, above the {} byte renderer limit",
                    required,
                    MAX_INSTANCE_BUFFER_SIZE
                );
            }
            let mut pool = self.instance_buffer_pool.lock();
            if pool.buffer_size < required {
                let target = required
                    .checked_next_power_of_two()
                    .unwrap_or(MAX_INSTANCE_BUFFER_SIZE)
                    .min(MAX_INSTANCE_BUFFER_SIZE);
                pool.reset(target);
                log::info!(
                    "pre-sized instance buffer to {} bytes for a {} byte scene",
                    target,
                    required
                );
            }
        }
        self.synchronize_raster_compositors(scene);
        let layer = self.layer.clone();
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let viewport = Viewport {
            size: viewport_size,
            origin: point(ScaledPixels(0.), ScaledPixels(0.)),
        };
        #[cfg(feature = "frame-trace")]
        let frame_trace_detailed = crate::frame_trace::is_detailed_enabled();
        #[cfg(feature = "frame-trace")]
        let next_drawable_started_ns =
            frame_trace_detailed.then(crate::frame_trace::monotonic_time_ns);
        let drawable = if let Some(drawable) = layer.next_drawable() {
            drawable
        } else {
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            let deferred = mem::take(&mut self.raster_compositor_deferred_renders);
            self.commit_deferred_renders(deferred);
            return;
        };
        #[cfg(feature = "frame-trace")]
        if let Some(next_drawable_started_ns) = next_drawable_started_ns {
            let mut event = self.frame_trace_event(
                crate::frame_trace::FrameTraceEventKind::DrawableAcquired,
                scene,
                crate::frame_trace::encode_drawable_id(drawable.drawable_id() as u64),
                0,
            );
            event.next_drawable_wait_ns =
                event.timestamp_ns.saturating_sub(next_drawable_started_ns);
            crate::frame_trace::record(event);
        }

        loop {
            let mut instance_buffer = self.instance_buffer_pool.lock().acquire(&self.device);

            let command_buffers =
                self.draw_primitives(scene, &mut instance_buffer, drawable, viewport);

            match command_buffers {
                Ok((command_buffer, deferred_renders)) => {
                    let instance_buffer_pool = self.instance_buffer_pool.clone();
                    let instance_buffer = Cell::new(Some(instance_buffer));
                    let block = ConcreteBlock::new(move |_| {
                        if let Some(instance_buffer) = instance_buffer.take() {
                            instance_buffer_pool.lock().release(instance_buffer);
                        }
                    });
                    let block = block.copy();
                    command_buffer.add_completed_handler(&block);

                    #[cfg(feature = "frame-trace")]
                    if frame_trace_detailed {
                        command_buffer.add_scheduled_handler(&self.frame_trace_scheduled_handler);
                        command_buffer.add_completed_handler(&self.frame_trace_completed_handler);
                    }

                    let frame_id = self.frame_index;
                    let submitted_at = Instant::now();
                    let samples = self.presented_frame_samples.clone();
                    let measured_command_buffer = command_buffer.to_owned();
                    #[cfg(feature = "frame-trace")]
                    let trace_renderer_instance_id = self.frame_trace_renderer_instance_id;
                    #[cfg(feature = "frame-trace")]
                    let trace_renderer_frame_id = self.frame_index;
                    #[cfg(feature = "frame-trace")]
                    let trace_logical_frame_id = scene.frame_trace_logical_frame_id;
                    #[cfg(feature = "frame-trace")]
                    let trace_input_sequence_id = scene.frame_trace_input_sequence_id;
                    #[cfg(feature = "frame-trace")]
                    let trace_drawable_id =
                        crate::frame_trace::encode_drawable_id(drawable.drawable_id() as u64);
                    #[cfg(feature = "frame-trace")]
                    let trace_command_buffer_id = command_buffer.as_ptr() as usize as u64;
                    #[cfg(feature = "frame-trace")]
                    let trace_target_display_time_ns =
                        crate::frame_trace::latest_display_target_ns();
                    #[cfg(feature = "frame-trace")]
                    let trace_display_tick_sequence =
                        crate::frame_trace::latest_display_tick_sequence();
                    #[cfg(feature = "frame-trace")]
                    let trace_reused_scene = trace_logical_frame_id != 0
                        && trace_logical_frame_id == self.frame_trace_last_logical_frame_id;
                    #[cfg(feature = "frame-trace")]
                    let trace_presentation_queue_depth =
                        self.frame_trace_presentation_queue_depth.clone();
                    let presented =
                        ConcreteBlock::new(move |presented_drawable: &metal::DrawableRef| {
                            #[cfg(feature = "frame-trace")]
                            let callback_observed_ns = crate::frame_trace::monotonic_time_ns();
                            #[cfg(feature = "frame-trace")]
                            let presented_time_seconds = presented_drawable.presented_time();
                            #[cfg(feature = "frame-trace")]
                            {
                                let presented_time_ns =
                                    crate::frame_trace::host_seconds_to_ns(presented_time_seconds);
                                let previous_queue_depth = trace_presentation_queue_depth
                                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |depth| {
                                        Some(depth.saturating_sub(1))
                                    })
                                    .unwrap_or_default();
                                let kind = if presented_time_ns == 0 {
                                    crate::frame_trace::FrameTraceEventKind::DrawableDropped
                                } else {
                                    crate::frame_trace::FrameTraceEventKind::DrawablePresented
                                };
                                let mut event = crate::frame_trace::FrameTraceEvent::now(kind);
                                event.timestamp_ns = if presented_time_ns == 0 {
                                    callback_observed_ns
                                } else {
                                    presented_time_ns
                                };
                                event.callback_observed_ns = callback_observed_ns;
                                event.input_sequence_id = trace_input_sequence_id;
                                event.logical_frame_id = trace_logical_frame_id;
                                event.renderer_instance_id = trace_renderer_instance_id;
                                event.renderer_frame_id = trace_renderer_frame_id;
                                event.drawable_id = crate::frame_trace::encode_drawable_id(
                                    presented_drawable.drawable_id(),
                                );
                                event.command_buffer_id = trace_command_buffer_id;
                                event.target_display_time_ns = trace_target_display_time_ns;
                                event.display_tick_sequence = trace_display_tick_sequence;
                                event.presentation_queue_depth =
                                    previous_queue_depth.saturating_sub(1);
                                if presented_time_ns == 0 {
                                    event.flags |=
                                        crate::frame_trace::FLAG_PRESENTATION_TIMESTAMP_INVALID;
                                }
                                if trace_target_display_time_ns == 0 {
                                    event.flags |= crate::frame_trace::FLAG_DISPLAY_TARGET_INVALID;
                                }
                                if trace_target_display_time_ns != 0
                                    && presented_time_ns
                                        > trace_target_display_time_ns.saturating_add(1_000_000)
                                {
                                    event.flags |=
                                        crate::frame_trace::FLAG_PRESENTED_AFTER_DISPLAY_TARGET;
                                }
                                if previous_queue_depth == 0 {
                                    event.flags |= crate::frame_trace::FLAG_QUEUE_DEPTH_UNDERFLOW;
                                }
                                if event.drawable_id != trace_drawable_id {
                                    event.flags |= crate::frame_trace::FLAG_DRAWABLE_ID_MISMATCH;
                                }
                                if trace_reused_scene {
                                    event.flags |= crate::frame_trace::FLAG_REUSED_SCENE;
                                }
                                crate::frame_trace::record(event);
                            }
                            let gpu_start: f64 = unsafe {
                                msg_send![measured_command_buffer.as_ref(), GPUStartTime]
                            };
                            let gpu_end: f64 =
                                unsafe { msg_send![measured_command_buffer.as_ref(), GPUEndTime] };
                            let gpu_duration = (gpu_start > 0. && gpu_end >= gpu_start)
                                .then(|| Duration::from_secs_f64(gpu_end - gpu_start));
                            #[cfg(not(feature = "frame-trace"))]
                            let presented_time_seconds = presented_drawable.presented_time();
                            let observed_at = Instant::now();
                            let current_host_time = current_media_time();
                            let callback_delay = (current_host_time.is_finite()
                                && presented_time_seconds.is_finite()
                                && current_host_time >= presented_time_seconds)
                                .then(|| {
                                    Duration::from_secs_f64(
                                        current_host_time - presented_time_seconds,
                                    )
                                })
                                .unwrap_or_default();
                            let presented_at = observed_at
                                .checked_sub(callback_delay)
                                .unwrap_or(submitted_at)
                                .max(submitted_at);
                            samples.lock().push(FramePresentationSample {
                                frame_id,
                                drawable_id: presented_drawable.drawable_id() as u64,
                                presented_time_seconds,
                                submitted_at,
                                presented_at,
                                observed_at,
                                gpu_duration,
                            });
                        });
                    let presented = presented.copy();
                    drawable.add_presented_handler(&presented);

                    #[cfg(feature = "frame-trace")]
                    let prepare_submission_event = || {
                        let queue_depth = self
                            .frame_trace_presentation_queue_depth
                            .fetch_add(1, Ordering::AcqRel)
                            + 1;
                        let mut event = self.frame_trace_event(
                            crate::frame_trace::FrameTraceEventKind::CommandBufferSubmitted,
                            scene,
                            trace_drawable_id,
                            trace_command_buffer_id,
                        );
                        event.presentation_queue_depth = queue_depth;
                        event.timestamp_ns = crate::frame_trace::monotonic_time_ns();
                        if event.target_display_time_ns != 0
                            && event.timestamp_ns > event.target_display_time_ns
                        {
                            event.flags |= crate::frame_trace::FLAG_MISSED_DISPLAY_TARGET;
                        }
                        event
                    };

                    if self.presents_with_transaction {
                        #[cfg(feature = "frame-trace")]
                        let submitted_event = prepare_submission_event();
                        command_buffer.commit();
                        #[cfg(feature = "frame-trace")]
                        {
                            crate::frame_trace::record(submitted_event);
                            if frame_trace_detailed {
                                crate::frame_trace::record(self.frame_trace_event(
                                    crate::frame_trace::FrameTraceEventKind::CommandBufferCommitReturned,
                                    scene,
                                    trace_drawable_id,
                                    trace_command_buffer_id,
                                ));
                            }
                        }
                        command_buffer.wait_until_scheduled();
                        drawable.present();
                    } else {
                        let display_link_target_seconds = (self.presentation_mode
                            == MetalPresentationMode::DisplayLinkTargetEarly)
                            .then_some(display_link_target_host_time)
                            .flatten()
                            .and_then(|target_host_time| {
                                // SAFETY: mach_absolute_time has no preconditions.
                                let now_host_time =
                                    unsafe { mach2::mach_time::mach_absolute_time() };
                                valid_display_link_target_seconds(target_host_time, now_host_time)
                            });
                        if let Some(target_seconds) = display_link_target_seconds {
                            unsafe {
                                let _: () = msg_send![
                                    command_buffer.as_ref(),
                                    presentDrawable: drawable
                                    atTime: target_seconds
                                ];
                            }
                        } else {
                            command_buffer.present_drawable(drawable);
                        }
                        #[cfg(feature = "frame-trace")]
                        let submitted_event = prepare_submission_event();
                        command_buffer.commit();
                        #[cfg(feature = "frame-trace")]
                        {
                            crate::frame_trace::record(submitted_event);
                            if frame_trace_detailed {
                                crate::frame_trace::record(self.frame_trace_event(
                                    crate::frame_trace::FrameTraceEventKind::CommandBufferCommitReturned,
                                    scene,
                                    trace_drawable_id,
                                    trace_command_buffer_id,
                                ));
                            }
                        }
                    }
                    #[cfg(feature = "frame-trace")]
                    {
                        self.frame_trace_last_logical_frame_id = trace_logical_frame_id;
                    }
                    let mut deferred = mem::take(&mut self.raster_compositor_deferred_renders);
                    deferred.extend(deferred_renders);
                    self.commit_deferred_renders(deferred);
                    return;
                }
                Err(err) => {
                    log::error!(
                        "failed to render: {}. retrying with larger instance buffer size",
                        err
                    );
                    let mut instance_buffer_pool = self.instance_buffer_pool.lock();
                    let buffer_size = instance_buffer_pool.buffer_size;
                    if buffer_size >= 256 * 1024 * 1024 {
                        log::error!("instance buffer size grew too large: {}", buffer_size);
                        break;
                    }
                    instance_buffer_pool.reset(buffer_size * 2);
                    log::info!(
                        "increased instance buffer size to {}",
                        instance_buffer_pool.buffer_size
                    );
                }
            }
        }
        let deferred = mem::take(&mut self.raster_compositor_deferred_renders);
        self.commit_deferred_renders(deferred);
    }

    fn draw_primitives(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        drawable: &metal::MetalDrawableRef,
        viewport: Viewport,
    ) -> Result<(metal::CommandBuffer, Vec<DeferredRender>)> {
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        #[cfg(feature = "frame-trace")]
        if crate::frame_trace::is_detailed_enabled() && scene.frame_trace_logical_frame_id != 0 {
            let drawable_id = crate::frame_trace::encode_drawable_id(drawable.drawable_id());
            let command_buffer_id = command_buffer.as_ptr() as usize as u64;
            crate::frame_trace::record(self.frame_trace_event(
                crate::frame_trace::FrameTraceEventKind::CommandBufferCreated,
                scene,
                drawable_id,
                command_buffer_id,
            ));
        }
        let mut deferred_renders = Vec::new();
        let alpha = if self.layer.is_opaque() { 1. } else { 0. };
        let mut instance_offset = 0;

        let protected = scene
            .raster_tiles
            .iter()
            .map(|tile| (tile.cache_id, tile.key, tile.revision))
            .chain(
                scene
                    .raster_tile_update_batches
                    .iter()
                    .flat_map(|batch| batch.scene.raster_tiles.iter())
                    .map(|tile| (tile.cache_id, tile.key, tile.revision)),
            )
            .chain(scene.raster_tile_updates.iter().map(|update| {
                (
                    update.cache.id(),
                    update.key.value(),
                    update.revision.value(),
                )
            }))
            .chain(scene.raster_tile_update_batches.iter().flat_map(|batch| {
                batch.targets.iter().map(|target| {
                    (
                        batch.cache.id(),
                        target.key.value(),
                        target.revision.value(),
                    )
                })
            }))
            .collect::<HashSet<_>>();
        for update in &scene.raster_tile_updates {
            let Some(texture) = self.prepare_raster_texture(update, &protected) else {
                log::error!(
                    "raster tile allocation skipped: cache={} key={} revision={}",
                    update.cache.id(),
                    update.key.value(),
                    update.revision.value()
                );
                continue;
            };
            let gutter = ScaledPixels(update.gutter.0 as f32);
            let tile_viewport = Viewport {
                size: update.texture_size,
                origin: point(
                    update.source_bounds.origin.x - gutter,
                    update.source_bounds.origin.y - gutter,
                ),
            };
            self.encode_scene(
                &update.scene,
                &texture,
                tile_viewport,
                command_buffer,
                instance_buffer,
                &mut instance_offset,
                0.,
            )?;
        }
        for batch in &scene.raster_tile_update_batches {
            if batch.targets.is_empty() {
                continue;
            }
            let Some(source_bounds) = batch
                .targets
                .iter()
                .map(|target| target.source_bounds)
                .reduce(|left, right| left.union(&right))
            else {
                continue;
            };
            let gutter = batch.gutter.0.max(0);
            let covered_width = batch
                .targets
                .iter()
                .map(|target| {
                    (target.source_bounds.origin.x.0 - source_bounds.origin.x.0).round() as i32
                        + batch.texture_size.width.0
                })
                .max()
                .unwrap_or_default();
            let covered_height = batch
                .targets
                .iter()
                .map(|target| {
                    (target.source_bounds.origin.y.0 - source_bounds.origin.y.0).round() as i32
                        + batch.texture_size.height.0
                })
                .max()
                .unwrap_or_default();
            let batch_width =
                (source_bounds.size.width.0.ceil() as i32 + 2 * gutter).max(covered_width);
            let batch_height =
                (source_bounds.size.height.0.ceil() as i32 + 2 * gutter).max(covered_height);
            if batch_width <= 0 || batch_height <= 0 {
                continue;
            }

            let batch_size = size(DevicePixels(batch_width), DevicePixels(batch_height));
            let Some(prepared_batch_texture) =
                self.prepare_raster_batch_texture(batch, source_bounds, batch_size, &protected)
            else {
                log::error!(
                    "raster tile batch allocation skipped: cache={} targets={}",
                    batch.cache.id(),
                    batch.targets.len()
                );
                continue;
            };
            let PreparedRasterBatchTexture::Allocated(batch_texture) = prepared_batch_texture
            else {
                continue;
            };
            let batch_viewport = Viewport {
                size: batch_size,
                origin: point(
                    source_bounds.origin.x - ScaledPixels(gutter as f32),
                    source_bounds.origin.y - ScaledPixels(gutter as f32),
                ),
            };
            if batch.deferred {
                let deferred_command_buffer = command_queue.new_command_buffer();
                let mut deferred_instance_buffer =
                    self.instance_buffer_pool.lock().acquire(&self.device);
                let mut deferred_instance_offset = 0;
                self.encode_scene(
                    &batch.scene,
                    &batch_texture,
                    batch_viewport,
                    deferred_command_buffer,
                    &mut deferred_instance_buffer,
                    &mut deferred_instance_offset,
                    0.,
                )?;
                if batch.verify {
                    let comparisons = self
                        .raster_namespaces
                        .get(&batch.cache.id())
                        .expect("raster batch allocation creates its namespace")
                        .comparisons
                        .clone();
                    self.encode_raster_comparison(
                        batch,
                        &batch_texture,
                        batch_size,
                        batch_viewport,
                        deferred_command_buffer,
                        &mut deferred_instance_buffer,
                        &mut deferred_instance_offset,
                        comparisons,
                    )?;
                }
                deferred_instance_buffer
                    .metal_buffer
                    .did_modify_range(NSRange {
                        location: 0,
                        length: deferred_instance_offset as NSUInteger,
                    });
                deferred_renders.push(DeferredRender {
                    command_buffer: deferred_command_buffer.to_owned(),
                    instance_buffer: deferred_instance_buffer,
                });
            } else {
                self.encode_scene(
                    &batch.scene,
                    &batch_texture,
                    batch_viewport,
                    command_buffer,
                    instance_buffer,
                    &mut instance_offset,
                    0.,
                )?;
            }
        }
        let cache_limits = scene
            .raster_tile_updates
            .iter()
            .map(|update| (update.cache.id(), update.config.soft_limit_bytes()))
            .chain(
                scene
                    .raster_tile_update_batches
                    .iter()
                    .map(|batch| (batch.cache.id(), batch.config.soft_limit_bytes())),
            )
            .collect::<HashMap<_, _>>();
        for (cache_id, soft_limit) in cache_limits {
            while self
                .raster_namespaces
                .get(&cache_id)
                .is_some_and(|namespace| namespace.stats.resident_bytes > soft_limit)
            {
                let candidate = self
                    .raster_textures
                    .iter()
                    .filter(
                        |((candidate_cache, candidate_key, candidate_revision), _)| {
                            *candidate_cache == cache_id
                                && !protected.contains(&(
                                    *candidate_cache,
                                    *candidate_key,
                                    *candidate_revision,
                                ))
                        },
                    )
                    .min_by_key(|(_, texture)| texture.last_used_frame)
                    .map(|(key, _)| *key);
                let Some(candidate) = candidate else {
                    break;
                };
                self.remove_raster_texture(candidate, true);
            }
        }

        self.encode_scene(
            scene,
            drawable.texture(),
            viewport,
            command_buffer,
            instance_buffer,
            &mut instance_offset,
            alpha,
        )?;

        instance_buffer.metal_buffer.did_modify_range(NSRange {
            location: 0,
            length: instance_offset as NSUInteger,
        });
        Ok((command_buffer.to_owned(), deferred_renders))
    }

    fn encode_scene(
        &mut self,
        scene: &Scene,
        target: &metal::TextureRef,
        viewport: Viewport,
        command_buffer: &metal::CommandBufferRef,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        clear_alpha: f64,
    ) -> Result<()> {
        // A raster resample batch normally contains only cached tile primitives. Allocating a
        // full-size color target plus a 4× MSAA target for every such deferred command buffer
        // needlessly multiplies transient GPU memory and delays exact-zoom completion.
        let (path_intermediate, path_intermediate_msaa) = if scene_needs_path_intermediate(scene) {
            self.path_textures_for_viewport(viewport.size)
        } else {
            (None, None)
        };

        let mut command_encoder =
            new_command_encoder(command_buffer, target, viewport, |color_attachment| {
                color_attachment.set_load_action(metal::MTLLoadAction::Clear);
                color_attachment.set_clear_color(metal::MTLClearColor::new(
                    0.,
                    0.,
                    0.,
                    clear_alpha,
                ));
            });

        for batch in scene.batches() {
            let batch_name = match &batch {
                PrimitiveBatch::Shadows(_) => "shadows",
                PrimitiveBatch::Quads(_) => "quads",
                PrimitiveBatch::Paths(_) => "paths",
                PrimitiveBatch::Underlines(_) => "underlines",
                PrimitiveBatch::MonochromeSprites { .. } => "monochrome_sprites",
                PrimitiveBatch::PolychromeSprites { .. } => "polychrome_sprites",
                PrimitiveBatch::Surfaces(_) => "surfaces",
                PrimitiveBatch::RasterTiles(_) => "raster_tiles",
            };
            let ok = match batch {
                PrimitiveBatch::Shadows(shadows) => self.draw_shadows(
                    shadows,
                    instance_buffer,
                    instance_offset,
                    viewport,
                    command_encoder,
                ),
                PrimitiveBatch::Quads(quads) => self.draw_quads(
                    quads,
                    instance_buffer,
                    instance_offset,
                    viewport,
                    command_encoder,
                ),
                PrimitiveBatch::Paths(paths) => {
                    command_encoder.end_encoding();

                    let did_draw = self.draw_paths_to_intermediate(
                        paths,
                        instance_buffer,
                        instance_offset,
                        viewport,
                        command_buffer,
                        path_intermediate.as_ref(),
                        path_intermediate_msaa.as_ref(),
                    );

                    command_encoder =
                        new_command_encoder(command_buffer, target, viewport, |color_attachment| {
                            color_attachment.set_load_action(metal::MTLLoadAction::Load);
                        });

                    if did_draw {
                        self.draw_paths_from_intermediate(
                            paths,
                            instance_buffer,
                            instance_offset,
                            viewport,
                            command_encoder,
                            path_intermediate.as_ref(),
                        )
                    } else {
                        false
                    }
                }
                PrimitiveBatch::Underlines(underlines) => self.draw_underlines(
                    underlines,
                    instance_buffer,
                    instance_offset,
                    viewport,
                    command_encoder,
                ),
                PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    sprites,
                } => self.draw_monochrome_sprites(
                    texture_id,
                    sprites,
                    instance_buffer,
                    instance_offset,
                    viewport,
                    command_encoder,
                ),
                PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    sprites,
                } => self.draw_polychrome_sprites(
                    texture_id,
                    sprites,
                    instance_buffer,
                    instance_offset,
                    viewport,
                    command_encoder,
                ),
                PrimitiveBatch::Surfaces(surfaces) => self.draw_surfaces(
                    surfaces,
                    instance_buffer,
                    instance_offset,
                    viewport,
                    command_encoder,
                ),
                PrimitiveBatch::RasterTiles(tiles) => self.draw_raster_tiles(
                    tiles,
                    instance_buffer,
                    instance_offset,
                    viewport,
                    command_encoder,
                ),
            };
            if !ok {
                command_encoder.end_encoding();
                anyhow::bail!(
                    "scene too large at {batch_name}: offset={} capacity={}; {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                    *instance_offset,
                    instance_buffer.size,
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.polychrome_sprites.len(),
                    scene.surfaces.len(),
                );
            }
        }

        command_encoder.end_encoding();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_raster_comparison(
        &mut self,
        batch: &crate::RasterTileUpdateBatch,
        reference_texture: &metal::TextureRef,
        texture_size: Size<DevicePixels>,
        viewport: Viewport,
        command_buffer: &metal::CommandBufferRef,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        comparisons: Arc<Mutex<RasterComparisonStats>>,
    ) -> Result<()> {
        let width = texture_size.width.0.max(0) as usize;
        let height = texture_size.height.0.max(0) as usize;
        if width == 0 || height == 0 {
            return Ok(());
        }

        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(width as u64);
        descriptor.set_height(height as u64);
        descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        let candidate_texture = self.device.new_texture(&descriptor);

        let mut candidate_scene = Scene::default();
        for target in &batch.targets {
            candidate_scene.insert_primitive(RasterTile {
                order: 0,
                bounds: target.source_bounds,
                content_mask: ContentMask {
                    bounds: target.source_bounds,
                },
                cache_id: batch.cache.id(),
                key: target.key.value(),
                revision: target.revision.value(),
                gutter: batch.gutter.0.max(0) as u32,
            });
        }
        candidate_scene.finish();
        self.encode_scene(
            &candidate_scene,
            &candidate_texture,
            viewport,
            command_buffer,
            instance_buffer,
            instance_offset,
            0.,
        )?;

        let row_bytes = width.saturating_mul(4).saturating_add(255) & !255;
        let buffer_len = row_bytes.saturating_mul(height);
        let reference_buffer = self
            .device
            .new_buffer(buffer_len as u64, MTLResourceOptions::StorageModeShared);
        let candidate_buffer = self
            .device
            .new_buffer(buffer_len as u64, MTLResourceOptions::StorageModeShared);
        let blit = command_buffer.new_blit_command_encoder();
        let source_size = MTLSize::new(width as NSUInteger, height as NSUInteger, 1);
        for (texture, buffer) in [
            (reference_texture, reference_buffer.as_ref()),
            (candidate_texture.as_ref(), candidate_buffer.as_ref()),
        ] {
            blit.copy_from_texture_to_buffer(
                texture,
                0,
                0,
                MTLOrigin::default(),
                source_size,
                buffer,
                0,
                row_bytes as NSUInteger,
                buffer_len as NSUInteger,
                MTLBlitOption::None,
            );
        }
        blit.end_encoding();

        let comparison_regions = batch
            .targets
            .iter()
            .map(|target| RasterComparisonRegion {
                x: (target.source_bounds.origin.x.0 - viewport.origin.x.0)
                    .round()
                    .max(0.) as usize,
                y: (target.source_bounds.origin.y.0 - viewport.origin.y.0)
                    .round()
                    .max(0.) as usize,
                width: target.source_bounds.size.width.0.round().max(0.) as usize,
                height: target.source_bounds.size.height.0.round().max(0.) as usize,
            })
            .collect::<Vec<_>>();
        let completed = ConcreteBlock::new(move |_| {
            let reference = unsafe {
                std::slice::from_raw_parts(reference_buffer.contents() as *const u8, buffer_len)
            };
            let candidate = unsafe {
                std::slice::from_raw_parts(candidate_buffer.contents() as *const u8, buffer_len)
            };
            let sample = compare_bgra_images(
                reference,
                candidate,
                width,
                height,
                row_bytes,
                &comparison_regions,
            );
            let mut accumulated = comparisons.lock();
            accumulated.samples = accumulated.samples.saturating_add(1);
            accumulated.min_ssim_ppb = if accumulated.samples == 1 {
                sample.ssim_ppb
            } else {
                accumulated.min_ssim_ppb.min(sample.ssim_ppb)
            };
            accumulated.p99_channel_error =
                accumulated.p99_channel_error.max(sample.p99_channel_error);
            accumulated.max_channel_error =
                accumulated.max_channel_error.max(sample.max_channel_error);
        });
        let completed = completed.copy();
        command_buffer.add_completed_handler(&completed);
        Ok(())
    }

    fn path_textures_for_viewport(
        &self,
        size: Size<DevicePixels>,
    ) -> (Option<metal::Texture>, Option<metal::Texture>) {
        if self
            .path_intermediate_texture
            .as_ref()
            .is_some_and(|texture| {
                texture.width() == size.width.0.max(0) as u64
                    && texture.height() == size.height.0.max(0) as u64
            })
        {
            return (
                self.path_intermediate_texture.clone(),
                self.path_intermediate_msaa_texture.clone(),
            );
        }
        self.allocate_path_intermediate_textures(size)
    }

    fn allocate_path_intermediate_textures(
        &self,
        size: Size<DevicePixels>,
    ) -> (Option<metal::Texture>, Option<metal::Texture>) {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            return (None, None);
        }

        let resolve_descriptor = metal::TextureDescriptor::new();
        resolve_descriptor.set_width(size.width.0 as u64);
        resolve_descriptor.set_height(size.height.0 as u64);
        resolve_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        resolve_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        resolve_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        let intermediate = self.device.new_texture(&resolve_descriptor);

        let msaa = if self.path_sample_count > 1 {
            let descriptor = metal::TextureDescriptor::new();
            descriptor.set_width(size.width.0 as u64);
            descriptor.set_height(size.height.0 as u64);
            descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
            descriptor.set_sample_count(self.path_sample_count as u64);
            descriptor.set_usage(metal::MTLTextureUsage::RenderTarget);
            descriptor.set_storage_mode(
                if self.device.supports_family(metal::MTLGPUFamily::Apple1) {
                    metal::MTLStorageMode::Memoryless
                } else {
                    metal::MTLStorageMode::Private
                },
            );
            Some(self.device.new_texture(&descriptor))
        } else {
            None
        };
        (Some(intermediate), msaa)
    }

    fn prepare_raster_texture(
        &mut self,
        update: &crate::RasterTileUpdate,
        protected: &HashSet<(u64, u64, u64)>,
    ) -> Option<metal::Texture> {
        self.prepare_raster_texture_fields(
            &update.cache,
            update.config,
            update.key,
            update.revision,
            update.texture_size,
            update.gutter,
            protected,
        )
    }

    fn prepare_raster_texture_fields(
        &mut self,
        cache: &RasterCacheHandle,
        config: crate::RasterCacheConfig,
        tile_key: RasterTileKey,
        tile_revision: RasterTileRevision,
        texture_size: Size<DevicePixels>,
        gutter: DevicePixels,
        protected: &HashSet<(u64, u64, u64)>,
    ) -> Option<metal::Texture> {
        let cache_id = cache.id();
        let key = tile_key.value();
        let revision = tile_revision.value();
        let texture_key = cached_raster_texture_key(cache_id, key, revision);
        let width = texture_size.width.0.max(0) as usize;
        let height = texture_size.height.0.max(0) as usize;
        let bytes = width.checked_mul(height)?.checked_mul(4)?;
        if width == 0 || height == 0 || bytes > config.hard_limit_bytes() {
            return None;
        }

        if let Some(existing) = self.raster_textures.get_mut(&texture_key)
            && existing.source_size.width.0.max(0) as usize == width
            && existing.source_size.height.0.max(0) as usize == height
        {
            existing.last_used_frame = self.frame_index;
            return Some(existing.allocation.texture.clone());
        }

        self.remove_raster_texture(texture_key, false);
        while self
            .raster_namespaces
            .get(&cache_id)
            .map(|namespace| namespace.stats.resident_bytes + bytes)
            .unwrap_or(bytes)
            > config.hard_limit_bytes()
        {
            let candidate = self
                .raster_textures
                .iter()
                .filter(
                    |((candidate_cache, candidate_key, candidate_revision), _)| {
                        *candidate_cache == cache_id
                            && !protected.contains(&(
                                *candidate_cache,
                                *candidate_key,
                                *candidate_revision,
                            ))
                    },
                )
                .min_by_key(|(_, texture)| texture.last_used_frame)
                .map(|(key, _)| *key);
            let Some(candidate) = candidate else {
                return None;
            };
            self.remove_raster_texture(candidate, true);
        }

        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(width as u64);
        descriptor.set_height(height as u64);
        descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        let texture = self.device.new_texture(&descriptor);
        let allocation = Arc::new(CachedRasterAllocation {
            texture: texture.clone(),
            bytes,
        });
        self.raster_textures.insert(
            texture_key,
            CachedRasterTexture {
                allocation,
                source_origin: point(DevicePixels(0), DevicePixels(0)),
                source_size: texture_size,
                gutter: gutter.0.max(0) as u32,
                last_used_frame: self.frame_index,
            },
        );
        let namespace = self
            .raster_namespaces
            .entry(cache_id)
            .or_insert_with(|| RasterNamespace::new(cache));
        namespace.stats.resident_bytes += bytes;
        namespace.stats.resident_tiles += 1;
        Some(texture)
    }

    fn prepare_raster_batch_texture(
        &mut self,
        batch: &crate::RasterTileUpdateBatch,
        source_bounds: Bounds<ScaledPixels>,
        texture_size: Size<DevicePixels>,
        protected: &HashSet<(u64, u64, u64)>,
    ) -> Option<PreparedRasterBatchTexture> {
        let cache_id = batch.cache.id();
        let width = texture_size.width.0.max(0) as usize;
        let height = texture_size.height.0.max(0) as usize;
        let bytes = width.checked_mul(height)?.checked_mul(4)?;
        if width == 0 || height == 0 || bytes > batch.config.hard_limit_bytes() {
            return None;
        }

        let mut entries = Vec::with_capacity(batch.targets.len());
        for target in &batch.targets {
            let source_x =
                (target.source_bounds.origin.x.0 - source_bounds.origin.x.0).round() as i32;
            let source_y =
                (target.source_bounds.origin.y.0 - source_bounds.origin.y.0).round() as i32;
            if source_x < 0
                || source_y < 0
                || source_x.saturating_add(batch.texture_size.width.0) > texture_size.width.0
                || source_y.saturating_add(batch.texture_size.height.0) > texture_size.height.0
            {
                return None;
            }
            let texture_key =
                cached_raster_texture_key(cache_id, target.key.value(), target.revision.value());
            if let Some(existing) = self.raster_textures.get_mut(&texture_key)
                && existing.source_size == batch.texture_size
                && existing.gutter == batch.gutter.0.max(0) as u32
            {
                existing.last_used_frame = self.frame_index;
                continue;
            }
            self.remove_raster_texture(texture_key, false);
            entries.push((
                target,
                point(DevicePixels(source_x), DevicePixels(source_y)),
            ));
        }
        if entries.is_empty() {
            return Some(PreparedRasterBatchTexture::AlreadyReady);
        }
        while self
            .raster_namespaces
            .get(&cache_id)
            .map(|namespace| namespace.stats.resident_bytes + bytes)
            .unwrap_or(bytes)
            > batch.config.hard_limit_bytes()
        {
            let candidate = self
                .raster_textures
                .iter()
                .filter(
                    |((candidate_cache, candidate_key, candidate_revision), _)| {
                        *candidate_cache == cache_id
                            && !protected.contains(&(
                                *candidate_cache,
                                *candidate_key,
                                *candidate_revision,
                            ))
                    },
                )
                .min_by_key(|(_, texture)| texture.last_used_frame)
                .map(|(key, _)| *key);
            let Some(candidate) = candidate else {
                return None;
            };
            self.remove_raster_texture(candidate, true);
        }

        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(width as u64);
        descriptor.set_height(height as u64);
        descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        let texture = self.device.new_texture(&descriptor);
        let allocation = Arc::new(CachedRasterAllocation {
            texture: texture.clone(),
            bytes,
        });
        let resident_tiles = entries.len();
        for (target, source_origin) in entries {
            self.raster_textures.insert(
                cached_raster_texture_key(cache_id, target.key.value(), target.revision.value()),
                CachedRasterTexture {
                    allocation: allocation.clone(),
                    source_origin,
                    source_size: batch.texture_size,
                    gutter: batch.gutter.0.max(0) as u32,
                    last_used_frame: self.frame_index,
                },
            );
        }
        let namespace = self
            .raster_namespaces
            .entry(cache_id)
            .or_insert_with(|| RasterNamespace::new(&batch.cache));
        namespace.stats.resident_bytes += bytes;
        namespace.stats.resident_tiles += resident_tiles;
        Some(PreparedRasterBatchTexture::Allocated(texture))
    }

    fn remove_raster_texture(&mut self, key: CachedRasterTextureKey, evicted: bool) {
        let Some(texture) = self.raster_textures.remove(&key) else {
            return;
        };
        let allocation_released = Arc::strong_count(&texture.allocation) == 1;
        if let Some(namespace) = self.raster_namespaces.get_mut(&key.0) {
            if allocation_released {
                namespace.stats.resident_bytes = namespace
                    .stats
                    .resident_bytes
                    .saturating_sub(texture.allocation.bytes);
            }
            namespace.stats.resident_tiles = namespace.stats.resident_tiles.saturating_sub(1);
            if evicted {
                namespace.stats.evicted_tiles += 1;
            }
        }
    }

    fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport: Viewport,
        command_buffer: &metal::CommandBufferRef,
        intermediate_texture: Option<&metal::Texture>,
        intermediate_msaa_texture: Option<&metal::Texture>,
    ) -> bool {
        if paths.is_empty() {
            return true;
        }
        let Some(intermediate_texture) = intermediate_texture else {
            return false;
        };

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));

        if let Some(msaa_texture) = intermediate_msaa_texture {
            color_attachment.set_texture(Some(msaa_texture));
            color_attachment.set_resolve_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
        }

        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        command_encoder.set_render_pipeline_state(&self.paths_rasterization_pipeline_state);

        for path in paths {
            align_offset(instance_offset);
            let vertices = path
                .vertices
                .iter()
                .map(|vertex| PathRasterizationVertex {
                    xy_position: vertex.xy_position,
                    st_position: vertex.st_position,
                })
                .collect::<Vec<_>>();
            let vertices_bytes_len = mem::size_of_val(vertices.as_slice());
            let next_offset = *instance_offset + vertices_bytes_len;
            if next_offset > instance_buffer.size {
                command_encoder.end_encoding();
                return false;
            }
            let style = PathRasterizationStyle {
                color: path.color,
                bounds: path.bounds.intersect(&path.content_mask.bounds),
            };
            command_encoder.set_vertex_buffer(
                PathRasterizationInputIndex::Vertices as u64,
                Some(&instance_buffer.metal_buffer),
                *instance_offset as u64,
            );
            command_encoder.set_vertex_bytes(
                PathRasterizationInputIndex::ViewportSize as u64,
                mem::size_of_val(&viewport) as u64,
                &viewport as *const Viewport as *const _,
            );
            command_encoder.set_vertex_bytes(
                PathRasterizationInputIndex::Style as u64,
                mem::size_of_val(&style) as u64,
                &style as *const PathRasterizationStyle as *const _,
            );
            command_encoder.set_fragment_bytes(
                PathRasterizationInputIndex::Style as u64,
                mem::size_of_val(&style) as u64,
                &style as *const PathRasterizationStyle as *const _,
            );
            let buffer_contents = unsafe {
                (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset)
            };
            unsafe {
                ptr::copy_nonoverlapping(
                    vertices.as_ptr() as *const u8,
                    buffer_contents,
                    vertices_bytes_len,
                );
            }
            command_encoder.draw_primitives(
                metal::MTLPrimitiveType::Triangle,
                0,
                vertices.len() as u64,
            );
            *instance_offset = next_offset;
        }

        command_encoder.end_encoding();
        true
    }

    fn draw_shadows(
        &self,
        shadows: &[Shadow],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport: Viewport,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if shadows.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.shadows_pipeline_state);
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        command_encoder.set_vertex_bytes(
            ShadowInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport) as u64,
            &viewport as *const Viewport as *const _,
        );

        let shadow_bytes_len = mem::size_of_val(shadows);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + shadow_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(
                shadows.as_ptr() as *const u8,
                buffer_contents,
                shadow_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            shadows.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_quads(
        &self,
        quads: &[Quad],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport: Viewport,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if quads.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.quads_pipeline_state);
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        command_encoder.set_vertex_bytes(
            QuadInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport) as u64,
            &viewport as *const Viewport as *const _,
        );

        let quad_bytes_len = mem::size_of_val(quads);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + quad_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(quads.as_ptr() as *const u8, buffer_contents, quad_bytes_len);
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            quads.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport: Viewport,
        command_encoder: &metal::RenderCommandEncoderRef,
        intermediate_texture: Option<&metal::Texture>,
    ) -> bool {
        let Some(first_path) = paths.first() else {
            return true;
        };

        let Some(intermediate_texture) = intermediate_texture else {
            return false;
        };

        command_encoder.set_render_pipeline_state(&self.path_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport) as u64,
            &viewport as *const Viewport as *const _,
        );

        command_encoder.set_fragment_texture(
            SpriteInputIndex::AtlasTexture as u64,
            Some(intermediate_texture),
        );

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites;
        if paths.last().unwrap().order == first_path.order {
            sprites = paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                })
                .collect();
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            sprites = vec![PathSprite { bounds }];
        }

        align_offset(instance_offset);
        let sprite_bytes_len = mem::size_of_val(sprites.as_slice());
        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };
        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;

        true
    }

    fn draw_underlines(
        &self,
        underlines: &[Underline],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport: Viewport,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if underlines.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.underlines_pipeline_state);
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        command_encoder.set_vertex_bytes(
            UnderlineInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport) as u64,
            &viewport as *const Viewport as *const _,
        );

        let underline_bytes_len = mem::size_of_val(underlines);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + underline_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(
                underlines.as_ptr() as *const u8,
                buffer_contents,
                underline_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            underlines.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: &[MonochromeSprite],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport: Viewport,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let sprite_bytes_len = mem::size_of_val(sprites);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.monochrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport) as u64,
            &viewport as *const Viewport as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: &[PolychromeSprite],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport: Viewport,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.polychrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport) as u64,
            &viewport as *const Viewport as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        let sprite_bytes_len = mem::size_of_val(sprites);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport: Viewport,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        command_encoder.set_render_pipeline_state(&self.surfaces_pipeline_state);
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SurfaceInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport) as u64,
            &viewport as *const Viewport as *const _,
        );

        for surface in surfaces {
            let texture_size = size(
                DevicePixels::from(surface.image_buffer.get_width() as i32),
                DevicePixels::from(surface.image_buffer.get_height() as i32),
            );

            assert_eq!(
                surface.image_buffer.get_pixel_format(),
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            );

            let y_texture = self
                .core_video_texture_cache
                .create_texture_from_image(
                    surface.image_buffer.as_concrete_TypeRef(),
                    None,
                    MTLPixelFormat::R8Unorm,
                    surface.image_buffer.get_width_of_plane(0),
                    surface.image_buffer.get_height_of_plane(0),
                    0,
                )
                .unwrap();
            let cb_cr_texture = self
                .core_video_texture_cache
                .create_texture_from_image(
                    surface.image_buffer.as_concrete_TypeRef(),
                    None,
                    MTLPixelFormat::RG8Unorm,
                    surface.image_buffer.get_width_of_plane(1),
                    surface.image_buffer.get_height_of_plane(1),
                    1,
                )
                .unwrap();

            align_offset(instance_offset);
            let next_offset = *instance_offset + mem::size_of::<Surface>();
            if next_offset > instance_buffer.size {
                return false;
            }

            command_encoder.set_vertex_buffer(
                SurfaceInputIndex::Surfaces as u64,
                Some(&instance_buffer.metal_buffer),
                *instance_offset as u64,
            );
            command_encoder.set_vertex_bytes(
                SurfaceInputIndex::TextureSize as u64,
                mem::size_of_val(&texture_size) as u64,
                &texture_size as *const Size<DevicePixels> as *const _,
            );
            // let y_texture = y_texture.get_texture().unwrap().
            command_encoder.set_fragment_texture(SurfaceInputIndex::YTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(y_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });
            command_encoder.set_fragment_texture(SurfaceInputIndex::CbCrTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(cb_cr_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });

            unsafe {
                let buffer_contents = (instance_buffer.metal_buffer.contents() as *mut u8)
                    .add(*instance_offset)
                    as *mut SurfaceBounds;
                ptr::write(
                    buffer_contents,
                    SurfaceBounds {
                        bounds: surface.bounds,
                        content_mask: surface.content_mask.clone(),
                    },
                );
            }

            command_encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
            *instance_offset = next_offset;
        }
        true
    }

    fn draw_raster_tiles(
        &self,
        tiles: &[RasterTile],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport: Viewport,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        command_encoder.set_render_pipeline_state(&self.raster_tiles_pipeline_state);
        command_encoder.set_vertex_buffer(
            RasterTileInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            RasterTileInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport) as u64,
            &viewport as *const Viewport as *const _,
        );

        for tile in tiles {
            let Some(texture) = self.raster_textures.get(&cached_raster_texture_key(
                tile.cache_id,
                tile.key,
                tile.revision,
            )) else {
                continue;
            };
            align_offset(instance_offset);
            let next_offset = *instance_offset + mem::size_of::<RasterTileBounds>();
            if next_offset > instance_buffer.size {
                return false;
            }
            unsafe {
                let destination = (instance_buffer.metal_buffer.contents() as *mut u8)
                    .add(*instance_offset)
                    as *mut RasterTileBounds;
                ptr::write(
                    destination,
                    RasterTileBounds {
                        bounds: tile.bounds,
                        content_mask: tile.content_mask.clone(),
                        texture_size: size(
                            DevicePixels(texture.allocation.texture.width() as i32),
                            DevicePixels(texture.allocation.texture.height() as i32),
                        ),
                        source_origin: texture.source_origin,
                        source_size: texture.source_size,
                        gutter: DevicePixels(tile.gutter as i32),
                        _padding: DevicePixels(0),
                    },
                );
            }
            command_encoder.set_vertex_buffer(
                RasterTileInputIndex::Tiles as u64,
                Some(&instance_buffer.metal_buffer),
                *instance_offset as u64,
            );
            command_encoder.set_fragment_texture(
                RasterTileInputIndex::Texture as u64,
                Some(&texture.allocation.texture),
            );
            command_encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
            *instance_offset = next_offset;
        }
        true
    }
}

#[cfg(test)]
mod presentation_mode_tests {
    use super::{MetalPresentationMode, parse_metal_presentation_mode, valid_target_seconds};

    #[test]
    fn presentation_mode_defaults_to_asap() {
        assert_eq!(
            parse_metal_presentation_mode(None),
            MetalPresentationMode::Asap
        );
    }

    #[test]
    fn presentation_mode_accepts_closed_values() {
        assert_eq!(
            parse_metal_presentation_mode(Some("asap")),
            MetalPresentationMode::Asap
        );
        assert_eq!(
            parse_metal_presentation_mode(Some("display-link-target-early")),
            MetalPresentationMode::DisplayLinkTargetEarly
        );
        assert_eq!(
            parse_metal_presentation_mode(Some("unexpected")),
            MetalPresentationMode::Asap
        );
    }

    #[test]
    fn display_link_target_must_be_near_and_in_the_future() {
        assert_eq!(valid_target_seconds(10.016, 10.), Some(10.015));
        assert_eq!(valid_target_seconds(10.001, 10.), None);
        assert_eq!(valid_target_seconds(10., 10.), None);
        assert_eq!(valid_target_seconds(9.999, 10.), None);
        assert_eq!(valid_target_seconds(10.051, 10.), None);
    }
}

fn new_command_encoder<'a>(
    command_buffer: &'a metal::CommandBufferRef,
    target: &'a metal::TextureRef,
    viewport: Viewport,
    configure_color_attachment: impl Fn(&RenderPassColorAttachmentDescriptorRef),
) -> &'a metal::RenderCommandEncoderRef {
    let render_pass_descriptor = metal::RenderPassDescriptor::new();
    let color_attachment = render_pass_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();
    color_attachment.set_texture(Some(target));
    color_attachment.set_store_action(metal::MTLStoreAction::Store);
    configure_color_attachment(color_attachment);

    let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
    command_encoder.set_viewport(metal::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport.size.width) as f64,
        height: i32::from(viewport.size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    command_encoder
}

fn build_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_sprite_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_rasterization_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    path_sample_count: u32,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.set_raster_sample_count(path_sample_count as _);
        descriptor.set_alpha_to_coverage_enabled(false);
    }
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RasterComparisonSample {
    ssim_ppb: u32,
    p99_channel_error: u8,
    max_channel_error: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RasterComparisonRegion {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

fn compare_bgra_images(
    reference: &[u8],
    candidate: &[u8],
    width: usize,
    height: usize,
    row_bytes: usize,
    regions: &[RasterComparisonRegion],
) -> RasterComparisonSample {
    let mut count = 0_u64;
    let mut channel_count = 0_u64;
    let mut reference_sum = 0_f64;
    let mut candidate_sum = 0_f64;
    let mut reference_squared_sum = 0_f64;
    let mut candidate_squared_sum = 0_f64;
    let mut product_sum = 0_f64;
    let mut errors = [0_u64; 256];

    for region in regions {
        let start_x = region.x.min(width);
        let start_y = region.y.min(height);
        let end_x = region.x.saturating_add(region.width).min(width);
        let end_y = region.y.saturating_add(region.height).min(height);
        for y in start_y..end_y {
            let row = y.saturating_mul(row_bytes);
            for x in start_x..end_x {
                let offset = row.saturating_add(x.saturating_mul(4));
                if offset.saturating_add(4) > reference.len()
                    || offset.saturating_add(4) > candidate.len()
                {
                    continue;
                }
                let reference_pixel = &reference[offset..offset + 4];
                let candidate_pixel = &candidate[offset..offset + 4];
                for channel in 0..4 {
                    let error = reference_pixel[channel].abs_diff(candidate_pixel[channel]);
                    errors[usize::from(error)] = errors[usize::from(error)].saturating_add(1);
                    channel_count = channel_count.saturating_add(1);
                }
                let luma = |pixel: &[u8]| {
                    (77. * f64::from(pixel[2])
                        + 150. * f64::from(pixel[1])
                        + 29. * f64::from(pixel[0]))
                        / 256.
                };
                let reference_luma = luma(reference_pixel);
                let candidate_luma = luma(candidate_pixel);
                reference_sum += reference_luma;
                candidate_sum += candidate_luma;
                reference_squared_sum += reference_luma * reference_luma;
                candidate_squared_sum += candidate_luma * candidate_luma;
                product_sum += reference_luma * candidate_luma;
                count = count.saturating_add(1);
            }
        }
    }

    if count == 0 || channel_count == 0 {
        return RasterComparisonSample {
            ssim_ppb: 0,
            p99_channel_error: u8::MAX,
            max_channel_error: u8::MAX,
        };
    }
    let count_f64 = count as f64;
    let reference_mean = reference_sum / count_f64;
    let candidate_mean = candidate_sum / count_f64;
    let reference_variance =
        (reference_squared_sum / count_f64 - reference_mean * reference_mean).max(0.);
    let candidate_variance =
        (candidate_squared_sum / count_f64 - candidate_mean * candidate_mean).max(0.);
    let covariance = product_sum / count_f64 - reference_mean * candidate_mean;
    let c1 = (0.01_f64 * 255.).powi(2);
    let c2 = (0.03_f64 * 255.).powi(2);
    let ssim = (((2. * reference_mean * candidate_mean + c1) * (2. * covariance + c2))
        / ((reference_mean.powi(2) + candidate_mean.powi(2) + c1)
            * (reference_variance + candidate_variance + c2)))
        .clamp(0., 1.);
    let p99_rank = channel_count.saturating_mul(99).div_ceil(100);
    let mut cumulative = 0_u64;
    let mut p99_channel_error = None;
    let mut max_channel_error = 0_u8;
    for (error, occurrences) in errors.into_iter().enumerate() {
        if occurrences == 0 {
            continue;
        }
        max_channel_error = error as u8;
        cumulative = cumulative.saturating_add(occurrences);
        if cumulative >= p99_rank && p99_channel_error.is_none() {
            p99_channel_error = Some(error as u8);
        }
    }
    RasterComparisonSample {
        ssim_ppb: (ssim * 1_000_000_000.).round() as u32,
        p99_channel_error: p99_channel_error.unwrap_or(max_channel_error),
        max_channel_error,
    }
}

#[cfg(test)]
mod raster_comparison_tests {
    use super::*;

    #[test]
    fn raster_texture_identity_keeps_content_revisions_distinct() {
        let visible = cached_raster_texture_key(7, 11, 41);
        let replacement = cached_raster_texture_key(7, 11, 42);

        assert_ne!(visible, replacement);
        let protected = HashSet::from([visible, replacement]);
        assert!(protected.contains(&visible));
        assert!(protected.contains(&replacement));
    }

    #[test]
    fn identical_images_have_perfect_similarity() {
        let image = vec![17_u8; 4 * 4 * 4];
        let regions = [RasterComparisonRegion {
            x: 0,
            y: 0,
            width: 4,
            height: 4,
        }];
        assert_eq!(
            compare_bgra_images(&image, &image, 4, 4, 16, &regions),
            RasterComparisonSample {
                ssim_ppb: 1_000_000_000,
                p99_channel_error: 0,
                max_channel_error: 0,
            }
        );
    }

    #[test]
    fn comparison_reports_channel_error_and_honors_gutter() {
        let reference = vec![0_u8; 4 * 4 * 4];
        let mut candidate = reference.clone();
        candidate[0] = 255;
        let inner_region = [RasterComparisonRegion {
            x: 1,
            y: 1,
            width: 2,
            height: 2,
        }];
        assert_eq!(
            compare_bgra_images(&reference, &candidate, 4, 4, 16, &inner_region),
            RasterComparisonSample {
                ssim_ppb: 1_000_000_000,
                p99_channel_error: 0,
                max_channel_error: 0,
            }
        );
        let full_region = [RasterComparisonRegion {
            x: 0,
            y: 0,
            width: 4,
            height: 4,
        }];
        let sample = compare_bgra_images(&reference, &candidate, 4, 4, 16, &full_region);
        assert_eq!(sample.max_channel_error, 255);
    }

    #[test]
    fn renderer_instance_types_remain_compact() {
        assert!(mem::size_of::<MonochromeSprite>() <= 128);
        assert!(mem::size_of::<Quad>() <= 192);
        assert!(mem::size_of::<Shadow>() <= 192);
        assert_eq!(mem::size_of::<PathRasterizationVertex>(), 16);
    }

    #[test]
    fn path_intermediate_is_reserved_only_for_scenes_with_paths() {
        let mut scene = Scene::default();
        assert!(!scene_needs_path_intermediate(&scene));

        scene
            .paths
            .push(crate::Path::new(point(crate::px(0.), crate::px(0.))).scale(1.));
        assert!(scene_needs_path_intermediate(&scene));
    }

    #[test]
    fn compositor_rejects_transforms_that_expose_uncaptured_pixels() {
        let raster_size = NSSize {
            width: 1_824.,
            height: 1_024.,
        };
        let clip_size = NSSize {
            width: 800.,
            height: 600.,
        };
        assert!(raster_compositor_surface_covers_clip(
            NSPoint { x: -512., y: -212. },
            1.,
            raster_size,
            clip_size,
        ));
        assert!(!raster_compositor_surface_covers_clip(
            NSPoint { x: 1., y: -212. },
            1.,
            raster_size,
            clip_size,
        ));
        assert!(!raster_compositor_surface_covers_clip(
            NSPoint {
                x: -1_025.,
                y: -212.
            },
            1.,
            raster_size,
            clip_size,
        ));
        assert!(!raster_compositor_surface_covers_clip(
            NSPoint { x: -512., y: -212. },
            0.,
            raster_size,
            clip_size,
        ));
    }
}

fn scene_needs_path_intermediate(scene: &Scene) -> bool {
    !scene.paths.is_empty()
}

fn required_instance_buffer_size(scene: &Scene) -> usize {
    fn reserve(offset: &mut usize, bytes: usize) {
        if bytes == 0 {
            return;
        }
        align_offset(offset);
        *offset = offset.saturating_add(bytes);
    }

    fn visit(scene: &Scene, offset: &mut usize) {
        for batch in scene.batches() {
            match batch {
                PrimitiveBatch::Shadows(values) => reserve(offset, mem::size_of_val(values)),
                PrimitiveBatch::Quads(values) => reserve(offset, mem::size_of_val(values)),
                PrimitiveBatch::Paths(paths) => {
                    for path in paths {
                        reserve(
                            offset,
                            path.vertices
                                .len()
                                .saturating_mul(mem::size_of::<PathRasterizationVertex>()),
                        );
                    }
                    let sprite_count = paths.first().map_or(0, |first| {
                        if paths.last().is_some_and(|last| last.order == first.order) {
                            paths.len()
                        } else {
                            1
                        }
                    });
                    reserve(
                        offset,
                        sprite_count.saturating_mul(mem::size_of::<PathSprite>()),
                    );
                }
                PrimitiveBatch::Underlines(values) => reserve(offset, mem::size_of_val(values)),
                PrimitiveBatch::MonochromeSprites { sprites, .. } => {
                    reserve(offset, mem::size_of_val(sprites));
                }
                PrimitiveBatch::PolychromeSprites { sprites, .. } => {
                    reserve(offset, mem::size_of_val(sprites));
                }
                PrimitiveBatch::Surfaces(values) => {
                    for _ in values {
                        reserve(offset, mem::size_of::<Surface>());
                    }
                }
                PrimitiveBatch::RasterTiles(values) => {
                    for _ in values {
                        reserve(offset, mem::size_of::<RasterTileBounds>());
                    }
                }
            }
        }
    }

    let mut required = 0usize;
    for update in &scene.raster_tile_updates {
        visit(&update.scene, &mut required);
    }
    for batch in scene
        .raster_tile_update_batches
        .iter()
        .filter(|batch| !batch.deferred)
    {
        visit(&batch.scene, &mut required);
    }
    visit(scene, &mut required);
    let deferred_required = scene
        .raster_tile_update_batches
        .iter()
        .filter(|batch| batch.deferred)
        .map(|batch| {
            let mut batch_required = 0;
            visit(&batch.scene, &mut batch_required);
            if batch.verify {
                for _ in &batch.targets {
                    reserve(&mut batch_required, mem::size_of::<RasterTileBounds>());
                }
            }
            batch_required
        })
        .max()
        .unwrap_or_default();
    let compositor_required = scene
        .raster_compositor_surfaces
        .iter()
        .map(|surface| required_instance_buffer_size(&surface.scene))
        .max()
        .unwrap_or_default();
    required.max(deferred_required).max(compositor_required)
}

// Align to multiples of 256 make Metal happy.
fn align_offset(offset: &mut usize) {
    *offset = (*offset).div_ceil(256) * 256;
}

#[repr(C)]
enum ShadowInputIndex {
    Vertices = 0,
    Shadows = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum QuadInputIndex {
    Vertices = 0,
    Quads = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum UnderlineInputIndex {
    Vertices = 0,
    Underlines = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum SpriteInputIndex {
    Vertices = 0,
    Sprites = 1,
    ViewportSize = 2,
    AtlasTextureSize = 3,
    AtlasTexture = 4,
}

#[repr(C)]
enum SurfaceInputIndex {
    Vertices = 0,
    Surfaces = 1,
    ViewportSize = 2,
    TextureSize = 3,
    YTexture = 4,
    CbCrTexture = 5,
}

#[repr(C)]
enum PathRasterizationInputIndex {
    Vertices = 0,
    ViewportSize = 1,
    Style = 2,
}

#[repr(C)]
enum RasterTileInputIndex {
    Vertices = 0,
    Tiles = 1,
    ViewportSize = 2,
    Texture = 3,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PathSprite {
    pub bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SurfaceBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct RasterTileBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub texture_size: Size<DevicePixels>,
    pub source_origin: Point<DevicePixels>,
    pub source_size: Size<DevicePixels>,
    pub gutter: DevicePixels,
    pub _padding: DevicePixels,
}
