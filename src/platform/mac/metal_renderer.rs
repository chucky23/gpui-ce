use super::drawable_prefetch::{DrawablePrefetcher, DrawableTake, async_drawable_enabled};
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

use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    ffi::c_void,
    mem, ptr,
    sync::Arc,
    time::{Duration, Instant},
};

#[cfg(feature = "frame-trace")]
type FrameTraceCommandBufferHandler = RcBlock<(&'static metal::CommandBufferRef,), ()>;
#[cfg(feature = "frame-trace")]
const FRAME_TRACE_COMMAND_METADATA_SLOTS: usize = 64;
#[cfg(feature = "frame-trace")]
const FRAME_TRACE_COMMAND_METADATA_FIELDS: usize = 25;

/// Глубина очереди показа корневого слоя для трассы кадра.
///
/// Отправка увеличивает глубину, обработчик показа уменьшает её на другом потоке, а номер
/// события выдаётся только при записи. Изменение глубины и запись события выполняются под одной
/// блокировкой, поэтому порядок номеров событий совпадает с порядком изменений глубины.
#[cfg(feature = "frame-trace")]
#[derive(Default)]
struct PresentationQueueTrace {
    depth: AtomicU64,
    order: Mutex<()>,
}

#[cfg(feature = "frame-trace")]
impl PresentationQueueTrace {
    fn depth(&self) -> u64 {
        self.depth.load(Ordering::Acquire)
    }

    /// Увеличивает глубину и возвращает её с блокировкой, которую нужно держать до записи
    /// события отправки.
    fn begin_submission(&self) -> (parking_lot::MutexGuard<'_, ()>, u64) {
        let guard = self.order.lock();
        let depth = self.depth.fetch_add(1, Ordering::AcqRel) + 1;
        (guard, depth)
    }

    /// Уменьшает глубину и под той же блокировкой записывает показ: `record` получает глубину
    /// до показа.
    fn record_presentation(&self, record: impl FnOnce(u64)) {
        let _guard = self.order.lock();
        let previous = self.depth.load(Ordering::Acquire);
        self.depth
            .store(previous.saturating_sub(1), Ordering::Release);
        record(previous);
    }
}

#[cfg(feature = "frame-trace")]
struct FrameTraceCommandMetadataSlot {
    command_buffer_key: AtomicU64,
    fields: [AtomicU64; FRAME_TRACE_COMMAND_METADATA_FIELDS],
}

#[cfg(feature = "frame-trace")]
impl FrameTraceCommandMetadataSlot {
    fn new() -> Self {
        Self {
            command_buffer_key: AtomicU64::new(0),
            fields: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

#[cfg(feature = "frame-trace")]
struct FrameTraceCommandMetadataTable {
    slots: [FrameTraceCommandMetadataSlot; FRAME_TRACE_COMMAND_METADATA_SLOTS],
}

#[cfg(feature = "frame-trace")]
impl FrameTraceCommandMetadataTable {
    fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| FrameTraceCommandMetadataSlot::new()),
        }
    }

    fn register(&self, command_buffer_key: u64, event: &crate::frame_trace::FrameTraceEvent) {
        let slot =
            &self.slots[event.renderer_frame_id as usize % FRAME_TRACE_COMMAND_METADATA_SLOTS];
        slot.command_buffer_key.store(0, Ordering::Release);
        let token = event.presentation_token.unwrap_or_default();
        let fields = [
            event.input_sequence_id,
            event.logical_frame_id,
            event.gpui_window_frame_id,
            event.renderer_instance_id,
            event.renderer_frame_id,
            event.drawable_id,
            u64::from(event.presentation_token.is_some()),
            token.run_id_hash,
            token.input_sequence,
            token.snapshot_generation,
            token.canvas_render_generation,
            event.display_id,
            event.display_current_time_ns,
            event.display_refresh_period_ns,
            event.scene_build_display_tick_sequence,
            event.scene_build_target_display_time_ns,
            event.presentation_display_tick_sequence,
            event.presentation_target_display_time_ns,
            event.coalesced_display_tick_count,
            event.flags,
            event.display_current_host_time_raw,
            event.display_output_host_time_raw,
            event.scene_build_tick_flags,
            event.presentation_tick_flags,
            event.command_buffer_id,
        ];
        for (target, value) in slot.fields.iter().zip(fields) {
            target.store(value, Ordering::Relaxed);
        }
        slot.command_buffer_key
            .store(command_buffer_key, Ordering::Release);
    }

    fn populate(&self, command_buffer_key: u64, event: &mut crate::frame_trace::FrameTraceEvent) {
        let Some(slot) = self
            .slots
            .iter()
            .find(|slot| slot.command_buffer_key.load(Ordering::Acquire) == command_buffer_key)
        else {
            return;
        };
        let fields = std::array::from_fn::<_, FRAME_TRACE_COMMAND_METADATA_FIELDS, _>(|index| {
            slot.fields[index].load(Ordering::Relaxed)
        });
        if slot.command_buffer_key.load(Ordering::Acquire) != command_buffer_key {
            return;
        }
        event.input_sequence_id = fields[0];
        event.logical_frame_id = fields[1];
        event.gpui_window_frame_id = fields[2];
        event.renderer_instance_id = fields[3];
        event.renderer_frame_id = fields[4];
        event.drawable_id = fields[5];
        event.presentation_token =
            (fields[6] != 0).then_some(crate::frame_trace::PresentationToken {
                run_id_hash: fields[7],
                input_sequence: fields[8],
                snapshot_generation: fields[9],
                canvas_render_generation: fields[10],
            });
        event.display_id = fields[11];
        event.display_current_time_ns = fields[12];
        event.display_refresh_period_ns = fields[13];
        event.scene_build_display_tick_sequence = fields[14];
        event.scene_build_target_display_time_ns = fields[15];
        event.presentation_display_tick_sequence = fields[16];
        event.presentation_target_display_time_ns = fields[17];
        event.coalesced_display_tick_count = fields[18];
        event.flags = fields[19];
        event.display_current_host_time_raw = fields[20];
        event.display_output_host_time_raw = fields[21];
        event.scene_build_tick_flags = fields[22];
        event.presentation_tick_flags = fields[23];
        event.command_buffer_id = fields[24];
    }

    fn remove(&self, command_buffer_key: u64) {
        for slot in &self.slots {
            let _ = slot.command_buffer_key.compare_exchange(
                command_buffer_key,
                0,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }
}

#[cfg(feature = "frame-trace")]
fn project_frame_trace_capture_level(
    event: &mut crate::frame_trace::FrameTraceEvent,
    detailed: bool,
) {
    if !detailed {
        event.coalesced_display_tick_count = 0;
    }
}

/// Потребители `Reference` и `Detailed` ждут ноль в `next_drawable_wait_ns` отправки:
/// в Detailed ожидание несёт `DrawableAcquired`.
#[cfg(feature = "frame-trace")]
fn project_frame_trace_submission_timing(
    event: &mut crate::frame_trace::FrameTraceEvent,
    reference_timed: bool,
    next_drawable_wait_ns: u64,
) {
    if reference_timed {
        event.next_drawable_wait_ns = next_drawable_wait_ns;
    }
}

/// Интервал GPU корневого command buffer в событии показа; в Detailed его несёт
/// `GpuCompleted`, поэтому правила валидности те же.
#[cfg(feature = "frame-trace")]
fn project_frame_trace_gpu_interval(
    event: &mut crate::frame_trace::FrameTraceEvent,
    reference_timed: bool,
    gpu_start_seconds: f64,
    gpu_end_seconds: f64,
) {
    if !reference_timed {
        return;
    }
    event.gpu_start_time_ns = crate::frame_trace::host_seconds_to_ns(gpu_start_seconds);
    event.gpu_end_time_ns = crate::frame_trace::host_seconds_to_ns(gpu_end_seconds);
    if event.gpu_start_time_ns == 0
        || event.gpu_end_time_ns == 0
        || event.gpu_end_time_ns < event.gpu_start_time_ns
    {
        event.flags |= crate::frame_trace::FLAG_GPU_TIMESTAMPS_INVALID;
    }
}

#[cfg(feature = "frame-trace")]
fn frame_trace_presentation_event(
    mut event: crate::frame_trace::FrameTraceEvent,
    presented_time_ns: u64,
    callback_observed_ns: u64,
    drawable_id: u64,
    previous_queue_depth: u64,
) -> crate::frame_trace::FrameTraceEvent {
    let submitted_drawable_id = event.drawable_id;
    event.kind = if presented_time_ns == 0 {
        crate::frame_trace::FrameTraceEventKind::DrawableDropped
    } else {
        crate::frame_trace::FrameTraceEventKind::DrawablePresented
    };
    event.timestamp_ns = if presented_time_ns == 0 {
        callback_observed_ns
    } else {
        presented_time_ns
    };
    event.callback_observed_ns = callback_observed_ns;
    event.drawable_id = crate::frame_trace::encode_drawable_id(drawable_id);
    event.presentation_queue_depth = previous_queue_depth.saturating_sub(1);
    if presented_time_ns == 0 {
        event.flags |= crate::frame_trace::FLAG_PRESENTATION_TIMESTAMP_INVALID;
    }
    if event.presentation_target_display_time_ns != 0
        && presented_time_ns
            > event
                .presentation_target_display_time_ns
                .saturating_add(1_000_000)
    {
        event.flags |= crate::frame_trace::FLAG_PRESENTED_AFTER_DISPLAY_TARGET;
    }
    if previous_queue_depth == 0 {
        event.flags |= crate::frame_trace::FLAG_QUEUE_DEPTH_UNDERFLOW;
    }
    if event.drawable_id != submitted_drawable_id {
        event.flags |= crate::frame_trace::FLAG_DRAWABLE_ID_MISMATCH;
    }
    event
}

// Exported to metal
pub(crate) type PointF = crate::Point<f32>;

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;
/// Сколько раз подряд показывать заново сцену, чей последний drawable Metal отбросил.
const MAX_DROPPED_FRAME_REPRESENTS: u64 = 3;

/// Очередь показа корневого слоя: порядковые номера отправленных кадров и последнего, чей
/// drawable Metal показал или отбросил. Metal показывает drawable по порядку, поэтому итог
/// позднего кадра закрывает и все ранние, даже если обработчик раннего не пришёл.
#[derive(Default)]
struct RootPresentationQueue {
    submitted: AtomicU64,
    completed: AtomicU64,
}

impl RootPresentationQueue {
    /// Отправка кадра на показ. Вызывает только главный поток. Возвращает номер кадра в очереди.
    fn submit(&self) -> u64 {
        self.submitted.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Итог показа кадра с номером `sequence`. Вызывает обработчик показа Metal.
    fn complete(&self, sequence: u64) {
        self.completed.fetch_max(sequence, Ordering::AcqRel);
    }

    fn in_flight(&self) -> u64 {
        let completed = self.completed.load(Ordering::Acquire);
        self.submitted
            .load(Ordering::Acquire)
            .saturating_sub(completed)
    }
}

/// Ждал бы `nextDrawable` свободный drawable. Кадр на экране держит свой drawable, пока его не
/// сменит следующий показ, поэтому очередь вмещает `maximum_drawable_count - 1` кадров.
fn root_presentation_would_block(in_flight: u64, maximum_drawable_count: u64) -> bool {
    in_flight >= maximum_drawable_count.saturating_sub(1).max(1)
}

/// Ждал бы показ drawable. `prefetched_drawable_ready` — состояние заранее взятого drawable при
/// заблаговременном drawable (по умолчанию) вне синхронного кадра, `None` — прежний путь: оценка по очереди.
fn presentation_would_block(
    prefetched_drawable_ready: Option<bool>,
    in_flight: u64,
    maximum_drawable_count: u64,
) -> bool {
    match prefetched_drawable_ready {
        Some(ready) => !ready,
        None => root_presentation_would_block(in_flight, maximum_drawable_count),
    }
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

/// Сколько синхронный кадр ждёт заранее взятый drawable: около трёх тактов дисплея 120 Гц.
const SYNC_FRAME_DRAWABLE_WAIT: std::time::Duration = std::time::Duration::from_millis(24);

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

#[derive(Clone, Copy, PartialEq, Eq)]
enum DrawableMiss {
    Unavailable,
    NotReady,
}

pub(crate) struct MetalRenderer {
    device: metal::Device,
    layer: metal::MetalLayer,
    presents_with_transaction: bool,
    /// Заранее взятые drawable основного слоя (по умолчанию; `GPUI_ASYNC_DRAWABLE=0` отключает). `None` — прежний
    /// режим: `nextDrawable` блокирует главный поток.
    drawable_prefetcher: Option<DrawablePrefetcher<metal::MetalDrawable>>,
    /// Показ отложен, потому что drawable не был готов. Сцена остаётся в окне, показ
    /// повторяет такт DisplayLink или сигнал готовности.
    deferred_presentation: bool,
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
    /// Тесты сравнивают проходы путей с эталонными: по всему кадру и с прежним копированием.
    #[cfg(test)]
    reference_path_passes: bool,
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
    /// Номер последнего кадра основного слоя, отправленного на показ.
    last_submitted_frame: Arc<AtomicU64>,
    /// Номер кадра, чей drawable Metal отбросил, не показав, когда новее ничего не отправлено.
    /// Ноль — повторный показ не нужен.
    dropped_latest_frame: Arc<AtomicU64>,
    /// Сколько последних кадров подряд отброшено. Сбрасывается любым показанным кадром и
    /// ограничивает повторы, чтобы скрытое окно не перерисовывалось на каждом такте.
    consecutive_dropped_frames: Arc<AtomicU64>,
    root_presentation_queue: Arc<RootPresentationQueue>,
    #[cfg(feature = "frame-trace")]
    frame_trace_renderer_instance_id: u64,
    #[cfg(feature = "frame-trace")]
    frame_trace_presentation_queue: Arc<PresentationQueueTrace>,
    #[cfg(feature = "frame-trace")]
    frame_trace_command_metadata: Arc<FrameTraceCommandMetadataTable>,
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
        let frame_trace_presentation_queue = Arc::new(PresentationQueueTrace::default());
        #[cfg(feature = "frame-trace")]
        let frame_trace_command_metadata = Arc::new(FrameTraceCommandMetadataTable::new());
        #[cfg(feature = "frame-trace")]
        let frame_trace_scheduled_handler = ConcreteBlock::new({
            let presentation_queue = frame_trace_presentation_queue.clone();
            let command_metadata = frame_trace_command_metadata.clone();
            move |command_buffer: &'static metal::CommandBufferRef| {
                if !crate::frame_trace::is_detailed_enabled() {
                    return;
                }
                let mut event = crate::frame_trace::FrameTraceEvent::now(
                    crate::frame_trace::FrameTraceEventKind::GpuScheduled,
                );
                event.command_buffer_id = command_buffer.as_ptr() as usize as u64;
                command_metadata.populate(event.command_buffer_id, &mut event);
                event.presentation_queue_depth = presentation_queue.depth();
                crate::frame_trace::record(event);
            }
        })
        .copy();
        #[cfg(feature = "frame-trace")]
        let frame_trace_completed_handler = ConcreteBlock::new({
            let presentation_queue = frame_trace_presentation_queue.clone();
            let command_metadata = frame_trace_command_metadata.clone();
            move |command_buffer: &'static metal::CommandBufferRef| {
                let command_buffer_key = command_buffer.as_ptr() as usize as u64;
                if !crate::frame_trace::is_detailed_enabled() {
                    command_metadata.remove(command_buffer_key);
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
                event.command_buffer_id = command_buffer_key;
                command_metadata.populate(event.command_buffer_id, &mut event);
                command_metadata.remove(command_buffer_key);
                event.presentation_queue_depth = presentation_queue.depth();
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
            drawable_prefetcher: None,
            deferred_presentation: false,
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
            #[cfg(test)]
            reference_path_passes: false,
            raster_textures: HashMap::new(),
            raster_namespaces: HashMap::new(),
            raster_compositor_layers: HashMap::new(),
            raster_compositor_deferred_renders: Vec::new(),
            frame_index: 0,
            presented_frame_samples: Arc::new(Mutex::new(Vec::new())),
            raster_compositor_presentation_samples: Arc::new(Mutex::new(Vec::new())),
            last_submitted_frame: Arc::new(AtomicU64::new(0)),
            dropped_latest_frame: Arc::new(AtomicU64::new(0)),
            consecutive_dropped_frames: Arc::new(AtomicU64::new(0)),
            root_presentation_queue: Arc::default(),
            #[cfg(feature = "frame-trace")]
            frame_trace_renderer_instance_id: crate::frame_trace::next_renderer_instance_id(),
            #[cfg(feature = "frame-trace")]
            frame_trace_presentation_queue,
            #[cfg(feature = "frame-trace")]
            frame_trace_scheduled_handler,
            #[cfg(feature = "frame-trace")]
            frame_trace_command_metadata,
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
        scene.populate_frame_trace_event(&mut event);
        project_frame_trace_capture_level(&mut event, crate::frame_trace::is_detailed_enabled());
        event.renderer_instance_id = self.frame_trace_renderer_instance_id;
        event.renderer_frame_id = self.frame_index;
        event.drawable_id = drawable_id;
        event.command_buffer_id = command_buffer_id;
        event.presentation_queue_depth = self.frame_trace_presentation_queue.depth();
        if scene.frame_trace_logical_frame_id != 0
            && scene.frame_trace_logical_frame_id == self.frame_trace_last_logical_frame_id
        {
            event.flags |= crate::frame_trace::FLAG_REUSED_SCENE;
        }
        event
    }

    #[cfg(feature = "frame-trace")]
    fn record_frame_submission_skipped(
        &self,
        scene: &Scene,
        reason: crate::frame_trace::FrameSubmissionSkipReason,
        next_drawable_wait_ns: u64,
    ) {
        let mut event = self.frame_trace_event(
            crate::frame_trace::FrameTraceEventKind::FrameSubmissionSkipped,
            scene,
            0,
            0,
        );
        event.submission_skip_reason = reason as u64;
        event.next_drawable_wait_ns = next_drawable_wait_ns;
        crate::frame_trace::record(event);
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

    /// Metal отбросил drawable последнего отправленного кадра, и новее ничего не отправлено:
    /// на экране остался предыдущий кадр, и сцену нужно показать ещё раз. Сбрасывает признак.
    pub fn take_dropped_latest_frame(&self) -> bool {
        self.dropped_latest_frame.swap(0, Ordering::AcqRel) != 0
    }

    /// Ждал бы показ сейчас свободный drawable корневого слоя: все, кроме кадра на экране, ещё в
    /// очереди показа. Обработчик показа приходит чуть позже освобождения drawable, поэтому
    /// ответ осторожный: «ждал бы» может длиться ещё доли миллисекунды после освобождения.
    ///
    /// Когда drawable берётся заранее (по умолчанию), главный поток вне синхронных кадров не ждёт `nextDrawable`,
    /// поэтому ответ — готов ли заранее взятый drawable, а не оценка по очереди показа. Иначе
    /// запаздывающий обработчик показа пропускал бы повтор, для которого drawable уже в руках.
    pub fn root_presentation_would_block(&self) -> bool {
        let prefetched_drawable_ready = (self.drawable_prefetcher.is_some()
            && !self.presents_with_transaction)
            .then(|| self.root_drawable_ready_for_frame());
        presentation_would_block(
            prefetched_drawable_ready,
            self.root_presentation_queue.in_flight(),
            self.layer.maximum_drawable_count(),
        )
    }

    /// Включает заблаговременное получение drawable, если не задано `GPUI_ASYNC_DRAWABLE=0`.
    /// `notify_ready` вызывается из вспомогательного потока, когда отложенный показ можно
    /// повторить, и должен запросить кадр на главном потоке.
    pub fn start_drawable_prefetch(&mut self, notify_ready: impl Fn() + Send + 'static) {
        if !async_drawable_enabled() || self.drawable_prefetcher.is_some() {
            return;
        }
        let layer = self.layer.clone();
        match DrawablePrefetcher::spawn(
            "gpui-drawable-prefetch",
            move || objc::rc::autoreleasepool(|| layer.next_drawable().map(ToOwned::to_owned)),
            notify_ready,
        ) {
            Ok(prefetcher) => self.drawable_prefetcher = Some(prefetcher),
            Err(error) => log::error!("failed to start drawable prefetch thread: {error}"),
        }
    }

    /// Можно ли строить кадр на этом такте. В режиме заранее взятого drawable (по умолчанию) без готового
    /// drawable корневого слоя ответ «нет»: кадр, собранный сейчас, ушёл бы в пропуск и был бы
    /// вытеснен. Тогда drawable запрашивается, и сигнал готовности запросит кадр снова. В прежнем
    /// режиме и для синхронных кадров ответ всегда «да».
    pub fn root_drawable_ready_for_frame(&self) -> bool {
        let Some(prefetcher) = &self.drawable_prefetcher else {
            return true;
        };
        if self.presents_with_transaction {
            return true;
        }
        let viewport_size = self.layer.drawable_size();
        let width = viewport_size.width.ceil() as u64;
        let height = viewport_size.height.ceil() as u64;
        prefetcher.poll_ready(|drawable: &metal::MetalDrawable| {
            let texture = drawable.texture();
            texture.width() == width && texture.height() == height
        })
    }

    /// Показ был отложен из-за неготового drawable и ещё не повторён. Сбрасывает признак.
    pub fn take_deferred_presentation(&mut self) -> bool {
        mem::take(&mut self.deferred_presentation)
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        self.layer
            .set_presents_with_transaction(presents_with_transaction);
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
                Ok((command_buffer, deferred_renders, _)) => {
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
        const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;
        let required = required_instance_buffer_size(scene);
        #[cfg(feature = "frame-trace")]
        let frame_trace_instance_buffer_limit = required > MAX_INSTANCE_BUFFER_SIZE;
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
        drop(pool);
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
        let frame_trace_reference_timed = crate::frame_trace::is_reference_timing_enabled();
        #[cfg(feature = "frame-trace")]
        let next_drawable_started_ns =
            crate::frame_trace::is_enabled().then(crate::frame_trace::monotonic_time_ns);
        #[cfg(feature = "frame-trace")]
        if frame_trace_detailed {
            let mut event = self.frame_trace_event(
                crate::frame_trace::FrameTraceEventKind::NextDrawableStarted,
                scene,
                0,
                0,
            );
            event.timestamp_ns = next_drawable_started_ns.unwrap_or_default();
            crate::frame_trace::record(event);
        }
        let drawable_matches_viewport = |drawable: &metal::MetalDrawable| {
            let texture = drawable.texture();
            texture.width() == viewport_size.width.0 as u64
                && texture.height() == viewport_size.height.0 as u64
        };
        let acquired_drawable = match &self.drawable_prefetcher {
            None => layer
                .next_drawable()
                .map(ToOwned::to_owned)
                .ok_or(DrawableMiss::Unavailable),
            // Синхронный кадр (resize, becomeKey) ждёт drawable ограниченное время: предыдущий
            // освобождается лишь после фиксации транзакции, которую держит этот же поток, и
            // бесконечное ожидание длилось бы до секунды. Не дождавшись, кадр откладывается.
            Some(prefetcher) => match if self.presents_with_transaction {
                prefetcher.take_blocking(drawable_matches_viewport, SYNC_FRAME_DRAWABLE_WAIT)
            } else {
                prefetcher.try_take(drawable_matches_viewport)
            } {
                DrawableTake::Ready(drawable) => Ok(drawable),
                DrawableTake::NotReady => Err(DrawableMiss::NotReady),
                DrawableTake::Unavailable => Err(DrawableMiss::Unavailable),
            },
        };
        let acquired_drawable = match acquired_drawable {
            Ok(drawable) => drawable,
            Err(miss) => {
                #[cfg(feature = "frame-trace")]
                if let Some(started_ns) = next_drawable_started_ns {
                    self.record_frame_submission_skipped(
                        scene,
                        match miss {
                            DrawableMiss::Unavailable => {
                                crate::frame_trace::FrameSubmissionSkipReason::NextDrawableUnavailable
                            }
                            DrawableMiss::NotReady => {
                                crate::frame_trace::FrameSubmissionSkipReason::DrawableNotReady
                            }
                        },
                        crate::frame_trace::monotonic_time_ns().saturating_sub(started_ns),
                    );
                }
                match miss {
                    DrawableMiss::Unavailable => log::error!(
                        "failed to retrieve next drawable, drawable size: {:?}",
                        viewport_size
                    ),
                    DrawableMiss::NotReady => self.deferred_presentation = true,
                }
                let deferred = mem::take(&mut self.raster_compositor_deferred_renders);
                self.commit_deferred_renders(deferred);
                return;
            }
        };
        let drawable: &metal::MetalDrawableRef = &acquired_drawable;
        #[cfg(feature = "frame-trace")]
        let trace_next_drawable_wait_ns = next_drawable_started_ns
            .map(|next_drawable_started_ns| {
                let mut event = self.frame_trace_event(
                    crate::frame_trace::FrameTraceEventKind::DrawableAcquired,
                    scene,
                    crate::frame_trace::encode_drawable_id(drawable.drawable_id() as u64),
                    0,
                );
                event.next_drawable_wait_ns =
                    event.timestamp_ns.saturating_sub(next_drawable_started_ns);
                crate::frame_trace::record(event);
                event.next_drawable_wait_ns
            })
            .unwrap_or_default();

        loop {
            let mut instance_buffer = self.instance_buffer_pool.lock().acquire(&self.device);

            let command_buffers =
                self.draw_primitives(scene, &mut instance_buffer, drawable, viewport);

            match command_buffers {
                Ok((command_buffer, deferred_renders, trace_command_buffer_id)) => {
                    let instance_buffer_pool = self.instance_buffer_pool.clone();
                    let instance_buffer = Cell::new(Some(instance_buffer));
                    let block = ConcreteBlock::new(move |_| {
                        if let Some(instance_buffer) = instance_buffer.take() {
                            instance_buffer_pool.lock().release(instance_buffer);
                        }
                    });
                    let block = block.copy();
                    command_buffer.add_completed_handler(&block);

                    let frame_id = self.frame_index;
                    let submitted_at = Instant::now();
                    let samples = self.presented_frame_samples.clone();
                    let last_submitted_frame = self.last_submitted_frame.clone();
                    let dropped_latest_frame = self.dropped_latest_frame.clone();
                    let consecutive_dropped_frames = self.consecutive_dropped_frames.clone();
                    let root_presentation_queue = self.root_presentation_queue.clone();
                    let root_presentation_sequence = self.root_presentation_queue.submit();
                    let measured_command_buffer = command_buffer.to_owned();
                    #[cfg(feature = "frame-trace")]
                    let trace_drawable_id =
                        crate::frame_trace::encode_drawable_id(drawable.drawable_id() as u64);
                    #[cfg(feature = "frame-trace")]
                    let trace_presented_event = self.frame_trace_event(
                        crate::frame_trace::FrameTraceEventKind::DrawablePresented,
                        scene,
                        trace_drawable_id,
                        trace_command_buffer_id,
                    );
                    #[cfg(feature = "frame-trace")]
                    if frame_trace_detailed {
                        command_buffer.add_scheduled_handler(&self.frame_trace_scheduled_handler);
                        command_buffer.add_completed_handler(&self.frame_trace_completed_handler);
                    }
                    #[cfg(feature = "frame-trace")]
                    let trace_presentation_queue = self.frame_trace_presentation_queue.clone();
                    let presented =
                        ConcreteBlock::new(move |presented_drawable: &metal::DrawableRef| {
                            root_presentation_queue.complete(root_presentation_sequence);
                            // presentedTime == 0: drawable отброшен. Если это последний отправленный
                            // кадр, его содержимое так и не попадёт на экран без повторного показа.
                            if presented_drawable.presented_time() > 0. {
                                consecutive_dropped_frames.store(0, Ordering::Release);
                            } else if last_submitted_frame.load(Ordering::Acquire) == frame_id
                                && consecutive_dropped_frames.fetch_add(1, Ordering::AcqRel)
                                    < MAX_DROPPED_FRAME_REPRESENTS
                            {
                                dropped_latest_frame.store(frame_id, Ordering::Release);
                            }
                            #[cfg(feature = "frame-trace")]
                            let callback_observed_ns = crate::frame_trace::monotonic_time_ns();
                            #[cfg(feature = "frame-trace")]
                            let presented_time_seconds = presented_drawable.presented_time();
                            let gpu_start: f64 = unsafe {
                                msg_send![measured_command_buffer.as_ref(), GPUStartTime]
                            };
                            let gpu_end: f64 =
                                unsafe { msg_send![measured_command_buffer.as_ref(), GPUEndTime] };
                            #[cfg(feature = "frame-trace")]
                            {
                                let presented_time_ns =
                                    crate::frame_trace::host_seconds_to_ns(presented_time_seconds);
                                trace_presentation_queue.record_presentation(
                                    |previous_queue_depth| {
                                        let mut event = frame_trace_presentation_event(
                                            trace_presented_event,
                                            presented_time_ns,
                                            callback_observed_ns,
                                            presented_drawable.drawable_id(),
                                            previous_queue_depth,
                                        );
                                        project_frame_trace_gpu_interval(
                                            &mut event,
                                            frame_trace_reference_timed,
                                            gpu_start,
                                            gpu_end,
                                        );
                                        crate::frame_trace::record(event);
                                    },
                                );
                            }
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
                    self.last_submitted_frame.store(frame_id, Ordering::Release);

                    #[cfg(feature = "frame-trace")]
                    let trace_presentation_queue = self.frame_trace_presentation_queue.clone();
                    // Блокировка очереди держится до записи события отправки: обработчик показа
                    // более раннего кадра не получит номер события между ними.
                    #[cfg(feature = "frame-trace")]
                    let prepare_submission_event = || {
                        let (queue_order, queue_depth) =
                            trace_presentation_queue.begin_submission();
                        let mut event = self.frame_trace_event(
                            crate::frame_trace::FrameTraceEventKind::CommandBufferSubmitted,
                            scene,
                            trace_drawable_id,
                            trace_command_buffer_id,
                        );
                        event.presentation_queue_depth = queue_depth;
                        event.timestamp_ns = crate::frame_trace::monotonic_time_ns();
                        if event.presentation_target_display_time_ns != 0
                            && event.timestamp_ns > event.presentation_target_display_time_ns
                        {
                            event.flags |= crate::frame_trace::FLAG_MISSED_DISPLAY_TARGET;
                        }
                        project_frame_trace_submission_timing(
                            &mut event,
                            frame_trace_reference_timed,
                            trace_next_drawable_wait_ns,
                        );
                        (queue_order, event)
                    };

                    if self.presents_with_transaction {
                        #[cfg(feature = "frame-trace")]
                        let (queue_order, submitted_event) = prepare_submission_event();
                        #[cfg(feature = "frame-trace")]
                        if frame_trace_detailed {
                            self.frame_trace_command_metadata.register(
                                command_buffer.as_ptr() as usize as u64,
                                &submitted_event,
                            );
                        }
                        command_buffer.commit();
                        #[cfg(feature = "frame-trace")]
                        {
                            crate::frame_trace::record(submitted_event);
                            drop(queue_order);
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
                        command_buffer.present_drawable(drawable);
                        #[cfg(feature = "frame-trace")]
                        let (queue_order, submitted_event) = prepare_submission_event();
                        #[cfg(feature = "frame-trace")]
                        if frame_trace_detailed {
                            self.frame_trace_command_metadata.register(
                                command_buffer.as_ptr() as usize as u64,
                                &submitted_event,
                            );
                        }
                        command_buffer.commit();
                        #[cfg(feature = "frame-trace")]
                        {
                            crate::frame_trace::record(submitted_event);
                            drop(queue_order);
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
                        self.frame_trace_last_logical_frame_id = scene.frame_trace_logical_frame_id;
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
        #[cfg(feature = "frame-trace")]
        if crate::frame_trace::is_enabled() {
            self.record_frame_submission_skipped(
                scene,
                if frame_trace_instance_buffer_limit {
                    crate::frame_trace::FrameSubmissionSkipReason::InstanceBufferLimit
                } else {
                    crate::frame_trace::FrameSubmissionSkipReason::EncodeRetryExhausted
                },
                0,
            );
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
    ) -> Result<(metal::CommandBuffer, Vec<DeferredRender>, u64)> {
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        #[cfg(feature = "frame-trace")]
        let trace_command_buffer_id = if crate::frame_trace::is_enabled() {
            crate::frame_trace::next_command_buffer_id()
        } else {
            0
        };
        #[cfg(not(feature = "frame-trace"))]
        let trace_command_buffer_id = 0;
        #[cfg(feature = "frame-trace")]
        if crate::frame_trace::is_detailed_enabled() && scene.frame_trace_logical_frame_id != 0 {
            let drawable_id = crate::frame_trace::encode_drawable_id(drawable.drawable_id());
            let command_buffer_id = trace_command_buffer_id;
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
        Ok((
            command_buffer.to_owned(),
            deferred_renders,
            trace_command_buffer_id,
        ))
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

        let path_plan = self.plan_path_passes(scene, viewport);
        // Первый проход путей идёт до основного: пакеты, которым он подходит, наносятся на кадр
        // без разрыва основного прохода.
        let mut rasterized_path_passes = 0;
        if let Some(first) = path_plan.passes.first() {
            if !self.draw_paths_to_intermediate(
                first,
                instance_buffer,
                instance_offset,
                viewport,
                command_buffer,
                path_intermediate.as_ref(),
                path_intermediate_msaa.as_ref(),
            ) {
                anyhow::bail!(
                    "scene too large at paths: offset={} capacity={}; {} paths",
                    *instance_offset,
                    instance_buffer.size,
                    scene.paths.len(),
                );
            }
            rasterized_path_passes = 1;
        }

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

        for (batch_index, batch) in scene.batches().enumerate() {
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
                PrimitiveBatch::Paths(_) => {
                    // Пакет вне цели рендера ничего не рисует: основной проход не прерывается.
                    let Some(planned) = path_plan.batches.get(&batch_index) else {
                        continue;
                    };
                    let mut did_draw = true;
                    if planned.pass >= rasterized_path_passes {
                        command_encoder.end_encoding();
                        did_draw = self.draw_paths_to_intermediate(
                            &path_plan.passes[planned.pass],
                            instance_buffer,
                            instance_offset,
                            viewport,
                            command_buffer,
                            path_intermediate.as_ref(),
                            path_intermediate_msaa.as_ref(),
                        );
                        rasterized_path_passes = planned.pass + 1;
                        command_encoder = new_command_encoder(
                            command_buffer,
                            target,
                            viewport,
                            |color_attachment| {
                                color_attachment.set_load_action(metal::MTLLoadAction::Load);
                            },
                        );
                    }

                    did_draw
                        && self.draw_paths_from_intermediate(
                            &planned.sprites,
                            path_plan.passes[planned.pass].area,
                            instance_buffer,
                            instance_offset,
                            viewport,
                            command_encoder,
                            path_intermediate.as_ref(),
                        )
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

    /// Раскладывает пакеты путей сцены по проходам растеризации в промежуточную текстуру.
    ///
    /// Пакет присоединяется к текущему проходу, если ячейки, которых касаются его треугольники,
    /// не заняты пакетами прохода, а рамка прохода растёт не больше площади пакета. Тогда
    /// пакеты не мешают друг другу в общей текстуре, а проход не дороже отдельных. Пакет с
    /// прежним копированием (без ячеек) занимает проход один.
    fn plan_path_passes<'a>(&self, scene: &'a Scene, viewport: Viewport) -> PathPassPlan<'a> {
        let mut plan = PathPassPlan::default();
        if !scene_needs_path_intermediate(scene) {
            return plan;
        }
        let grid_columns = (viewport.size.width.0.max(0) as u32).div_ceil(PATH_COPY_CELL) as usize;
        let grid_rows = (viewport.size.height.0.max(0) as u32).div_ceil(PATH_COPY_CELL) as usize;
        let mut occupancy = vec![false; grid_columns * grid_rows];
        let mut shareable_pass = false;
        for (batch_index, batch) in scene.batches().enumerate() {
            let PrimitiveBatch::Paths(paths) = batch else {
                continue;
            };
            let Some(region) = self.path_batch_region(paths, viewport) else {
                continue;
            };
            let clips = path_copy_clips(paths);
            let cells = TouchedCells::of_paths(region, paths, viewport);
            let cell_sprites = self.path_copy_sprites(&cells, region, &clips, viewport);
            let shareable = cell_sprites.is_some();
            let current = plan.passes.last_mut().filter(|pass| {
                if !(shareable_pass && shareable) {
                    return false;
                }
                let merged = pass.area.union(region);
                merged.area() <= pass.area.area() + region.area()
                    && !cells.overlaps(&occupancy, grid_columns)
            });
            if let Some(pass) = current {
                pass.area = pass.area.union(region);
                pass.paths.push(paths);
            } else {
                plan.passes.push(PathPass {
                    area: region,
                    paths: vec![paths],
                });
                occupancy.fill(false);
            }
            if shareable {
                cells.occupy(&mut occupancy, grid_columns);
            }
            shareable_pass = shareable;
            plan.batches.insert(
                batch_index,
                PlannedPathBatch {
                    pass: plan.passes.len() - 1,
                    sprites: cell_sprites.unwrap_or(clips),
                },
            );
        }
        plan
    }

    /// Область цели рендера, которой касается пакет путей; `None`, если пакет её не касается.
    fn path_batch_region(
        &self,
        paths: &[Path<ScaledPixels>],
        viewport: Viewport,
    ) -> Option<PathBatchRegion> {
        #[cfg(test)]
        if self.reference_path_passes {
            return PathBatchRegion::full_frame(viewport);
        }
        PathBatchRegion::of_paths(paths, viewport)
    }

    fn path_copy_sprites(
        &self,
        cells: &TouchedCells,
        region: PathBatchRegion,
        clips: &[PathSprite],
        viewport: Viewport,
    ) -> Option<Vec<PathSprite>> {
        #[cfg(test)]
        if self.reference_path_passes {
            return None;
        }
        cells.copy_sprites(region, clips, viewport)
    }

    /// Растеризует пути прохода в его область промежуточной текстуры.
    #[allow(clippy::too_many_arguments)]
    fn draw_paths_to_intermediate(
        &self,
        pass: &PathPass,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport: Viewport,
        command_buffer: &metal::CommandBufferRef,
        intermediate_texture: Option<&metal::Texture>,
        intermediate_msaa_texture: Option<&metal::Texture>,
    ) -> bool {
        let Some(intermediate_texture) = intermediate_texture else {
            return false;
        };
        let region = pass.area;

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        // Проход очищает, растеризует и сводит MSAA только в своей области: она лежит в левом
        // верхнем углу промежуточной текстуры, а не на своём месте во всём кадре.
        render_pass_descriptor.set_render_target_width(region.width as u64);
        render_pass_descriptor.set_render_target_height(region.height as u64);
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
        command_encoder.set_viewport(metal::MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: region.width as f64,
            height: region.height as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        let region_viewport = Viewport {
            size: size(
                DevicePixels(region.width as i32),
                DevicePixels(region.height as i32),
            ),
            origin: point(
                viewport.origin.x + ScaledPixels(region.x as f32),
                viewport.origin.y + ScaledPixels(region.y as f32),
            ),
        };
        let intermediate_origin = region.origin();

        for path in pass.paths.iter().flat_map(|paths| paths.iter()) {
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
                mem::size_of_val(&region_viewport) as u64,
                &region_viewport as *const Viewport as *const _,
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
            command_encoder.set_fragment_bytes(
                PathRasterizationInputIndex::IntermediateOrigin as u64,
                mem::size_of_val(&intermediate_origin) as u64,
                &intermediate_origin as *const [f32; 2] as *const _,
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

    /// Наносит пакет путей на кадр прямоугольниками `sprites` из области прохода `area`.
    #[allow(clippy::too_many_arguments)]
    fn draw_paths_from_intermediate(
        &self,
        sprites: &[PathSprite],
        area: PathBatchRegion,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport: Viewport,
        command_encoder: &metal::RenderCommandEncoderRef,
        intermediate_texture: Option<&metal::Texture>,
    ) -> bool {
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

        let intermediate_origin = area.origin();
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::IntermediateOrigin as u64,
            mem::size_of_val(&intermediate_origin) as u64,
            &intermediate_origin as *const [f32; 2] as *const _,
        );

        command_encoder.set_fragment_texture(
            SpriteInputIndex::AtlasTexture as u64,
            Some(intermediate_texture),
        );

        align_offset(instance_offset);
        let sprite_bytes_len = mem::size_of_val(sprites);
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

        if !sprites.is_empty() {
            command_encoder.draw_primitives_instanced(
                metal::MTLPrimitiveType::Triangle,
                0,
                6,
                sprites.len() as u64,
            );
        }
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
mod root_presentation_queue_tests {
    use super::*;

    #[test]
    fn queue_counts_frames_until_their_presentation_outcome() {
        let queue = RootPresentationQueue::default();
        assert_eq!(queue.in_flight(), 0);
        let first = queue.submit();
        let second = queue.submit();
        assert_eq!(queue.in_flight(), 2);
        queue.complete(first);
        assert_eq!(queue.in_flight(), 1);
        queue.complete(second);
        assert_eq!(queue.in_flight(), 0);
    }

    #[test]
    fn later_outcome_closes_earlier_frames_and_late_callback_does_not_reopen_them() {
        let queue = RootPresentationQueue::default();
        let first = queue.submit();
        let second = queue.submit();
        queue.complete(second);
        assert_eq!(queue.in_flight(), 0);
        queue.complete(first);
        assert_eq!(queue.in_flight(), 0);
        queue.submit();
        assert_eq!(queue.in_flight(), 1);
    }

    #[test]
    fn without_async_drawable_would_block_is_the_queue_estimate() {
        for in_flight in 0..4 {
            for count in 1..4 {
                assert_eq!(
                    presentation_would_block(None, in_flight, count),
                    root_presentation_would_block(in_flight, count),
                );
            }
        }
    }

    #[test]
    fn with_async_drawable_would_block_follows_prefetched_drawable() {
        // Drawable в руках: повтор не ждёт, даже если обработчик показа ещё не пришёл.
        assert!(!presentation_would_block(Some(true), 1, 2));
        assert!(presentation_would_block(Some(false), 0, 2));
    }

    #[test]
    fn presentation_waits_when_every_drawable_but_the_screen_one_is_queued() {
        assert!(!root_presentation_would_block(0, 2));
        assert!(root_presentation_would_block(1, 2));
        assert!(!root_presentation_would_block(1, 3));
        assert!(root_presentation_would_block(2, 3));
        // Слой с одним drawable не бывает, но и тогда очередь не считается бесконечной.
        assert!(root_presentation_would_block(1, 1));
        assert!(!root_presentation_would_block(0, 1));
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
    #[cfg(feature = "frame-trace")]
    #[test]
    fn reference_capture_omits_detailed_display_coalescing() {
        let mut event = crate::frame_trace::FrameTraceEvent::now(
            crate::frame_trace::FrameTraceEventKind::CommandBufferSubmitted,
        );
        event.coalesced_display_tick_count = 3;
        project_frame_trace_capture_level(&mut event, false);
        assert_eq!(event.coalesced_display_tick_count, 0);

        event.coalesced_display_tick_count = 3;
        project_frame_trace_capture_level(&mut event, true);
        assert_eq!(event.coalesced_display_tick_count, 3);
    }

    #[cfg(feature = "frame-trace")]
    #[test]
    fn reference_timed_capture_fills_frame_timing_fields() {
        use crate::frame_trace::{
            FLAG_GPU_TIMESTAMPS_INVALID, FrameTraceEvent, FrameTraceEventKind,
        };

        let mut scene = Scene::default();
        scene.set_frame_trace_correlation(5, 0, None);
        scene.set_frame_trace_logical_frame_started_ns(1_000);

        for reference_timed in [false, true] {
            let mut submitted = FrameTraceEvent::now(FrameTraceEventKind::CommandBufferSubmitted);
            scene.populate_frame_trace_event(&mut submitted);
            project_frame_trace_submission_timing(&mut submitted, reference_timed, 250);
            assert_eq!(submitted.logical_frame_started_ns, 1_000);
            assert_eq!(
                submitted.next_drawable_wait_ns,
                if reference_timed { 250 } else { 0 }
            );

            let mut template = FrameTraceEvent::now(FrameTraceEventKind::DrawablePresented);
            scene.populate_frame_trace_event(&mut template);
            let mut presented = frame_trace_presentation_event(template, 3_000, 3_100, 0, 1);
            project_frame_trace_gpu_interval(&mut presented, reference_timed, 1.25, 1.5);
            assert_eq!(presented.kind, FrameTraceEventKind::DrawablePresented);
            assert_eq!(presented.logical_frame_started_ns, 1_000);
            assert_eq!(presented.next_drawable_wait_ns, 0);
            if reference_timed {
                assert_eq!(presented.gpu_start_time_ns, 1_250_000_000);
                assert_eq!(presented.gpu_end_time_ns, 1_500_000_000);
            } else {
                assert_eq!(presented.gpu_start_time_ns, 0);
                assert_eq!(presented.gpu_end_time_ns, 0);
            }
            assert_eq!(presented.flags & FLAG_GPU_TIMESTAMPS_INVALID, 0);
        }

        for (gpu_start_seconds, gpu_end_seconds) in [(0.0, 1.5), (1.5, 0.0), (1.5, 1.25)] {
            let mut dropped = FrameTraceEvent::now(FrameTraceEventKind::DrawableDropped);
            project_frame_trace_gpu_interval(
                &mut dropped,
                true,
                gpu_start_seconds,
                gpu_end_seconds,
            );
            assert_ne!(dropped.flags & FLAG_GPU_TIMESTAMPS_INVALID, 0);
        }

        scene.clear();
        let mut submitted = FrameTraceEvent::now(FrameTraceEventKind::CommandBufferSubmitted);
        scene.populate_frame_trace_event(&mut submitted);
        assert_eq!(submitted.logical_frame_started_ns, 0);
    }

    #[cfg(feature = "frame-trace")]
    #[test]
    fn frame_trace_command_metadata_preserves_token_and_both_ticks() {
        let table = FrameTraceCommandMetadataTable::new();
        let token = crate::frame_trace::PresentationToken {
            run_id_hash: 1,
            input_sequence: 2,
            snapshot_generation: 3,
            canvas_render_generation: 4,
        };
        let mut submitted = crate::frame_trace::FrameTraceEvent::now(
            crate::frame_trace::FrameTraceEventKind::CommandBufferSubmitted,
        );
        submitted.command_buffer_id = 41;
        submitted.input_sequence_id = 2;
        submitted.logical_frame_id = 5;
        submitted.gpui_window_frame_id = 6;
        submitted.renderer_instance_id = 7;
        submitted.renderer_frame_id = 8;
        submitted.drawable_id = 9;
        submitted.presentation_token = Some(token);
        submitted.display_id = 10;
        submitted.display_current_time_ns = 11;
        submitted.display_refresh_period_ns = 12;
        submitted.scene_build_display_tick_sequence = 13;
        submitted.scene_build_target_display_time_ns = 14;
        submitted.presentation_display_tick_sequence = 15;
        submitted.scene_build_tick_flags = 17;
        submitted.presentation_tick_flags = 18;
        submitted.presentation_target_display_time_ns = 16;
        submitted.flags = crate::frame_trace::FLAG_MISSED_DISPLAY_TARGET;
        submitted.coalesced_display_tick_count = 1;
        table.register(99, &submitted);

        let mut callback = crate::frame_trace::FrameTraceEvent::now(
            crate::frame_trace::FrameTraceEventKind::GpuCompleted,
        );
        callback.command_buffer_id = 99;
        table.populate(99, &mut callback);

        assert_eq!(callback.command_buffer_id, 41);
        assert_eq!(callback.presentation_token, Some(token));
        assert_eq!(callback.logical_frame_id, 5);
        assert_eq!(callback.gpui_window_frame_id, 6);
        assert_eq!(callback.renderer_instance_id, 7);
        assert_eq!(callback.renderer_frame_id, 8);
        assert_eq!(callback.drawable_id, 9);
        assert_eq!(callback.display_id, 10);
        assert_eq!(callback.scene_build_display_tick_sequence, 13);
        assert_eq!(callback.scene_build_target_display_time_ns, 14);
        assert_eq!(callback.presentation_display_tick_sequence, 15);
        assert_eq!(callback.scene_build_tick_flags, 17);
        assert_eq!(callback.presentation_tick_flags, 18);
        assert_eq!(
            callback.flags,
            crate::frame_trace::FLAG_MISSED_DISPLAY_TARGET
        );
        assert_eq!(callback.presentation_target_display_time_ns, 16);

        table.remove(99);
        let mut reused_pointer_submission = submitted;
        reused_pointer_submission.command_buffer_id = 42;
        reused_pointer_submission.renderer_frame_id = 9;
        table.register(99, &reused_pointer_submission);
        let mut reused_pointer_callback = crate::frame_trace::FrameTraceEvent::now(
            crate::frame_trace::FrameTraceEventKind::GpuCompleted,
        );
        table.populate(99, &mut reused_pointer_callback);
        assert_eq!(reused_pointer_callback.command_buffer_id, 42);
        assert_eq!(reused_pointer_callback.renderer_frame_id, 9);
    }

    #[cfg(feature = "frame-trace")]
    #[test]
    fn frame_trace_terminal_event_distinguishes_presented_dropped_and_mismatch() {
        let mut template = crate::frame_trace::FrameTraceEvent::now(
            crate::frame_trace::FrameTraceEventKind::DrawablePresented,
        );
        template.drawable_id = crate::frame_trace::encode_drawable_id(7);
        template.presentation_target_display_time_ns = 1_000_000;

        let presented = frame_trace_presentation_event(template, 1_500_000, 1_600_000, 7, 1);
        assert_eq!(
            presented.kind,
            crate::frame_trace::FrameTraceEventKind::DrawablePresented
        );
        assert_eq!(presented.presentation_queue_depth, 0);
        assert_eq!(
            presented.flags & crate::frame_trace::FLAG_DRAWABLE_ID_MISMATCH,
            0
        );

        let dropped = frame_trace_presentation_event(template, 0, 2_000_000, 8, 0);
        assert_eq!(
            dropped.kind,
            crate::frame_trace::FrameTraceEventKind::DrawableDropped
        );
        assert_ne!(
            dropped.flags & crate::frame_trace::FLAG_PRESENTATION_TIMESTAMP_INVALID,
            0
        );
        assert_ne!(
            dropped.flags & crate::frame_trace::FLAG_QUEUE_DEPTH_UNDERFLOW,
            0
        );
        assert_ne!(
            dropped.flags & crate::frame_trace::FLAG_DRAWABLE_ID_MISMATCH,
            0
        );
    }

    #[cfg(feature = "frame-trace")]
    #[test]
    fn presentation_queue_depth_follows_event_order_across_threads() {
        use std::sync::mpsc;

        #[derive(Clone, Copy)]
        enum Change {
            Submitted(u64),
            Presented(u64),
        }
        const FRAMES: usize = 20_000;
        let queue = Arc::new(PresentationQueueTrace::default());
        let sequence = Arc::new(AtomicU64::new(0));
        let log = Arc::new(Mutex::new(Vec::with_capacity(FRAMES * 2)));
        let (submitted, presentable) = mpsc::channel::<()>();
        let presenter = std::thread::spawn({
            let queue = queue.clone();
            let sequence = sequence.clone();
            let log = log.clone();
            move || {
                for () in presentable {
                    queue.record_presentation(|previous| {
                        let id = sequence.fetch_add(1, Ordering::SeqCst);
                        log.lock().push((id, Change::Presented(previous)));
                    });
                }
            }
        });
        for _ in 0..FRAMES {
            let (order, depth) = queue.begin_submission();
            submitted.send(()).expect("presenter is running");
            // Показ уже может ждать блокировку: номер события отправки выдаётся раньше.
            std::thread::yield_now();
            let id = sequence.fetch_add(1, Ordering::SeqCst);
            log.lock().push((id, Change::Submitted(depth)));
            drop(order);
        }
        drop(submitted);
        presenter.join().expect("presenter thread");

        let mut log = log.lock().clone();
        log.sort_by_key(|(id, _)| *id);
        let mut depth = 0_u64;
        for (id, change) in log {
            match change {
                Change::Submitted(recorded) => {
                    depth += 1;
                    assert_eq!(recorded, depth, "submission event {id}");
                }
                Change::Presented(previous) => {
                    assert_eq!(previous, depth, "presentation event {id}");
                    depth -= 1;
                }
            }
        }
        assert_eq!(depth, 0);
    }

    #[cfg(feature = "frame-trace")]
    #[test]
    fn frame_trace_skip_reasons_remain_closed_and_stable() {
        assert_eq!(
            crate::frame_trace::FrameSubmissionSkipReason::NextDrawableUnavailable as u64,
            1
        );
        assert_eq!(
            crate::frame_trace::FrameSubmissionSkipReason::InstanceBufferLimit as u64,
            2
        );
        assert_eq!(
            crate::frame_trace::FrameSubmissionSkipReason::EncodeRetryExhausted as u64,
            3
        );
        assert_eq!(
            crate::frame_trace::FrameSubmissionSkipReason::DiagnosticHold as u64,
            4
        );
        assert_eq!(
            crate::frame_trace::FrameSubmissionSkipReason::DrawableNotReady as u64,
            5
        );
    }
}

/// Сторона ячейки сетки, по которой копирование пакета путей пропускает пустые места.
const PATH_COPY_CELL: u32 = 32;

/// Предел прямоугольников копирования пакета путей: буфер экземпляров резервирует его заранее.
const MAX_PATH_COPY_SPRITES: usize = 1024;

/// Прямоугольник цели рендера в пикселях устройства, которого касается пакет путей.
///
/// Растеризация пакета в промежуточную текстуру ограничена этой областью: цена прохода
/// зависит от площади путей, а не от размера кадра.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PathBatchRegion {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl PathBatchRegion {
    /// Объединение обрезанных границ путей, расширенное до целых пикселей с запасом в пиксель
    /// и обрезанное целью рендера. Запас покрывает округление позиций выборок в шейдере.
    fn of_paths(paths: &[Path<ScaledPixels>], viewport: Viewport) -> Option<Self> {
        let mut min = [f32::INFINITY; 2];
        let mut max = [f32::NEG_INFINITY; 2];
        for path in paths {
            let bounds = path.clipped_bounds();
            if bounds.is_empty() {
                continue;
            }
            let x = bounds.origin.x.0 - viewport.origin.x.0;
            let y = bounds.origin.y.0 - viewport.origin.y.0;
            min[0] = min[0].min(x);
            min[1] = min[1].min(y);
            max[0] = max[0].max(x + bounds.size.width.0);
            max[1] = max[1].max(y + bounds.size.height.0);
        }
        let target = [
            viewport.size.width.0.max(0) as f32,
            viewport.size.height.0.max(0) as f32,
        ];
        let x0 = (min[0].floor() - 1.0).max(0.0);
        let y0 = (min[1].floor() - 1.0).max(0.0);
        let x1 = (max[0].ceil() + 1.0).min(target[0]);
        let y1 = (max[1].ceil() + 1.0).min(target[1]);
        // Сравнение отбрасывает и пустое объединение, и NaN.
        if !(x1 > x0 && y1 > y0) {
            return None;
        }
        Some(Self {
            x: x0 as u32,
            y: y0 as u32,
            width: (x1 - x0) as u32,
            height: (y1 - y0) as u32,
        })
    }

    /// Охватывающий прямоугольник двух областей.
    fn union(&self, other: Self) -> Self {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        Self {
            x,
            y,
            width: (self.x + self.width).max(other.x + other.width) - x,
            height: (self.y + self.height).max(other.y + other.height) - y,
        }
    }

    fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    #[cfg(test)]
    fn full_frame(viewport: Viewport) -> Option<Self> {
        let width = viewport.size.width.0.max(0) as u32;
        let height = viewport.size.height.0.max(0) as u32;
        (width > 0 && height > 0).then_some(Self {
            x: 0,
            y: 0,
            width,
            height,
        })
    }

    fn origin(&self) -> [f32; 2] {
        [self.x as f32, self.y as f32]
    }
}

/// Проходы растеризации путей кадра и то, что каждый пакет путей наносит на кадр.
#[derive(Default)]
struct PathPassPlan<'a> {
    passes: Vec<PathPass<'a>>,
    /// Пакеты путей по номеру в `Scene::batches`; пакетов вне цели рендера здесь нет.
    batches: std::collections::HashMap<usize, PlannedPathBatch>,
}

/// Проход растеризации в промежуточную текстуру: его область и пути его пакетов.
struct PathPass<'a> {
    area: PathBatchRegion,
    paths: Vec<&'a [Path<ScaledPixels>]>,
}

struct PlannedPathBatch {
    pass: usize,
    sprites: Vec<PathSprite>,
}

/// Прежние прямоугольники копирования пакета: каждый пиксель копируется один раз.
///
/// If all paths have the same draw order, then their bounds are all
/// disjoint, so we can copy each path's bounds individually. If this
/// batch combines different draw orders, we perform a single copy
/// for a minimal spanning rect.
fn path_copy_clips(paths: &[Path<ScaledPixels>]) -> Vec<PathSprite> {
    let (Some(first), Some(last)) = (paths.first(), paths.last()) else {
        return Vec::new();
    };
    if first.order == last.order {
        return paths
            .iter()
            .map(|path| PathSprite {
                bounds: path.clipped_bounds(),
            })
            .collect();
    }
    let mut bounds = first.clipped_bounds();
    for path in &paths[1..] {
        bounds = bounds.union(&path.clipped_bounds());
    }
    vec![PathSprite { bounds }]
}

/// Ячейки сетки цели рендера со стороной [`PATH_COPY_CELL`], которых касаются треугольники
/// пакета путей: пиксель с покрытием лежит в рамке своего треугольника и в обрезанных
/// границах пути, рамка расширена на пиксель. Вне этих ячеек промежуточная текстура пакета
/// прозрачна. Сетка общая для всех пакетов кадра.
struct TouchedCells {
    column: usize,
    row: usize,
    columns: usize,
    rows: usize,
    touched: Vec<bool>,
}

impl TouchedCells {
    fn of_paths(region: PathBatchRegion, paths: &[Path<ScaledPixels>], viewport: Viewport) -> Self {
        let column = (region.x / PATH_COPY_CELL) as usize;
        let row = (region.y / PATH_COPY_CELL) as usize;
        let columns = (region.x + region.width).div_ceil(PATH_COPY_CELL) as usize - column;
        let rows = (region.y + region.height).div_ceil(PATH_COPY_CELL) as usize - row;
        // Разностная сетка: прямоугольник ячеек отмечается за O(1), затем префиксные суммы.
        let stride = columns + 1;
        let mut marks = vec![0i32; stride * (rows + 1)];
        let origin = [viewport.origin.x.0, viewport.origin.y.0];
        let first_cell = [column, row];
        let region_min = [region.x as f32, region.y as f32];
        let region_max = [
            (region.x + region.width) as f32,
            (region.y + region.height) as f32,
        ];
        for path in paths {
            let clip = path.clipped_bounds();
            let clip_min = [clip.origin.x.0 - origin[0], clip.origin.y.0 - origin[1]];
            let clip_max = [
                clip_min[0] + clip.size.width.0,
                clip_min[1] + clip.size.height.0,
            ];
            for triangle in path.vertices.chunks_exact(3) {
                let mut min = [f32::INFINITY; 2];
                let mut max = [f32::NEG_INFINITY; 2];
                let mut finite = true;
                for vertex in triangle {
                    let position = [
                        vertex.xy_position.x.0 - origin[0],
                        vertex.xy_position.y.0 - origin[1],
                    ];
                    for axis in 0..2 {
                        finite &= position[axis].is_finite();
                        min[axis] = min[axis].min(position[axis]);
                        max[axis] = max[axis].max(position[axis]);
                    }
                }
                if !finite {
                    // Рамку такого треугольника не оценить: берутся границы пути целиком.
                    min = clip_min;
                    max = clip_max;
                }
                let mut cells = [[0usize; 2]; 2];
                let mut empty = false;
                for axis in 0..2 {
                    let low = (min[axis].max(clip_min[axis]).floor() - 1.0).max(region_min[axis]);
                    let high = (max[axis].min(clip_max[axis]).ceil() + 1.0).min(region_max[axis]);
                    if high.partial_cmp(&low) != Some(std::cmp::Ordering::Greater) {
                        empty = true;
                        break;
                    }
                    cells[axis] = [
                        (low as u32 / PATH_COPY_CELL) as usize - first_cell[axis],
                        (high as u32).div_ceil(PATH_COPY_CELL) as usize - first_cell[axis],
                    ];
                }
                if empty {
                    continue;
                }
                let [[x0, x1], [y0, y1]] = cells;
                marks[y0 * stride + x0] += 1;
                marks[y0 * stride + x1] -= 1;
                marks[y1 * stride + x0] -= 1;
                marks[y1 * stride + x1] += 1;
            }
        }
        for row in 0..rows {
            for column in 1..columns {
                marks[row * stride + column] += marks[row * stride + column - 1];
            }
        }
        for row in 1..rows {
            for column in 0..columns {
                marks[row * stride + column] += marks[(row - 1) * stride + column];
            }
        }
        let mut touched = Vec::with_capacity(columns * rows);
        for row in 0..rows {
            touched.extend(
                marks[row * stride..row * stride + columns]
                    .iter()
                    .map(|&m| m > 0),
            );
        }
        Self {
            column,
            row,
            columns,
            rows,
            touched,
        }
    }

    fn is_touched(&self, column: usize, row: usize) -> bool {
        self.touched[row * self.columns + column]
    }

    /// Прямоугольники копирования пакета: прежние `clips`, обрезанные затронутыми ячейками.
    /// `None`, если прямоугольников больше [`MAX_PATH_COPY_SPRITES`].
    fn copy_sprites(
        &self,
        region: PathBatchRegion,
        clips: &[PathSprite],
        viewport: Viewport,
    ) -> Option<Vec<PathSprite>> {
        let origin = [viewport.origin.x.0, viewport.origin.y.0];
        // Ячейки (относительно первой ячейки пакета), которые пересекает отрезок [low, high).
        let cell_range = |low: f32, high: f32, start: u32, extent: u32, first: usize| {
            let low = low.max(start as f32);
            let high = high.min((start + extent) as f32);
            if high.partial_cmp(&low) != Some(std::cmp::Ordering::Greater) {
                return 0..0;
            }
            (low as u32 / PATH_COPY_CELL) as usize - first
                ..(high.ceil() as u32).div_ceil(PATH_COPY_CELL) as usize - first
        };
        let mut sprites = Vec::new();
        for clip in clips {
            let clip = clip.bounds;
            let x = clip.origin.x.0 - origin[0];
            let y = clip.origin.y.0 - origin[1];
            let columns = cell_range(
                x,
                x + clip.size.width.0,
                region.x,
                region.width,
                self.column,
            );
            let rows = cell_range(y, y + clip.size.height.0, region.y, region.height, self.row);
            for row in rows {
                let mut column = columns.start;
                while column < columns.end {
                    if !self.is_touched(column, row) {
                        column += 1;
                        continue;
                    }
                    let run_start = column;
                    while column < columns.end && self.is_touched(column, row) {
                        column += 1;
                    }
                    let cell = PATH_COPY_CELL as f32;
                    let run = Bounds::new(
                        point(
                            ScaledPixels(origin[0] + (self.column + run_start) as f32 * cell),
                            ScaledPixels(origin[1] + (self.row + row) as f32 * cell),
                        ),
                        size(
                            ScaledPixels((column - run_start) as f32 * cell),
                            ScaledPixels(cell),
                        ),
                    );
                    let bounds = run.intersect(&clip);
                    if bounds.is_empty() {
                        continue;
                    }
                    if sprites.len() == MAX_PATH_COPY_SPRITES {
                        return None;
                    }
                    sprites.push(PathSprite { bounds });
                }
            }
        }
        Some(sprites)
    }

    fn cells(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.rows).flat_map(move |row| {
            (0..self.columns)
                .filter(move |&column| self.is_touched(column, row))
                .map(move |column| (self.column + column, self.row + row))
        })
    }

    fn overlaps(&self, occupancy: &[bool], grid_columns: usize) -> bool {
        self.cells()
            .any(|(column, row)| occupancy[row * grid_columns + column])
    }

    fn occupy(&self, occupancy: &mut [bool], grid_columns: usize) {
        for (column, row) in self.cells() {
            occupancy[row * grid_columns + column] = true;
        }
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
                        .max(MAX_PATH_COPY_SPRITES)
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
    IntermediateOrigin = 5,
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
    IntermediateOrigin = 3,
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

#[cfg(test)]
mod path_region_tests {
    use super::*;
    use crate::{
        Corners, Edges, TransformationMatrix, hsla, linear_color_stop, linear_gradient, px,
    };

    const WIDTH: i32 = 160;
    const HEIGHT: i32 = 120;

    fn viewport() -> Viewport {
        Viewport {
            size: size(DevicePixels(WIDTH), DevicePixels(HEIGHT)),
            origin: point(ScaledPixels(0.), ScaledPixels(0.)),
        }
    }

    fn mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: Bounds::new(
                point(ScaledPixels(0.), ScaledPixels(0.)),
                size(ScaledPixels(WIDTH as f32), ScaledPixels(HEIGHT as f32)),
            ),
        }
    }

    fn curved_path(origin: (f32, f32), extent: f32, color: Background) -> Path<ScaledPixels> {
        let (x, y) = origin;
        let mut path = crate::Path::new(point(px(x), px(y)));
        path.line_to(point(px(x + extent), px(y + extent * 0.15)));
        path.curve_to(
            point(px(x + extent * 0.4), px(y + extent)),
            point(px(x + extent * 1.1), px(y + extent * 0.9)),
        );
        path.line_to(point(px(x), px(y)));
        let mut path = path.scale(1.);
        path.color = color;
        path.content_mask = mask();
        path
    }

    fn region_of(path: &Path<ScaledPixels>) -> Option<PathBatchRegion> {
        PathBatchRegion::of_paths(std::slice::from_ref(path), viewport())
    }

    #[test]
    fn region_rounds_clipped_bounds_out_with_pixel_margin() {
        let mut path = curved_path((10.25, 20.5), 30., hsla(0., 1., 0.5, 1.).into());
        path.bounds = Bounds::new(
            point(ScaledPixels(10.25), ScaledPixels(20.5)),
            size(ScaledPixels(10.5), ScaledPixels(5.)),
        );
        assert_eq!(
            region_of(&path),
            Some(PathBatchRegion {
                x: 9,
                y: 19,
                width: 13,
                height: 8,
            })
        );
    }

    #[test]
    fn region_is_clamped_to_target_and_skips_batches_outside_it() {
        let edge = curved_path((140., 100.), 60., hsla(0., 1., 0.5, 1.).into());
        let region = region_of(&edge).expect("path crosses the corner of the target");
        assert_eq!(
            (region.x + region.width, region.y + region.height),
            (WIDTH as u32, HEIGHT as u32)
        );

        let mut outside = curved_path((200., 10.), 20., hsla(0., 1., 0.5, 1.).into());
        outside.content_mask.bounds.size.width = ScaledPixels(1000.);
        assert_eq!(region_of(&outside), None);
    }

    #[test]
    fn region_is_relative_to_viewport_origin() {
        let path = curved_path((50., 40.), 10., hsla(0., 1., 0.5, 1.).into());
        let mut viewport = viewport();
        viewport.origin = point(ScaledPixels(20.), ScaledPixels(10.));
        let region = PathBatchRegion::of_paths(std::slice::from_ref(&path), viewport)
            .expect("path is inside the target");
        assert_eq!((region.x, region.y), (29, 29));
    }

    fn render(renderer: &mut MetalRenderer, scene: &Scene) -> Vec<u8> {
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(WIDTH as u64);
        descriptor.set_height(HEIGHT as u64);
        descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        let texture = renderer.device.new_texture(&descriptor);
        let mut instance_buffer = renderer
            .instance_buffer_pool
            .lock()
            .acquire(&renderer.device);
        let command_buffer = renderer.command_queue.new_command_buffer().to_owned();
        let mut offset = 0;
        renderer
            .encode_scene(
                scene,
                &texture,
                viewport(),
                &command_buffer,
                &mut instance_buffer,
                &mut offset,
                1.,
            )
            .expect("scene fits the instance buffer");
        let bytes_per_row = WIDTH as u64 * 4;
        let length = bytes_per_row * HEIGHT as u64;
        let readback = renderer
            .device
            .new_buffer(length, metal::MTLResourceOptions::StorageModeShared);
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            metal::MTLOrigin { x: 0, y: 0, z: 0 },
            metal::MTLSize {
                width: WIDTH as u64,
                height: HEIGHT as u64,
                depth: 1,
            },
            &readback,
            0,
            bytes_per_row,
            length,
            metal::MTLBlitOption::empty(),
        );
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        renderer
            .instance_buffer_pool
            .lock()
            .release(instance_buffer);
        unsafe { std::slice::from_raw_parts(readback.contents() as *const u8, length as usize) }
            .to_vec()
    }

    fn gradient() -> Background {
        linear_gradient(
            45.,
            linear_color_stop(hsla(0.6, 0.9, 0.5, 1.), 0.),
            linear_color_stop(hsla(0.1, 0.9, 0.6, 0.7), 1.),
        )
    }

    fn translucent_quad(x: f32, y: f32, width: f32, height: f32) -> Quad {
        Quad {
            order: 0,
            border_style: Default::default(),
            bounds: Bounds::new(
                point(ScaledPixels(x), ScaledPixels(y)),
                size(ScaledPixels(width), ScaledPixels(height)),
            ),
            content_mask: mask(),
            background: hsla(0.3, 0.5, 0.5, 0.6).into(),
            border_color: Default::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
            transformation: TransformationMatrix::unit(),
        }
    }

    /// Пакеты путей, разделённые квадом, чьи пути касаются общих ячеек: проходы раздельные.
    fn scene_with_crossing_batches() -> Scene {
        let mut scene = Scene::default();
        scene.insert_primitive(curved_path((12.5, 8.25), 50., gradient()));
        scene.insert_primitive(translucent_quad(30., 30., 70., 50.));
        scene.insert_primitive(curved_path(
            (60.75, 50.5),
            40.,
            hsla(0.9, 0.8, 0.5, 0.8).into(),
        ));
        scene.insert_primitive(curved_path(
            (130.3, 95.6),
            45.,
            hsla(0.5, 0.8, 0.4, 1.).into(),
        ));
        scene.insert_primitive(diagonal_stroke(hsla(0.15, 0.9, 0.5, 0.9).into()));
        scene.finish();
        scene
    }

    /// Пакеты путей, разделённые квадом, в непересекающихся ячейках: проход общий. Второй
    /// пакет с градиентом и выходом за край цели.
    fn scene_with_shared_pass() -> Scene {
        let mut scene = Scene::default();
        scene.insert_primitive(diagonal_stroke(hsla(0.15, 0.9, 0.5, 0.9).into()));
        scene.insert_primitive(translucent_quad(100., 80., 50., 30.));
        scene.insert_primitive(curved_path((12.5, 8.25), 40., gradient()));
        scene.insert_primitive(curved_path(
            (130.3, 95.6),
            45.,
            hsla(0.5, 0.8, 0.4, 1.).into(),
        ));
        scene.finish();
        scene
    }

    /// Тонкая линия через всю цель из мелких треугольников, как у связей холста.
    fn diagonal_stroke(color: Background) -> Path<ScaledPixels> {
        let mut path = crate::Path::new(point(px(0.), px(0.)));
        let steps = 40;
        for step in 0..steps {
            let t0 = step as f32 / steps as f32;
            let t1 = (step + 1) as f32 / steps as f32;
            let a = point(px(2. + t0 * 150.), px(115. - t0 * 110.));
            let b = point(px(2. + t1 * 150.), px(115. - t1 * 110.));
            let offset = point(px(1.3), px(1.1));
            path.push_triangle(
                (a, b, b + offset),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
            path.push_triangle(
                (a, b + offset, a + offset),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        let mut path = path.scale(1.);
        path.color = color;
        path.content_mask = mask();
        path
    }

    #[test]
    fn copy_sprites_cover_only_cells_touched_by_triangles() {
        let mut viewport = viewport();
        viewport.size = size(DevicePixels(1000), DevicePixels(400));
        let mut path = crate::Path::new(point(px(0.), px(0.)));
        let tri = (point(0., 1.), point(0., 1.), point(0., 1.));
        path.push_triangle(
            (
                point(px(10.), px(10.)),
                point(px(20.), px(10.)),
                point(px(10.), px(20.)),
            ),
            tri,
        );
        path.push_triangle(
            (
                point(px(900.), px(300.)),
                point(px(950.), px(300.)),
                point(px(900.), px(310.)),
            ),
            tri,
        );
        let mut path = path.scale(1.);
        path.content_mask = ContentMask {
            bounds: Bounds::new(
                point(ScaledPixels(0.), ScaledPixels(0.)),
                size(ScaledPixels(1000.), ScaledPixels(400.)),
            ),
        };
        let region = PathBatchRegion::of_paths(std::slice::from_ref(&path), viewport)
            .expect("paths are inside the target");
        let clips = [PathSprite {
            bounds: path.clipped_bounds(),
        }];
        let sprites = TouchedCells::of_paths(region, std::slice::from_ref(&path), viewport)
            .copy_sprites(region, &clips, viewport)
            .expect("few cells");
        let area: f32 = sprites
            .iter()
            .map(|sprite| sprite.bounds.size.width.0 * sprite.bounds.size.height.0)
            .sum();
        let region_area = (region.width * region.height) as f32;
        assert!(
            area < region_area * 0.05,
            "{area} of {region_area}: {sprites:?}"
        );
        for corner in [(10., 10.), (19.5, 19.5), (900., 300.), (949.5, 309.5)] {
            let inside = sprites.iter().filter(|sprite| {
                let b = sprite.bounds;
                corner.0 >= b.origin.x.0
                    && corner.0 < b.origin.x.0 + b.size.width.0
                    && corner.1 >= b.origin.y.0
                    && corner.1 < b.origin.y.0 + b.size.height.0
            });
            assert_eq!(inside.count(), 1, "{corner:?} must be copied exactly once");
        }
    }

    #[test]
    fn copy_sprites_give_up_when_runs_exceed_the_reserve() {
        let mut viewport = viewport();
        viewport.size = size(DevicePixels(4096), DevicePixels(4096));
        let mut path = crate::Path::new(point(px(0.), px(0.)));
        let tri = (point(0., 1.), point(0., 1.), point(0., 1.));
        // Шахматка: каждая вторая ячейка, отрезков больше предела.
        for row in 0..128 {
            for column in (row % 2..128).step_by(2) {
                let x = column as f32 * 32. + 8.;
                let y = row as f32 * 32. + 8.;
                path.push_triangle(
                    (
                        point(px(x), px(y)),
                        point(px(x + 4.), px(y)),
                        point(px(x), px(y + 4.)),
                    ),
                    tri,
                );
            }
        }
        let mut path = path.scale(1.);
        path.content_mask = ContentMask {
            bounds: Bounds::new(
                point(ScaledPixels(0.), ScaledPixels(0.)),
                size(ScaledPixels(4096.), ScaledPixels(4096.)),
            ),
        };
        let region = PathBatchRegion::of_paths(std::slice::from_ref(&path), viewport)
            .expect("paths are inside the target");
        let clips = [PathSprite {
            bounds: path.clipped_bounds(),
        }];
        assert!(
            TouchedCells::of_paths(region, std::slice::from_ref(&path), viewport)
                .copy_sprites(region, &clips, viewport)
                .is_none()
        );
    }

    #[test]
    fn path_passes_render_like_full_frame_pass_per_batch() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let pool = Arc::new(Mutex::new(InstanceBufferPool {
            buffer_size: 4 * 1024 * 1024,
            buffers: Vec::new(),
        }));
        let mut renderer = MetalRenderer::new(pool);
        for (name, scene, expected_passes) in [
            ("crossing", scene_with_crossing_batches(), 2),
            ("shared", scene_with_shared_pass(), 1),
        ] {
            let path_batches = scene
                .batches()
                .filter(|batch| matches!(batch, PrimitiveBatch::Paths(_)))
                .count();
            assert_eq!(path_batches, 2, "{name}: path batches");
            let passes = renderer.plan_path_passes(&scene, viewport()).passes.len();
            assert_eq!(passes, expected_passes, "{name}: path passes");

            renderer.reference_path_passes = true;
            let reference = render(&mut renderer, &scene);
            renderer.reference_path_passes = false;
            let candidate = render(&mut renderer, &scene);

            let painted = reference
                .chunks_exact(4)
                .filter(|pixel| pixel[..3] != [0, 0, 0])
                .count();
            assert!(
                painted > 1000,
                "{name}: paths painted only {painted} pixels"
            );
            let differing = reference
                .iter()
                .zip(&candidate)
                .filter(|(reference, candidate)| reference != candidate)
                .count();
            assert_eq!(
                differing, 0,
                "{name}: {differing} channels differ from full-frame passes per batch"
            );
        }
    }
}
