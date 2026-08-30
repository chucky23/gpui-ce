//! GPU-resident raster cache contracts.

use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicU32, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use crate::{Pixels, Point, point, px};

/// Default point at which least-recently-used tiles begin to be evicted.
pub const DEFAULT_RASTER_CACHE_SOFT_LIMIT_BYTES: usize = 224 * 1024 * 1024;

/// Default hard upper bound for GPU-resident raster tiles.
pub const DEFAULT_RASTER_CACHE_HARD_LIMIT_BYTES: usize = 256 * 1024 * 1024;

static NEXT_RASTER_CACHE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_RASTER_COMPOSITOR_ID: AtomicU64 = AtomicU64::new(1);

/// Camera transform consumed by a platform compositor independently of scene rebuilding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RasterCompositorTransform {
    /// Uniform Canvas scale.
    pub scale: f32,
    /// Canvas origin in window coordinates.
    pub translation: Point<Pixels>,
}

impl RasterCompositorTransform {
    /// Creates a finite transform with a positive scale.
    pub fn new(scale: f32, translation: Point<Pixels>) -> Option<Self> {
        (scale.is_finite()
            && scale > 0.
            && translation.x.0.is_finite()
            && translation.y.0.is_finite())
        .then_some(Self { scale, translation })
    }
}

impl Default for RasterCompositorTransform {
    fn default() -> Self {
        Self {
            scale: 1.,
            translation: point(px(0.), px(0.)),
        }
    }
}

#[derive(Debug)]
struct RasterCompositorTransformState {
    id: u64,
    generation: AtomicU64,
    scale_bits: AtomicU32,
    translation_x_bits: AtomicU32,
    translation_y_bits: AtomicU32,
    updates: Mutex<VecDeque<(u64, Instant)>>,
}

/// Thread-safe latest-value camera transform for late compositor latching.
///
/// A generation sequence protects readers from observing a mixture of two updates. The handle
/// carries no platform object and can therefore be updated directly from ordinary input handling.
#[derive(Clone, Debug)]
pub struct RasterCompositorTransformHandle(Arc<RasterCompositorTransformState>);

impl RasterCompositorTransformHandle {
    /// Creates a transform handle when the initial value is finite and has a positive scale.
    pub fn new(initial: RasterCompositorTransform) -> Option<Self> {
        RasterCompositorTransform::new(initial.scale, initial.translation).map(|initial| {
            Self(Arc::new(RasterCompositorTransformState {
                id: NEXT_RASTER_COMPOSITOR_ID.fetch_add(1, Ordering::Relaxed),
                generation: AtomicU64::new(0),
                scale_bits: AtomicU32::new(initial.scale.to_bits()),
                translation_x_bits: AtomicU32::new(initial.translation.x.0.to_bits()),
                translation_y_bits: AtomicU32::new(initial.translation.y.0.to_bits()),
                updates: Mutex::new(VecDeque::from([(0, Instant::now())])),
            }))
        })
    }

    /// Publishes one coherent transform. Invalid values leave the previous value unchanged.
    pub fn update(&self, transform: RasterCompositorTransform) -> bool {
        self.update_at(transform, Instant::now())
    }

    /// Publishes one coherent transform associated with its original input observation time.
    pub fn update_at(&self, transform: RasterCompositorTransform, updated_at: Instant) -> bool {
        let Some(transform) =
            RasterCompositorTransform::new(transform.scale, transform.translation)
        else {
            return false;
        };
        if self.snapshot().1 == transform {
            return false;
        }

        let generation = loop {
            let generation = self.0.generation.load(Ordering::SeqCst);
            if generation & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            if self
                .0
                .generation
                .compare_exchange(
                    generation,
                    generation.wrapping_add(1),
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                break generation;
            }
        };

        self.0
            .scale_bits
            .store(transform.scale.to_bits(), Ordering::SeqCst);
        self.0
            .translation_x_bits
            .store(transform.translation.x.0.to_bits(), Ordering::SeqCst);
        self.0
            .translation_y_bits
            .store(transform.translation.y.0.to_bits(), Ordering::SeqCst);
        let revision = generation.wrapping_add(2) / 2;
        let mut updates = self
            .0
            .updates
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        updates.push_back((revision, updated_at));
        while updates.len() > 128 {
            updates.pop_front();
        }
        drop(updates);
        // Publish the even generation only after its timestamp is available. Otherwise a
        // display-link reader could latch the transform first and fabricate a near-zero sample.
        self.0
            .generation
            .store(generation.wrapping_add(2), Ordering::SeqCst);
        true
    }

    /// Reads a coherent latest transform and its monotonic revision.
    pub fn snapshot(&self) -> (u64, RasterCompositorTransform) {
        loop {
            let before = self.0.generation.load(Ordering::SeqCst);
            if before & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let transform = RasterCompositorTransform {
                scale: f32::from_bits(self.0.scale_bits.load(Ordering::SeqCst)),
                translation: point(
                    px(f32::from_bits(
                        self.0.translation_x_bits.load(Ordering::SeqCst),
                    )),
                    px(f32::from_bits(
                        self.0.translation_y_bits.load(Ordering::SeqCst),
                    )),
                ),
            };
            let after = self.0.generation.load(Ordering::SeqCst);
            if before == after {
                return (after / 2, transform);
            }
        }
    }

    /// Stable namespace used to associate presentation samples with this handle.
    pub fn id(&self) -> u64 {
        self.0.id
    }

    pub(crate) fn updated_at(&self, revision: u64) -> Option<Instant> {
        self.0
            .updates
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .rev()
            .find_map(|(candidate, updated_at)| (*candidate == revision).then_some(*updated_at))
    }
}

impl PartialEq for RasterCompositorTransformHandle {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

impl Eq for RasterCompositorTransformHandle {}

/// A camera revision observed on the Core Animation presentation layer.
#[derive(Clone, Copy, Debug)]
pub struct RasterCompositorPresentationSample {
    /// Compositor namespace that produced the sample.
    pub compositor_id: u64,
    /// Transform revision visible in the presentation layer.
    pub revision: u64,
    /// Time at which input published the revision.
    pub updated_at: Instant,
    /// Display-link time at which the presentation layer exposed the revision.
    pub presented_at: Instant,
}

impl RasterCompositorPresentationSample {
    /// Latency from publishing the camera transform to observing it in presented layer state.
    pub fn input_to_presentation(self) -> Duration {
        self.presented_at.saturating_duration_since(self.updated_at)
    }
}

/// Memory limits for one raster cache namespace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RasterCacheConfig {
    soft_limit_bytes: usize,
    hard_limit_bytes: usize,
}

impl RasterCacheConfig {
    /// Creates a configuration when the limits are non-zero and ordered.
    pub fn new(soft_limit_bytes: usize, hard_limit_bytes: usize) -> Option<Self> {
        (soft_limit_bytes > 0 && soft_limit_bytes <= hard_limit_bytes).then_some(Self {
            soft_limit_bytes,
            hard_limit_bytes,
        })
    }

    /// Returns the point at which offscreen tiles should be evicted.
    pub fn soft_limit_bytes(self) -> usize {
        self.soft_limit_bytes
    }

    /// Returns the maximum number of bytes retained by this cache.
    pub fn hard_limit_bytes(self) -> usize {
        self.hard_limit_bytes
    }
}

impl Default for RasterCacheConfig {
    fn default() -> Self {
        Self {
            soft_limit_bytes: DEFAULT_RASTER_CACHE_SOFT_LIMIT_BYTES,
            hard_limit_bytes: DEFAULT_RASTER_CACHE_HARD_LIMIT_BYTES,
        }
    }
}

#[derive(Debug)]
pub(crate) struct RasterCacheIdentity {
    id: u64,
    config: RasterCacheConfig,
}

/// Opaque owner and namespace for a group of GPU-resident raster tiles.
///
/// Clones refer to the same namespace. Platform resources live no longer than
/// the window and may also be released explicitly through [`crate::Window`].
#[derive(Clone, Debug)]
pub struct RasterCacheHandle(Arc<RasterCacheIdentity>);

impl RasterCacheHandle {
    /// Allocates a new cache namespace with the supplied memory limits.
    pub fn new(config: RasterCacheConfig) -> Self {
        let id = NEXT_RASTER_CACHE_ID.fetch_add(1, Ordering::Relaxed);
        Self(Arc::new(RasterCacheIdentity { id, config }))
    }

    pub(crate) fn id(&self) -> u64 {
        self.0.id
    }

    pub(crate) fn config(&self) -> RasterCacheConfig {
        self.0.config
    }

    pub(crate) fn weak_identity(&self) -> Weak<RasterCacheIdentity> {
        Arc::downgrade(&self.0)
    }

    pub(crate) fn tile_hit(
        &self,
        key: RasterTileKey,
        revision: RasterTileRevision,
        gutter: u32,
    ) -> RasterTileHit {
        RasterTileHit {
            cache: self.clone(),
            key,
            revision,
            gutter,
        }
    }

    pub(crate) fn tile_miss(
        &self,
        key: RasterTileKey,
        revision: RasterTileRevision,
    ) -> RasterTileMiss {
        RasterTileMiss {
            cache: self.clone(),
            key,
            revision,
        }
    }
}

impl PartialEq for RasterCacheHandle {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

impl Eq for RasterCacheHandle {}

/// Application-defined identity of one tile inside a cache namespace.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RasterTileKey(u64);

impl RasterTileKey {
    /// Creates a tile key from an application-owned stable value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) const fn value(self) -> u64 {
        self.0
    }
}

/// Monotonic content revision for one tile key.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RasterTileRevision(u64);

impl RasterTileRevision {
    /// Creates a content revision.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) const fn value(self) -> u64 {
        self.0
    }
}

/// A renderer-confirmed tile that can be composed in the current frame.
#[derive(Clone, Debug)]
pub struct RasterTileHit {
    pub(crate) cache: RasterCacheHandle,
    pub(crate) key: RasterTileKey,
    pub(crate) revision: RasterTileRevision,
    pub(crate) gutter: u32,
}

/// Permission to populate a missing or stale tile.
#[derive(Clone, Debug)]
pub struct RasterTileMiss {
    pub(crate) cache: RasterCacheHandle,
    pub(crate) key: RasterTileKey,
    pub(crate) revision: RasterTileRevision,
}

/// Result of looking up a tile for the current frame.
#[derive(Clone, Debug)]
pub enum RasterTileLookup {
    /// The exact revision is resident and pinned for this frame.
    Hit(RasterTileHit),
    /// The tile must be populated before it can be reused.
    Miss(RasterTileMiss),
    /// The active platform renderer has no raster cache implementation.
    Unsupported,
}

/// Current resource counters for one cache namespace.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RasterCacheStats {
    /// Bytes currently retained by tile textures.
    pub resident_bytes: usize,
    /// Number of resident tile textures.
    pub resident_tiles: usize,
    /// Number of tiles evicted since this namespace was created.
    pub evicted_tiles: u64,
    /// Number of exact-revision lookup hits.
    pub hits: u64,
    /// Number of missing or stale lookups.
    pub misses: u64,
    /// Number of completed diagnostic comparisons between detailed and tiled output.
    pub comparison_samples: u64,
    /// Lowest observed structural similarity, scaled by one billion.
    pub comparison_min_ssim_ppb: u32,
    /// Worst observed 99th-percentile absolute BGRA channel error.
    pub comparison_p99_channel_error: u8,
    /// Largest observed absolute BGRA channel error.
    pub comparison_max_channel_error: u8,
}

/// Timing reported for a frame after its drawable was actually presented.
#[derive(Clone, Copy, Debug)]
pub struct FramePresentationSample {
    /// Renderer-assigned monotonic frame identifier.
    pub frame_id: u64,
    /// Identifier assigned by `CAMetalDrawable`.
    pub drawable_id: u64,
    /// Host timestamp reported by the drawable presentation callback.
    pub presented_time_seconds: f64,
    /// CPU instant at which the command buffer was submitted.
    pub submitted_at: Instant,
    /// CPU-clock projection of the drawable's actual host presentation timestamp.
    pub presented_at: Instant,
    /// CPU instant at which Metal reported presentation.
    pub observed_at: Instant,
    /// GPU execution duration reported by the command buffer, when available.
    pub gpu_duration: Option<Duration>,
}

impl FramePresentationSample {
    /// Wall-clock latency from command submission until the drawable was presented.
    pub fn submission_to_presentation(self) -> Duration {
        self.presented_at
            .saturating_duration_since(self.submitted_at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn cache_config_rejects_invalid_limits() {
        assert_eq!(RasterCacheConfig::new(0, 1), None);
        assert_eq!(RasterCacheConfig::new(2, 1), None);
        assert_eq!(
            RasterCacheConfig::new(1, 2),
            Some(RasterCacheConfig {
                soft_limit_bytes: 1,
                hard_limit_bytes: 2,
            })
        );
    }

    #[test]
    fn cache_handles_are_unique_and_clones_share_identity() {
        let first = RasterCacheHandle::new(RasterCacheConfig::default());
        let first_clone = first.clone();
        let second = RasterCacheHandle::new(RasterCacheConfig::default());

        assert_eq!(first, first_clone);
        assert_ne!(first, second);
    }

    #[test]
    fn compositor_transform_rejects_invalid_updates() {
        let handle = RasterCompositorTransformHandle::new(RasterCompositorTransform::default())
            .expect("default transform is valid");
        assert!(!handle.update(RasterCompositorTransform {
            scale: 0.,
            translation: point(px(10.), px(20.)),
        }));
        assert_eq!(handle.snapshot(), (0, RasterCompositorTransform::default()));
    }

    #[test]
    fn compositor_transform_snapshot_never_tears_between_writers() {
        let handle = RasterCompositorTransformHandle::new(RasterCompositorTransform {
            scale: 0.5,
            translation: point(px(1.), px(1.5)),
        })
        .expect("initial transform is valid");
        let writer = handle.clone();
        let thread = std::thread::spawn(move || {
            for value in 1..=10_000_u32 {
                let value = value as f32;
                assert!(writer.update(RasterCompositorTransform {
                    scale: value,
                    translation: point(px(value * 2.), px(value * 3.)),
                }));
            }
        });

        while !thread.is_finished() {
            let (_, transform) = handle.snapshot();
            assert_eq!(transform.translation.x.0, transform.scale * 2.);
            assert_eq!(transform.translation.y.0, transform.scale * 3.);
        }
        thread.join().expect("writer must finish");
        let (revision, transform) = handle.snapshot();
        assert_eq!(revision, 10_000);
        assert_eq!(transform.scale, 10_000.);
    }

    #[test]
    fn compositor_transform_publishes_timestamp_with_revision() {
        let handle = RasterCompositorTransformHandle::new(RasterCompositorTransform::default())
            .expect("default transform is valid");
        assert!(handle.update(RasterCompositorTransform {
            scale: 2.,
            translation: point(px(20.), px(30.)),
        }));

        let (revision, _) = handle.snapshot();
        assert_eq!(revision, 1);
        assert!(handle.updated_at(revision).is_some());
    }

    #[test]
    fn compositor_transform_preserves_input_time_and_skips_identical_updates() {
        let handle = RasterCompositorTransformHandle::new(RasterCompositorTransform::default())
            .expect("default transform is valid");
        let observed_at = Instant::now() - Duration::from_millis(3);
        let transform = RasterCompositorTransform {
            scale: 2.,
            translation: point(px(20.), px(30.)),
        };
        assert!(handle.update_at(transform, observed_at));
        assert!(!handle.update_at(transform, Instant::now()));

        let (revision, _) = handle.snapshot();
        assert_eq!(revision, 1);
        assert_eq!(handle.updated_at(revision), Some(observed_at));
    }

    #[test]
    fn weak_cache_identity_expires_after_last_handle_is_dropped() {
        let handle = RasterCacheHandle::new(RasterCacheConfig::default());
        let weak = handle.weak_identity();
        let clone = handle.clone();
        drop(handle);
        assert!(weak.upgrade().is_some());
        drop(clone);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn presentation_sample_reports_submission_latency() {
        let submitted_at = Instant::now();
        let observed_at = submitted_at + Duration::from_millis(7);
        let sample = FramePresentationSample {
            frame_id: 1,
            drawable_id: 2,
            presented_time_seconds: 3.,
            submitted_at,
            presented_at: observed_at,
            observed_at,
            gpu_duration: Some(Duration::from_millis(4)),
        };

        assert_eq!(
            sample.submission_to_presentation(),
            Duration::from_millis(7)
        );
    }
}
