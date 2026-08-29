//! GPU-resident raster cache contracts.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

/// Default point at which least-recently-used tiles begin to be evicted.
pub const DEFAULT_RASTER_CACHE_SOFT_LIMIT_BYTES: usize = 224 * 1024 * 1024;

/// Default hard upper bound for GPU-resident raster tiles.
pub const DEFAULT_RASTER_CACHE_HARD_LIMIT_BYTES: usize = 256 * 1024 * 1024;

static NEXT_RASTER_CACHE_ID: AtomicU64 = AtomicU64::new(1);

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
struct RasterCacheIdentity {
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
    pub(crate) texture_width: u32,
    pub(crate) texture_height: u32,
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
