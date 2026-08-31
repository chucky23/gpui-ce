// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla,
    Pixels, Point, Radians, RasterCacheConfig, RasterCacheHandle, RasterTileKey,
    RasterTileRevision, ScaledPixels, Size, bounds_tree::BoundsTree, point,
};
use std::{
    fmt::Debug,
    iter::Peekable,
    ops::{Add, Range, Sub},
    slice,
};

#[allow(non_camel_case_types, unused)]
pub(crate) type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

pub(crate) type DrawOrder = u32;

#[derive(Default)]
pub(crate) struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<DrawOrder>,
    pub(crate) shadows: Vec<Shadow>,
    pub(crate) quads: Vec<Quad>,
    pub(crate) paths: Vec<Path<ScaledPixels>>,
    pub(crate) underlines: Vec<Underline>,
    pub(crate) monochrome_sprites: Vec<MonochromeSprite>,
    pub(crate) polychrome_sprites: Vec<PolychromeSprite>,
    pub(crate) surfaces: Vec<PaintSurface>,
    pub(crate) raster_tiles: Vec<RasterTile>,
    pub(crate) raster_tile_updates: Vec<RasterTileUpdate>,
    pub(crate) raster_tile_update_batches: Vec<RasterTileUpdateBatch>,
    pub(crate) raster_compositor_surfaces: Vec<RasterCompositorSurface>,
}

impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.raster_tiles.clear();
        self.raster_tile_updates.clear();
        self.raster_tile_update_batches.clear();
        self.raster_compositor_surfaces.clear();
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let mut primitive = primitive.into();
        let clipped_bounds = primitive
            .bounds()
            .intersect(&primitive.content_mask().bounds);

        if clipped_bounds.is_empty() {
            return;
        }

        let order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds));
        match &mut primitive {
            Primitive::Shadow(shadow) => {
                shadow.order = order;
                self.shadows.push(shadow.clone());
            }
            Primitive::Quad(quad) => {
                quad.order = order;
                self.quads.push(quad.clone());
            }
            Primitive::Path(path) => {
                path.order = order;
                path.id = PathId(self.paths.len());
                self.paths.push(path.clone());
            }
            Primitive::Underline(underline) => {
                underline.order = order;
                self.underlines.push(underline.clone());
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.order = order;
                self.monochrome_sprites.push(sprite.clone());
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.order = order;
                self.polychrome_sprites.push(sprite.clone());
            }
            Primitive::Surface(surface) => {
                surface.order = order;
                self.surfaces.push(surface.clone());
            }
            Primitive::RasterTile(tile) => {
                tile.order = order;
                self.raster_tiles.push(tile.clone());
            }
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
            }
        }
    }

    pub(crate) fn clone_vector_paint(&self, range: Range<usize>) -> Option<Vec<PaintOperation>> {
        self.paint_operations[range]
            .iter()
            .map(|operation| match operation {
                PaintOperation::Primitive(Primitive::Surface(_))
                | PaintOperation::Primitive(Primitive::RasterTile(_)) => None,
                _ => Some(operation.clone()),
            })
            .collect()
    }

    pub(crate) fn replay_vector_translation(
        &mut self,
        operations: &[PaintOperation],
        translation: Point<ScaledPixels>,
    ) {
        for operation in operations {
            match operation {
                PaintOperation::Primitive(primitive) => {
                    self.insert_primitive(primitive.translated(translation));
                }
                PaintOperation::StartLayer(bounds) => {
                    let mut bounds = *bounds;
                    bounds.origin += translation;
                    self.push_layer(bounds);
                }
                PaintOperation::EndLayer => self.pop_layer(),
            }
        }
    }

    pub(crate) fn clone_text_paint(&self, range: Range<usize>) -> Option<Vec<PaintOperation>> {
        self.paint_operations[range]
            .iter()
            .map(|operation| match operation {
                PaintOperation::Primitive(Primitive::MonochromeSprite(_))
                | PaintOperation::Primitive(Primitive::PolychromeSprite(_))
                | PaintOperation::StartLayer(_)
                | PaintOperation::EndLayer => Some(operation.clone()),
                PaintOperation::Primitive(_) => None,
            })
            .collect()
    }

    pub(crate) fn replay_cached_text(
        &mut self,
        operations: &[PaintOperation],
        origin_delta: Point<ScaledPixels>,
        content_mask: ContentMask<ScaledPixels>,
        transformation: TransformationMatrix,
    ) {
        for operation in operations {
            match operation {
                PaintOperation::StartLayer(template_bounds) => {
                    let mut bounds = *template_bounds;
                    bounds.origin += origin_delta;
                    let clipped_bounds = bounds.intersect(&content_mask.bounds);
                    if clipped_bounds.is_empty() {
                        self.layer_stack
                            .push(self.layer_stack.last().copied().unwrap_or(DrawOrder::MAX));
                    } else {
                        let order = self.primitive_bounds.insert(clipped_bounds);
                        self.layer_stack.push(order);
                    }
                }
                PaintOperation::EndLayer => {
                    self.layer_stack.pop();
                }
                PaintOperation::Primitive(Primitive::MonochromeSprite(template)) => {
                    let mut sprite = template.clone();
                    sprite.bounds.origin += origin_delta;
                    sprite.content_mask = content_mask.clone();
                    sprite.transformation = transformation;
                    sprite.order = self
                        .layer_stack
                        .last()
                        .copied()
                        .unwrap_or_else(|| self.primitive_bounds.insert(sprite.bounds));
                    self.monochrome_sprites.push(sprite);
                }
                PaintOperation::Primitive(Primitive::PolychromeSprite(template)) => {
                    let mut sprite = template.clone();
                    sprite.bounds.origin += origin_delta;
                    sprite.content_mask = content_mask.clone();
                    sprite.transformation = transformation;
                    sprite.order = self
                        .layer_stack
                        .last()
                        .copied()
                        .unwrap_or_else(|| self.primitive_bounds.insert(sprite.bounds));
                    self.polychrome_sprites.push(sprite);
                }
                PaintOperation::Primitive(_) => {
                    unreachable!("cached text only contains layers and sprite primitives")
                }
            }
        }
    }

    pub fn finish(&mut self) {
        self.shadows.sort_by_key(|shadow| shadow.order);
        self.quads.sort_by_key(|quad| quad.order);
        self.paths.sort_by_key(|path| path.order);
        self.underlines.sort_by_key(|underline| underline.order);
        self.monochrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.polychrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.surfaces.sort_by_key(|surface| surface.order);
        self.raster_tiles.sort_by_key(|tile| tile.order);
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub(crate) fn batches(&self) -> impl Iterator<Item = PrimitiveBatch<'_>> {
        BatchIterator {
            shadows: &self.shadows,
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads: &self.quads,
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            paths: &self.paths,
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines: &self.underlines,
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites: &self.monochrome_sprites,
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            polychrome_sprites: &self.polychrome_sprites,
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces: &self.surfaces,
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
            raster_tiles: &self.raster_tiles,
            raster_tiles_start: 0,
            raster_tiles_iter: self.raster_tiles.iter().peekable(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    Shadow,
    #[default]
    Quad,
    Path,
    Underline,
    MonochromeSprite,
    PolychromeSprite,
    Surface,
    RasterTile,
}

#[derive(Clone)]
pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

#[derive(Clone)]
pub(crate) enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
    RasterTile(RasterTile),
}

impl Primitive {
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
            Primitive::RasterTile(tile) => &tile.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
            Primitive::RasterTile(tile) => &tile.content_mask,
        }
    }

    fn translated(&self, translation: Point<ScaledPixels>) -> Self {
        let translate_mask = |mask: &ContentMask<ScaledPixels>| {
            let mut mask = mask.clone();
            mask.bounds.origin += translation;
            mask
        };
        let conjugate = |matrix: TransformationMatrix| {
            let inverse_translation = Point::new(
                ScaledPixels(-translation.x.0),
                ScaledPixels(-translation.y.0),
            );
            TransformationMatrix::unit()
                .translate(translation)
                .compose(matrix)
                .translate(inverse_translation)
        };

        match self {
            Primitive::Shadow(template) => {
                let mut shadow = template.clone();
                shadow.bounds.origin += translation;
                shadow.content_mask = translate_mask(&shadow.content_mask);
                shadow.transformation = conjugate(shadow.transformation);
                Primitive::Shadow(shadow)
            }
            Primitive::Quad(template) => {
                let mut quad = template.clone();
                quad.bounds.origin += translation;
                quad.content_mask = translate_mask(&quad.content_mask);
                quad.transformation = conjugate(quad.transformation);
                Primitive::Quad(quad)
            }
            Primitive::Path(template) => {
                let mut path = template.clone();
                path.bounds.origin += translation;
                path.content_mask = translate_mask(&path.content_mask);
                path.start += translation;
                path.current += translation;
                for vertex in &mut path.vertices {
                    vertex.xy_position += translation;
                    vertex.content_mask = translate_mask(&vertex.content_mask);
                }
                Primitive::Path(path)
            }
            Primitive::Underline(template) => {
                let mut underline = template.clone();
                underline.bounds.origin += translation;
                underline.content_mask = translate_mask(&underline.content_mask);
                Primitive::Underline(underline)
            }
            Primitive::MonochromeSprite(template) => {
                let mut sprite = template.clone();
                sprite.bounds.origin += translation;
                sprite.content_mask = translate_mask(&sprite.content_mask);
                sprite.transformation = conjugate(sprite.transformation);
                Primitive::MonochromeSprite(sprite)
            }
            Primitive::PolychromeSprite(template) => {
                let mut sprite = template.clone();
                sprite.bounds.origin += translation;
                sprite.content_mask = translate_mask(&sprite.content_mask);
                sprite.transformation = conjugate(sprite.transformation);
                Primitive::PolychromeSprite(sprite)
            }
            Primitive::Surface(template) => {
                let mut surface = template.clone();
                surface.bounds.origin += translation;
                surface.content_mask = translate_mask(&surface.content_mask);
                Primitive::Surface(surface)
            }
            Primitive::RasterTile(template) => {
                let mut tile = template.clone();
                tile.bounds.origin += translation;
                tile.content_mask = translate_mask(&tile.content_mask);
                Primitive::RasterTile(tile)
            }
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows: &'a [Shadow],
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads: &'a [Quad],
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    paths: &'a [Path<ScaledPixels>],
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines: &'a [Underline],
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites: &'a [MonochromeSprite],
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    polychrome_sprites: &'a [PolychromeSprite],
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces: &'a [PaintSurface],
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
    raster_tiles: &'a [RasterTile],
    raster_tiles_start: usize,
    raster_tiles_iter: Peekable<slice::Iter<'a, RasterTile>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
            (
                self.raster_tiles_iter.peek().map(|t| t.order),
                PrimitiveKind::RasterTile,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(
                    &self.shadows[shadows_start..shadows_end],
                ))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(&self.quads[quads_start..quads_end]))
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(&self.paths[paths_start..paths_end]))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(
                    &self.underlines[underlines_start..underlines_end],
                ))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    sprites: &self.monochrome_sprites[sprites_start..sprites_end],
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = self.polychrome_sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    sprites: &self.polychrome_sprites[sprites_start..sprites_end],
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(
                    &self.surfaces[surfaces_start..surfaces_end],
                ))
            }
            PrimitiveKind::RasterTile => {
                let raster_tiles_start = self.raster_tiles_start;
                let mut raster_tiles_end = raster_tiles_start + 1;
                self.raster_tiles_iter.next();
                while self
                    .raster_tiles_iter
                    .next_if(|tile| (tile.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    raster_tiles_end += 1;
                }
                self.raster_tiles_start = raster_tiles_end;
                Some(PrimitiveBatch::RasterTiles(
                    &self.raster_tiles[raster_tiles_start..raster_tiles_end],
                ))
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveBatch<'a> {
    Shadows(&'a [Shadow]),
    Quads(&'a [Quad]),
    Paths(&'a [Path<ScaledPixels>]),
    Underlines(&'a [Underline]),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        sprites: &'a [MonochromeSprite],
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        sprites: &'a [PolychromeSprite],
    },
    Surfaces(&'a [PaintSurface]),
    RasterTiles(&'a [RasterTile]),
}

#[derive(Debug, Clone)]
pub(crate) struct RasterTile {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub cache_id: u64,
    pub key: u64,
    pub revision: u64,
    pub gutter: u32,
}

impl From<RasterTile> for Primitive {
    fn from(tile: RasterTile) -> Self {
        Primitive::RasterTile(tile)
    }
}

pub(crate) struct RasterTileUpdate {
    pub cache: RasterCacheHandle,
    pub config: RasterCacheConfig,
    pub key: RasterTileKey,
    pub revision: RasterTileRevision,
    pub texture_size: Size<DevicePixels>,
    pub gutter: DevicePixels,
    pub source_bounds: Bounds<ScaledPixels>,
    pub scene: Scene,
}

pub(crate) struct RasterTileUpdateBatch {
    pub cache: RasterCacheHandle,
    pub config: RasterCacheConfig,
    pub texture_size: Size<DevicePixels>,
    pub gutter: DevicePixels,
    pub targets: Vec<RasterTileUpdateTarget>,
    pub scene: Scene,
    pub deferred: bool,
    pub verify: bool,
}

pub(crate) struct RasterTileUpdateTarget {
    pub key: RasterTileKey,
    pub revision: RasterTileRevision,
    pub source_bounds: Bounds<ScaledPixels>,
}

pub(crate) struct RasterCompositorSurface {
    pub handle: crate::RasterCompositorTransformHandle,
    pub captured_transform: crate::RasterCompositorTransform,
    pub clip_bounds: Bounds<ScaledPixels>,
    pub raster_bounds: Bounds<ScaledPixels>,
    pub scene: Scene,
}

#[derive(Default, Debug, Clone)]
#[repr(C)]
pub(crate) struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
    pub transformation: TransformationMatrix,
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub(crate) struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: u32,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub(crate) struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub transformation: TransformationMatrix,
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(mut self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct MonochromeSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub grayscale: bool,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    #[cfg(target_os = "macos")]
    pub image_buffer: core_video::pixel_buffer::CVPixelBuffer,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PathId(pub(crate) usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub(crate) id: PathId,
    pub(crate) order: DrawOrder,
    pub(crate) bounds: Bounds<P>,
    pub(crate) content_mask: ContentMask<P>,
    pub(crate) vertices: Vec<PathVertex<P>>,
    pub(crate) color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    pub(crate) fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub(crate) xy_position: Point<P>,
    pub(crate) st_position: Point<f32>,
    pub(crate) content_mask: ContentMask<P>,
}

impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::px;

    fn scaled_point(x: f32, y: f32) -> Point<ScaledPixels> {
        Point::new(ScaledPixels(x), ScaledPixels(y))
    }

    fn scaled_bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            scaled_point(x, y),
            Size::new(ScaledPixels(width), ScaledPixels(height)),
        )
    }

    #[test]
    fn vector_translation_moves_bounds_masks_and_existing_transformation_together() {
        let bounds = scaled_bounds(10., 20., 30., 40.);
        let content_mask = ContentMask {
            bounds: scaled_bounds(0., 0., 100., 100.),
        };
        let transformation = TransformationMatrix::unit()
            .translate(scaled_point(10., 20.))
            .scale(Size::new(2., 2.))
            .translate(scaled_point(-10., -20.));
        let mut original = Scene::default();
        original.insert_primitive(Quad {
            bounds,
            content_mask,
            transformation,
            ..Default::default()
        });
        let Some(operations) = original.clone_vector_paint(0..original.len()) else {
            panic!("quad must be supported");
        };

        let translation = scaled_point(7., -3.);
        let mut replayed = Scene::default();
        replayed.replay_vector_translation(&operations, translation);
        let translated = &replayed.quads[0];

        assert_eq!(translated.bounds.origin, scaled_point(17., 17.));
        assert_eq!(translated.content_mask.bounds.origin, scaled_point(7., -3.));
        let old_output = transformation.apply(point(px(12.), px(22.)));
        let new_output = translated.transformation.apply(point(px(19.), px(19.)));
        assert_eq!(new_output, old_output + point(px(7.), px(-3.)));
    }

    #[test]
    fn vector_translation_moves_every_path_coordinate() {
        let mask = ContentMask {
            bounds: scaled_bounds(0., 0., 100., 100.),
        };
        let path = Path {
            id: PathId(0),
            order: 0,
            bounds: scaled_bounds(2., 3., 8., 9.),
            content_mask: mask.clone(),
            vertices: vec![PathVertex {
                xy_position: scaled_point(4., 5.),
                st_position: point(0., 1.),
                content_mask: mask,
            }],
            color: Background::default(),
            start: scaled_point(2., 3.),
            current: scaled_point(10., 12.),
            contour_count: 1,
        };
        let mut original = Scene::default();
        original.insert_primitive(path);
        let Some(operations) = original.clone_vector_paint(0..original.len()) else {
            panic!("path must be supported");
        };

        let mut replayed = Scene::default();
        replayed.replay_vector_translation(&operations, scaled_point(-2., 6.));
        let translated = &replayed.paths[0];

        assert_eq!(translated.bounds.origin, scaled_point(0., 9.));
        assert_eq!(translated.start, scaled_point(0., 9.));
        assert_eq!(translated.current, scaled_point(8., 18.));
        assert_eq!(translated.vertices[0].xy_position, scaled_point(2., 11.));
        assert_eq!(
            translated.vertices[0].content_mask.bounds.origin,
            scaled_point(-2., 6.)
        );
    }

    #[test]
    fn vector_capture_rejects_raster_tiles_atomically() {
        let mut scene = Scene::default();
        scene.insert_primitive(RasterTile {
            order: 0,
            bounds: scaled_bounds(0., 0., 10., 10.),
            content_mask: ContentMask {
                bounds: scaled_bounds(0., 0., 10., 10.),
            },
            cache_id: 1,
            key: 2,
            revision: 3,
            gutter: 0,
        });

        assert!(scene.clone_vector_paint(0..scene.len()).is_none());
    }
}
