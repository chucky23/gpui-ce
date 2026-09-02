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
    #[cfg(feature = "frame-trace")]
    pub(crate) frame_trace_logical_frame_id: u64,
    #[cfg(feature = "frame-trace")]
    pub(crate) frame_trace_input_sequence_id: u64,
    #[cfg(feature = "frame-trace")]
    pub(crate) frame_trace_presentation_token: Option<crate::frame_trace::PresentationToken>,
    #[cfg(feature = "frame-trace")]
    pub(crate) frame_trace_scene_build_tick: Option<crate::frame_trace::FrameTraceDisplayTick>,
    #[cfg(feature = "frame-trace")]
    pub(crate) frame_trace_presentation_tick: Option<crate::frame_trace::FrameTraceDisplayTick>,
    #[cfg(feature = "frame-trace")]
    pub(crate) frame_trace_gpui_window_frame_id: u64,
    #[cfg(feature = "frame-trace")]
    frame_trace_summary: crate::frame_trace::FrameTraceSceneSummary,
    #[cfg(feature = "frame-trace")]
    frame_trace_diagnostic_hold_ticks: u8,
    #[cfg(feature = "frame-trace")]
    frame_trace_last_held_tick_sequence: u64,
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
        #[cfg(feature = "frame-trace")]
        {
            self.frame_trace_logical_frame_id = 0;
            self.frame_trace_input_sequence_id = 0;
            self.frame_trace_presentation_token = None;
            self.frame_trace_scene_build_tick = None;
            self.frame_trace_presentation_tick = None;
            self.frame_trace_gpui_window_frame_id = 0;
            self.frame_trace_summary = Default::default();
            self.frame_trace_diagnostic_hold_ticks = 0;
            self.frame_trace_last_held_tick_sequence = 0;
        }
    }

    #[cfg(feature = "frame-trace")]
    pub(crate) fn set_frame_trace_correlation(
        &mut self,
        logical_frame_id: u64,
        input_sequence_id: u64,
        scene_build_tick: Option<crate::frame_trace::FrameTraceDisplayTick>,
    ) {
        self.frame_trace_logical_frame_id = logical_frame_id;
        self.frame_trace_input_sequence_id = input_sequence_id;
        self.frame_trace_scene_build_tick = scene_build_tick;
    }

    #[cfg(feature = "frame-trace")]
    pub(crate) fn set_frame_trace_presentation_token(
        &mut self,
        token: crate::frame_trace::PresentationToken,
    ) {
        self.frame_trace_presentation_token = Some(token);
    }

    #[cfg(feature = "frame-trace")]
    pub(crate) fn set_frame_trace_diagnostic_hold_ticks(&mut self, hold_ticks: u8) {
        self.frame_trace_diagnostic_hold_ticks = hold_ticks;
        self.frame_trace_last_held_tick_sequence = 0;
    }

    #[cfg(feature = "frame-trace")]
    pub(crate) fn set_frame_trace_presentation_attempt(
        &mut self,
        gpui_window_frame_id: u64,
        tick: Option<crate::frame_trace::FrameTraceDisplayTick>,
    ) {
        self.frame_trace_gpui_window_frame_id = gpui_window_frame_id;
        self.frame_trace_presentation_tick = tick;
    }

    #[cfg(feature = "frame-trace")]
    pub(crate) fn take_frame_trace_diagnostic_hold(&mut self) -> bool {
        if self.frame_trace_diagnostic_hold_ticks == 0 {
            return false;
        }
        if let Some(tick) = self.frame_trace_presentation_tick
            && tick.sequence != 0
            && tick.coalesced_count == 1
            && tick.sequence != self.frame_trace_last_held_tick_sequence
        {
            self.frame_trace_last_held_tick_sequence = tick.sequence;
            self.frame_trace_diagnostic_hold_ticks -= 1;
        }
        true
    }

    #[cfg(feature = "frame-trace")]
    pub(crate) fn populate_frame_trace_event(
        &self,
        event: &mut crate::frame_trace::FrameTraceEvent,
    ) {
        event.logical_frame_id = self.frame_trace_logical_frame_id;
        event.input_sequence_id = self.frame_trace_input_sequence_id;
        event.gpui_window_frame_id = self.frame_trace_gpui_window_frame_id;
        event.presentation_token = self.frame_trace_presentation_token;
        if let Some(tick) = self.frame_trace_scene_build_tick {
            event.scene_build_display_tick_sequence = tick.sequence;
            event.scene_build_target_display_time_ns = tick.target_time_ns();
            event.scene_build_tick_flags = tick.flags;
            event.flags |= tick.flags;
        } else {
            event.scene_build_tick_flags = crate::frame_trace::FLAG_DISPLAY_CURRENT_INVALID
                | crate::frame_trace::FLAG_DISPLAY_TARGET_INVALID
                | crate::frame_trace::FLAG_DISPLAY_REFRESH_INVALID;
            event.flags |= event.scene_build_tick_flags;
        }
        if let Some(tick) = self.frame_trace_presentation_tick {
            event.display_id = u64::from(tick.display_id);
            event.display_current_time_ns = tick.current_time_ns();
            event.display_refresh_period_ns =
                if tick.flags & crate::frame_trace::FLAG_DISPLAY_REFRESH_INVALID == 0 {
                    tick.refresh_period_ns
                } else {
                    0
                };
            event.display_current_host_time_raw = tick.current_host_time_raw;
            event.display_output_host_time_raw = tick.output_host_time_raw;
            event.display_video_refresh_period = tick.video_refresh_period;
            event.display_video_time_scale = i64::from(tick.video_time_scale);
            event.display_rate_scalar_bits = tick.rate_scalar.to_bits();
            event.presentation_display_tick_sequence = tick.sequence;
            event.presentation_target_display_time_ns = tick.target_time_ns();
            event.presentation_tick_flags = tick.flags;
            event.coalesced_display_tick_count = tick.coalesced_count;
            event.flags |= tick.flags;
        } else {
            event.presentation_tick_flags = crate::frame_trace::FLAG_DISPLAY_CURRENT_INVALID
                | crate::frame_trace::FLAG_DISPLAY_TARGET_INVALID
                | crate::frame_trace::FLAG_DISPLAY_REFRESH_INVALID;
            event.flags |= event.presentation_tick_flags;
        }
        let summary = self.frame_trace_summary;
        event.shadow_count = summary.shadow_count;
        event.quad_count = summary.quad_count;
        event.path_count = summary.path_count;
        event.sprite_count = summary.sprite_count;
        event.surface_count = summary.surface_count;
        event.shadow_expanded_area_device_px2 = summary.shadow_expanded_area_device_px2;
        event.quad_area_device_px2 = summary.quad_area_device_px2;
        event.path_segment_count = summary.path_segment_count;
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
        #[cfg(feature = "frame-trace")]
        {
            fn area(width: ScaledPixels, height: ScaledPixels) -> u64 {
                let area = f64::from(width).max(0.0) * f64::from(height).max(0.0);
                if !area.is_finite() || area <= 0.0 {
                    0
                } else if area >= u64::MAX as f64 {
                    u64::MAX
                } else {
                    area.round() as u64
                }
            }
            match &primitive {
                Primitive::Shadow(shadow) => {
                    self.frame_trace_summary.shadow_count =
                        self.frame_trace_summary.shadow_count.saturating_add(1);
                    let margin = f64::from(shadow.blur_radius).max(0.0) * 6.0;
                    let width = f64::from(shadow.bounds.size.width).max(0.0) + margin;
                    let height = f64::from(shadow.bounds.size.height).max(0.0) + margin;
                    let expanded_area = if width.is_finite() && height.is_finite() {
                        (width * height).round().clamp(0.0, u64::MAX as f64) as u64
                    } else {
                        0
                    };
                    self.frame_trace_summary.shadow_expanded_area_device_px2 = self
                        .frame_trace_summary
                        .shadow_expanded_area_device_px2
                        .saturating_add(expanded_area);
                }
                Primitive::Quad(_) => {
                    self.frame_trace_summary.quad_count =
                        self.frame_trace_summary.quad_count.saturating_add(1);
                    self.frame_trace_summary.quad_area_device_px2 = self
                        .frame_trace_summary
                        .quad_area_device_px2
                        .saturating_add(area(
                            clipped_bounds.size.width,
                            clipped_bounds.size.height,
                        ));
                }
                Primitive::Path(path) => {
                    self.frame_trace_summary.path_count =
                        self.frame_trace_summary.path_count.saturating_add(1);
                    self.frame_trace_summary.path_segment_count = self
                        .frame_trace_summary
                        .path_segment_count
                        .saturating_add(path.contour_count as u64);
                }
                Primitive::MonochromeSprite(_) | Primitive::PolychromeSprite(_) => {
                    self.frame_trace_summary.sprite_count =
                        self.frame_trace_summary.sprite_count.saturating_add(1);
                }
                Primitive::Surface(_) => {
                    self.frame_trace_summary.surface_count =
                        self.frame_trace_summary.surface_count.saturating_add(1);
                }
                Primitive::Underline(_) | Primitive::RasterTile(_) => {}
            }
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

    pub(crate) fn replay_vector_transform(
        &mut self,
        operations: &[PaintOperation],
        scale: f32,
        translation: Point<ScaledPixels>,
    ) {
        for operation in operations {
            match operation {
                PaintOperation::Primitive(primitive) => {
                    self.insert_primitive(primitive.transformed(scale, translation));
                }
                PaintOperation::StartLayer(bounds) => {
                    self.push_layer(transform_bounds(*bounds, scale, translation));
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

    fn transformed(&self, scale: f32, translation: Point<ScaledPixels>) -> Self {
        let transform_mask = |mask: &ContentMask<ScaledPixels>| {
            let mut mask = mask.clone();
            mask.bounds = transform_bounds(mask.bounds, scale, translation);
            mask
        };
        let conjugate = |matrix: TransformationMatrix| {
            let global = TransformationMatrix {
                rotation_scale: [[scale, 0.0], [0.0, scale]],
                translation: [translation.x.0, translation.y.0],
            };
            let inverse_scale = 1.0 / scale;
            let inverse = TransformationMatrix {
                rotation_scale: [[inverse_scale, 0.0], [0.0, inverse_scale]],
                translation: [
                    -translation.x.0 * inverse_scale,
                    -translation.y.0 * inverse_scale,
                ],
            };
            global.compose(matrix).compose(inverse)
        };
        let transform_corners = |corners: Corners<ScaledPixels>| Corners {
            top_left: corners.top_left * scale,
            top_right: corners.top_right * scale,
            bottom_right: corners.bottom_right * scale,
            bottom_left: corners.bottom_left * scale,
        };

        match self {
            Primitive::Shadow(template) => {
                let mut shadow = template.clone();
                shadow.bounds = transform_bounds(shadow.bounds, scale, translation);
                shadow.blur_radius *= scale;
                shadow.corner_radii = transform_corners(shadow.corner_radii);
                shadow.content_mask = transform_mask(&shadow.content_mask);
                shadow.transformation = conjugate(shadow.transformation);
                Primitive::Shadow(shadow)
            }
            Primitive::Quad(template) => {
                let mut quad = template.clone();
                quad.bounds = transform_bounds(quad.bounds, scale, translation);
                quad.content_mask = transform_mask(&quad.content_mask);
                quad.corner_radii = transform_corners(quad.corner_radii);
                quad.border_widths.top *= scale;
                quad.border_widths.right *= scale;
                quad.border_widths.bottom *= scale;
                quad.border_widths.left *= scale;
                quad.transformation = conjugate(quad.transformation);
                Primitive::Quad(quad)
            }
            Primitive::Path(template) => {
                let mut path = template.clone();
                path.bounds = transform_bounds(path.bounds, scale, translation);
                path.content_mask = transform_mask(&path.content_mask);
                path.start = transform_point(path.start, scale, translation);
                path.current = transform_point(path.current, scale, translation);
                for vertex in &mut path.vertices {
                    vertex.xy_position = transform_point(vertex.xy_position, scale, translation);
                    vertex.content_mask = transform_mask(&vertex.content_mask);
                }
                Primitive::Path(path)
            }
            Primitive::Underline(template) => {
                let mut underline = template.clone();
                underline.bounds = transform_bounds(underline.bounds, scale, translation);
                underline.content_mask = transform_mask(&underline.content_mask);
                underline.thickness *= scale;
                Primitive::Underline(underline)
            }
            Primitive::MonochromeSprite(template) => {
                let mut sprite = template.clone();
                sprite.bounds = transform_bounds(sprite.bounds, scale, translation);
                sprite.content_mask = transform_mask(&sprite.content_mask);
                sprite.transformation = conjugate(sprite.transformation);
                Primitive::MonochromeSprite(sprite)
            }
            Primitive::PolychromeSprite(template) => {
                let mut sprite = template.clone();
                sprite.bounds = transform_bounds(sprite.bounds, scale, translation);
                sprite.content_mask = transform_mask(&sprite.content_mask);
                sprite.corner_radii = transform_corners(sprite.corner_radii);
                sprite.transformation = conjugate(sprite.transformation);
                Primitive::PolychromeSprite(sprite)
            }
            Primitive::Surface(template) => {
                let mut surface = template.clone();
                surface.bounds = transform_bounds(surface.bounds, scale, translation);
                surface.content_mask = transform_mask(&surface.content_mask);
                Primitive::Surface(surface)
            }
            Primitive::RasterTile(template) => {
                let mut tile = template.clone();
                tile.bounds = transform_bounds(tile.bounds, scale, translation);
                tile.content_mask = transform_mask(&tile.content_mask);
                Primitive::RasterTile(tile)
            }
        }
    }
}

fn transform_point(
    point: Point<ScaledPixels>,
    scale: f32,
    translation: Point<ScaledPixels>,
) -> Point<ScaledPixels> {
    Point::new(
        point.x * scale + translation.x,
        point.y * scale + translation.y,
    )
}

fn transform_bounds(
    bounds: Bounds<ScaledPixels>,
    scale: f32,
    translation: Point<ScaledPixels>,
) -> Bounds<ScaledPixels> {
    Bounds::new(
        transform_point(bounds.origin, scale, translation),
        Size::new(bounds.size.width * scale, bounds.size.height * scale),
    )
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
        replayed.replay_vector_transform(&operations, 1.0, translation);
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
        replayed.replay_vector_transform(&operations, 1.0, scaled_point(-2., 6.));
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
    fn vector_transform_scales_geometry_style_masks_and_existing_transformation() {
        let bounds = scaled_bounds(10., 20., 30., 40.);
        let content_mask = ContentMask {
            bounds: scaled_bounds(0., 0., 100., 100.),
        };
        let transformation = TransformationMatrix::unit()
            .translate(scaled_point(10., 20.))
            .rotate(Radians(0.25))
            .translate(scaled_point(-10., -20.));
        let mut original = Scene::default();
        original.insert_primitive(Quad {
            bounds,
            content_mask,
            corner_radii: Corners {
                top_left: ScaledPixels(1.),
                top_right: ScaledPixels(2.),
                bottom_right: ScaledPixels(3.),
                bottom_left: ScaledPixels(4.),
            },
            border_widths: Edges {
                top: ScaledPixels(1.),
                right: ScaledPixels(2.),
                bottom: ScaledPixels(3.),
                left: ScaledPixels(4.),
            },
            transformation,
            ..Default::default()
        });
        let Some(operations) = original.clone_vector_paint(0..original.len()) else {
            panic!("quad must be supported");
        };

        let scale = 1.5;
        let translation = scaled_point(-7., 11.);
        let mut replayed = Scene::default();
        replayed.replay_vector_transform(&operations, scale, translation);
        let transformed = &replayed.quads[0];

        assert_eq!(transformed.bounds, scaled_bounds(8., 41., 45., 60.));
        assert_eq!(
            transformed.content_mask.bounds,
            scaled_bounds(-7., 11., 150., 150.)
        );
        assert_eq!(transformed.corner_radii.top_left, ScaledPixels(1.5));
        assert_eq!(transformed.corner_radii.bottom_left, ScaledPixels(6.));
        assert_eq!(transformed.border_widths.top, ScaledPixels(1.5));
        assert_eq!(transformed.border_widths.left, ScaledPixels(6.));

        let source_point = point(px(12.), px(22.));
        let old_output = transformation.apply(source_point);
        let transformed_input = point(
            px(source_point.x.0 * scale + translation.x.0),
            px(source_point.y.0 * scale + translation.y.0),
        );
        let expected_output = point(
            px(old_output.x.0 * scale + translation.x.0),
            px(old_output.y.0 * scale + translation.y.0),
        );
        let actual_output = transformed.transformation.apply(transformed_input);
        assert!((actual_output.x.0 - expected_output.x.0).abs() < 0.0001);
        assert!((actual_output.y.0 - expected_output.y.0).abs() < 0.0001);
    }

    #[test]
    fn vector_transform_scales_path_and_underline_coordinates() {
        let mask = ContentMask {
            bounds: scaled_bounds(-10., -10., 100., 100.),
        };
        let path = Path {
            id: PathId(0),
            order: 0,
            bounds: scaled_bounds(2., 3., 8., 9.),
            content_mask: mask.clone(),
            vertices: vec![PathVertex {
                xy_position: scaled_point(4., 5.),
                st_position: point(0., 1.),
                content_mask: mask.clone(),
            }],
            color: Background::default(),
            start: scaled_point(2., 3.),
            current: scaled_point(10., 12.),
            contour_count: 1,
        };
        let mut original = Scene::default();
        original.insert_primitive(path);
        original.insert_primitive(Underline {
            order: 0,
            pad: 0,
            bounds: scaled_bounds(1., 2., 20., 3.),
            content_mask: mask,
            color: Hsla::default(),
            thickness: ScaledPixels(2.),
            wavy: 0,
        });
        let Some(operations) = original.clone_vector_paint(0..original.len()) else {
            panic!("path and underline must be supported");
        };

        let mut replayed = Scene::default();
        replayed.replay_vector_transform(&operations, 2.0, scaled_point(-3., 4.));

        let path = &replayed.paths[0];
        assert_eq!(path.bounds, scaled_bounds(1., 10., 16., 18.));
        assert_eq!(path.start, scaled_point(1., 10.));
        assert_eq!(path.current, scaled_point(17., 28.));
        assert_eq!(path.vertices[0].xy_position, scaled_point(5., 14.));
        assert_eq!(
            path.vertices[0].content_mask.bounds,
            scaled_bounds(-23., -16., 200., 200.)
        );
        let underline = &replayed.underlines[0];
        assert_eq!(underline.bounds, scaled_bounds(-1., 8., 40., 6.));
        assert_eq!(underline.thickness, ScaledPixels(4.));
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
    #[cfg(feature = "frame-trace")]
    fn frame_trace_tick(
        display_id: u32,
        sequence: u64,
        coalesced_count: u64,
    ) -> crate::frame_trace::FrameTraceDisplayTick {
        crate::frame_trace::FrameTraceDisplayTick {
            display_id,
            sequence,
            worker_callback_time_ns: sequence * 10,
            current_host_time_raw: sequence * 100,
            output_host_time_raw: sequence * 100 + 50,
            video_refresh_period: 1,
            video_time_scale: 60,
            rate_scalar: 1.0,
            refresh_period_ns: 16_666_667,
            main_queue_delivery_time_ns: sequence * 10 + 1,
            coalesced_count,
            flags: 0,
        }
    }

    #[cfg(feature = "frame-trace")]
    #[test]
    fn frame_trace_dirty_draw_and_two_display_reuse_keep_exact_ticks() {
        let mut build_tick = frame_trace_tick(1, 10, 1);
        build_tick.flags = crate::frame_trace::FLAG_DISPLAY_TARGET_INVALID;
        let first_presentation_tick = frame_trace_tick(1, 11, 1);
        let second_presentation_tick = frame_trace_tick(2, 12, 1);
        let token = crate::frame_trace::PresentationToken {
            run_id_hash: 7,
            input_sequence: 8,
            snapshot_generation: 9,
            canvas_render_generation: 10,
        };
        let mut scene = Scene::default();
        scene.set_frame_trace_correlation(3, 8, Some(build_tick));
        scene.set_frame_trace_presentation_token(token);
        scene.insert_primitive(Quad {
            bounds: scaled_bounds(0., 0., 10., 20.),
            content_mask: ContentMask {
                bounds: scaled_bounds(0., 0., 10., 20.),
            },
            ..Default::default()
        });

        scene.set_frame_trace_presentation_attempt(20, Some(first_presentation_tick));
        let mut first = crate::frame_trace::FrameTraceEvent::now(
            crate::frame_trace::FrameTraceEventKind::CommandBufferSubmitted,
        );
        scene.populate_frame_trace_event(&mut first);
        scene.set_frame_trace_presentation_attempt(21, Some(second_presentation_tick));
        let mut reused = crate::frame_trace::FrameTraceEvent::now(
            crate::frame_trace::FrameTraceEventKind::CommandBufferSubmitted,
        );
        scene.populate_frame_trace_event(&mut reused);

        assert_eq!(first.presentation_token, Some(token));
        assert_eq!(reused.presentation_token, Some(token));
        assert_eq!(first.scene_build_display_tick_sequence, 10);
        assert_eq!(reused.scene_build_display_tick_sequence, 10);
        assert_eq!(first.presentation_display_tick_sequence, 11);
        assert_eq!(reused.presentation_display_tick_sequence, 12);
        assert_eq!(first.gpui_window_frame_id, 20);
        assert_eq!(reused.gpui_window_frame_id, 21);
        assert_eq!(first.display_id, 1);
        assert_eq!(reused.display_id, 2);
        assert_eq!(
            reused.scene_build_tick_flags,
            crate::frame_trace::FLAG_DISPLAY_TARGET_INVALID
        );
        assert_eq!(reused.presentation_tick_flags, 0);
        assert_ne!(
            reused.flags & crate::frame_trace::FLAG_DISPLAY_TARGET_INVALID,
            0
        );
        assert_eq!(reused.quad_count, 1);
        assert_eq!(reused.quad_area_device_px2, 200);
    }

    #[cfg(feature = "frame-trace")]
    #[test]
    fn frame_trace_hold_counts_only_distinct_non_coalesced_ticks() {
        let mut scene = Scene::default();
        scene.set_frame_trace_diagnostic_hold_ticks(2);

        scene.set_frame_trace_presentation_attempt(1, Some(frame_trace_tick(1, 10, 2)));
        assert!(scene.take_frame_trace_diagnostic_hold());
        scene.set_frame_trace_presentation_attempt(2, Some(frame_trace_tick(1, 11, 1)));
        assert!(scene.take_frame_trace_diagnostic_hold());
        scene.set_frame_trace_presentation_attempt(3, Some(frame_trace_tick(1, 11, 1)));
        assert!(scene.take_frame_trace_diagnostic_hold());
        scene.set_frame_trace_presentation_attempt(4, Some(frame_trace_tick(1, 12, 1)));
        assert!(scene.take_frame_trace_diagnostic_hold());
        scene.set_frame_trace_presentation_attempt(5, Some(frame_trace_tick(1, 13, 1)));
        assert!(!scene.take_frame_trace_diagnostic_hold());
    }
    #[cfg(feature = "frame-trace")]
    #[test]
    fn frame_trace_invalid_display_fields_never_reuse_raw_values_as_valid_times() {
        let mut tick = frame_trace_tick(7, 20, 1);
        tick.flags = crate::frame_trace::FLAG_DISPLAY_CURRENT_INVALID
            | crate::frame_trace::FLAG_DISPLAY_TARGET_INVALID
            | crate::frame_trace::FLAG_DISPLAY_REFRESH_INVALID;
        let mut scene = Scene::default();
        scene.set_frame_trace_correlation(1, 2, Some(tick));
        scene.set_frame_trace_presentation_attempt(3, Some(tick));
        let mut event = crate::frame_trace::FrameTraceEvent::now(
            crate::frame_trace::FrameTraceEventKind::CommandBufferSubmitted,
        );
        scene.populate_frame_trace_event(&mut event);

        assert_eq!(event.display_current_time_ns, 0);
        assert_eq!(event.presentation_target_display_time_ns, 0);
        assert_eq!(event.display_refresh_period_ns, 0);
        assert_eq!(
            event.display_current_host_time_raw,
            tick.current_host_time_raw
        );
        assert_eq!(
            event.display_output_host_time_raw,
            tick.output_host_time_raw
        );
        assert_eq!(event.flags & tick.flags, tick.flags);
    }
}
