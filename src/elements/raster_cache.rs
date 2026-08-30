use std::mem;

use crate::{
    AnyElement, App, Bounds, ContentMask, DevicePixels, Element, GlobalElementId,
    InspectorElementId, IntoElement, LayoutId, Pixels, RasterCompositorSurface,
    RasterCompositorTransform, RasterCompositorTransformHandle, RasterTile, RasterTileHit,
    RasterTileMiss, RasterTileUpdate, RasterTileUpdateBatch, RasterTileUpdateTarget, Size, Style,
    StyleRefinement, Styled, Window,
};
use refineable::Refineable as _;

/// Composes a renderer-resident tile without rebuilding its detailed child tree.
pub fn cached_raster_tile(hit: RasterTileHit) -> CachedRasterTile {
    CachedRasterTile {
        hit,
        style: StyleRefinement::default(),
    }
}

/// Captures a Canvas subtree into an independently composited raster surface.
///
/// The platform renderer owns the actual surface. `captured_transform` records the camera used to
/// build `child`; later display ticks can apply the handle's latest transform without rebuilding
/// the detailed GPUI scene.
pub fn raster_compositor_surface(
    handle: RasterCompositorTransformHandle,
    captured_transform: RasterCompositorTransform,
    child: impl IntoElement,
) -> RasterCompositorSurfaceElement {
    RasterCompositorSurfaceElement {
        handle,
        captured_transform,
        child: Some(child.into_any_element()),
    }
}

/// Element boundary for an independently composited Canvas scene.
pub struct RasterCompositorSurfaceElement {
    handle: RasterCompositorTransformHandle,
    captured_transform: RasterCompositorTransform,
    child: Option<AnyElement>,
}

impl IntoElement for RasterCompositorSurfaceElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for RasterCompositorSurfaceElement {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<crate::ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        (self.child.as_mut().unwrap().request_layout(window, cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        self.child.as_mut().unwrap().prepaint(window, cx);
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut (),
        _prepaint: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        let scale_factor = window.scale_factor();
        let raster_bounds = bounds.dilate(crate::px(512. / scale_factor));
        let parent_scene = mem::take(&mut window.next_frame.scene);
        window.with_content_mask(
            Some(ContentMask {
                bounds: raster_bounds,
            }),
            |window| self.child.as_mut().unwrap().paint(window, cx),
        );
        let mut surface_scene = mem::replace(&mut window.next_frame.scene, parent_scene);
        surface_scene.finish();
        window
            .next_frame
            .scene
            .raster_compositor_surfaces
            .push(RasterCompositorSurface {
                handle: self.handle.clone(),
                captured_transform: self.captured_transform,
                clip_bounds: bounds.scale(scale_factor),
                raster_bounds: raster_bounds.scale(scale_factor),
                scene: surface_scene,
            });
    }
}

/// Paints detailed content into a GPU-resident tile and composes it in the same frame.
pub fn rasterize_tile(
    miss: RasterTileMiss,
    texture_size: Size<DevicePixels>,
    gutter: DevicePixels,
    child: impl IntoElement,
) -> RasterizeTile {
    RasterizeTile {
        miss,
        texture_size,
        gutter,
        child: Some(child.into_any_element()),
        compose: true,
    }
}

/// Paints detailed content into a GPU-resident tile without composing it.
///
/// This is intended for application shadow modes that validate cache behavior
/// while keeping the ordinary detailed scene as the visible authority.
pub fn rasterize_tile_shadow(
    miss: RasterTileMiss,
    texture_size: Size<DevicePixels>,
    gutter: DevicePixels,
    child: impl IntoElement,
) -> RasterizeTile {
    RasterizeTile {
        miss,
        texture_size,
        gutter,
        child: Some(child.into_any_element()),
        compose: false,
    }
}

/// Captures detailed content once and populates several aligned GPU-resident tiles.
pub fn rasterize_tiles(
    captures: impl IntoIterator<Item = (RasterTileMiss, Bounds<Pixels>, Bounds<Pixels>)>,
    texture_size: Size<DevicePixels>,
    gutter: DevicePixels,
    child: impl IntoElement,
) -> RasterizeTiles {
    rasterize_tiles_inner(captures, texture_size, gutter, child, true, false)
}

/// Captures detailed content once and populates several tiles without composing them.
pub fn rasterize_tiles_shadow(
    captures: impl IntoIterator<Item = (RasterTileMiss, Bounds<Pixels>, Bounds<Pixels>)>,
    texture_size: Size<DevicePixels>,
    gutter: DevicePixels,
    child: impl IntoElement,
) -> RasterizeTiles {
    rasterize_tiles_inner(captures, texture_size, gutter, child, false, true)
}

/// Populates several tiles after the visible frame without diagnostic readback.
///
/// This supports atomic level replacement in production: the previous complete
/// level remains visible while the next one is prepared entirely on the GPU.
pub fn rasterize_tiles_deferred(
    captures: impl IntoIterator<Item = (RasterTileMiss, Bounds<Pixels>, Bounds<Pixels>)>,
    texture_size: Size<DevicePixels>,
    gutter: DevicePixels,
    child: impl IntoElement,
) -> RasterizeTiles {
    rasterize_tiles_inner(captures, texture_size, gutter, child, false, false)
}

fn rasterize_tiles_inner(
    captures: impl IntoIterator<Item = (RasterTileMiss, Bounds<Pixels>, Bounds<Pixels>)>,
    texture_size: Size<DevicePixels>,
    gutter: DevicePixels,
    child: impl IntoElement,
    compose: bool,
    verify: bool,
) -> RasterizeTiles {
    RasterizeTiles {
        captures: captures.into_iter().collect(),
        texture_size,
        gutter,
        child: Some(child.into_any_element()),
        compose,
        verify,
    }
}

/// An element that emits one cached texture composition primitive.
pub struct CachedRasterTile {
    hit: RasterTileHit,
    style: StyleRefinement,
}

impl IntoElement for CachedRasterTile {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for CachedRasterTile {
    type RequestLayoutState = Style;
    type PrepaintState = ();

    fn id(&self) -> Option<crate::ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Style) {
        let mut style = Style::default();
        style.refine(&self.style);
        (window.request_layout(style.clone(), [], cx), style)
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Style,
        _window: &mut Window,
        _cx: &mut App,
    ) {
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Style,
        _prepaint: &mut (),
        window: &mut Window,
        _cx: &mut App,
    ) {
        let scale_factor = window.scale_factor();
        window.next_frame.scene.insert_primitive(RasterTile {
            order: 0,
            bounds: bounds.scale(scale_factor),
            content_mask: window.content_mask().scale(scale_factor),
            cache_id: self.hit.cache.id(),
            key: self.hit.key.value(),
            revision: self.hit.revision.value(),
            gutter: self.hit.gutter,
        });
    }
}

impl Styled for CachedRasterTile {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

/// An element whose child is captured once into a group of renderer-owned raster tiles.
pub struct RasterizeTiles {
    captures: Vec<(RasterTileMiss, Bounds<Pixels>, Bounds<Pixels>)>,
    texture_size: Size<DevicePixels>,
    gutter: DevicePixels,
    child: Option<AnyElement>,
    compose: bool,
    verify: bool,
}

impl IntoElement for RasterizeTiles {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for RasterizeTiles {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<crate::ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        (self.child.as_mut().unwrap().request_layout(window, cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        self.child.as_mut().unwrap().prepaint(window, cx);
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut (),
        _prepaint: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        if self.captures.is_empty() {
            return;
        }

        let mut parent_scene = mem::take(&mut window.next_frame.scene);
        let scale_factor = window.scale_factor();
        let Some(capture_bounds) = self
            .captures
            .iter()
            .map(|(_, source_bounds, _)| *source_bounds)
            .reduce(|left, right| left.union(&right))
        else {
            return;
        };
        let capture_bounds = capture_bounds.dilate(crate::px(self.gutter.0 as f32 / scale_factor));
        window.with_content_mask(
            Some(ContentMask {
                bounds: capture_bounds,
            }),
            |window| self.child.as_mut().unwrap().paint(window, cx),
        );
        let mut tile_scene = mem::replace(&mut window.next_frame.scene, parent_scene);
        tile_scene.finish();

        let Some((first_miss, _, _)) = self.captures.first() else {
            return;
        };
        let cache = first_miss.cache.clone();
        debug_assert!(self.captures.iter().all(|(miss, _, _)| miss.cache == cache));
        let targets = self
            .captures
            .iter()
            .map(|(miss, source_bounds, _)| RasterTileUpdateTarget {
                key: miss.key,
                revision: miss.revision,
                source_bounds: source_bounds.scale(scale_factor),
            })
            .collect::<Vec<_>>();
        window
            .next_frame
            .scene
            .raster_tile_update_batches
            .push(RasterTileUpdateBatch {
                config: cache.config(),
                cache: cache.clone(),
                texture_size: self.texture_size,
                gutter: self.gutter,
                targets,
                scene: tile_scene,
                deferred: !self.compose,
                verify: self.verify,
            });

        if self.compose {
            for (miss, _, composition_bounds) in &self.captures {
                let scaled_bounds = composition_bounds.scale(scale_factor);
                window.next_frame.scene.insert_primitive(RasterTile {
                    order: 0,
                    bounds: scaled_bounds,
                    content_mask: ContentMask {
                        bounds: scaled_bounds,
                    },
                    cache_id: cache.id(),
                    key: miss.key.value(),
                    revision: miss.revision.value(),
                    gutter: self.gutter.0 as u32,
                });
            }
        }
    }
}

/// An element whose child is captured into a renderer-owned raster tile.
pub struct RasterizeTile {
    miss: RasterTileMiss,
    texture_size: Size<DevicePixels>,
    gutter: DevicePixels,
    child: Option<AnyElement>,
    compose: bool,
}

impl IntoElement for RasterizeTile {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for RasterizeTile {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<crate::ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        (self.child.as_mut().unwrap().request_layout(window, cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        self.child.as_mut().unwrap().prepaint(window, cx);
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut (),
        _prepaint: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        let mut parent_scene = mem::take(&mut window.next_frame.scene);
        let scale_factor = window.scale_factor();
        let capture_bounds = bounds.dilate(crate::px(self.gutter.0 as f32 / scale_factor));
        window.with_content_mask(
            Some(ContentMask {
                bounds: capture_bounds,
            }),
            |window| self.child.as_mut().unwrap().paint(window, cx),
        );
        let mut tile_scene = mem::replace(&mut window.next_frame.scene, parent_scene);
        tile_scene.finish();

        let scaled_bounds = bounds.scale(scale_factor);
        let cache = self.miss.cache.clone();
        window
            .next_frame
            .scene
            .raster_tile_updates
            .push(RasterTileUpdate {
                config: cache.config(),
                cache: cache.clone(),
                key: self.miss.key,
                revision: self.miss.revision,
                texture_size: self.texture_size,
                gutter: self.gutter,
                source_bounds: scaled_bounds,
                scene: tile_scene,
            });
        if self.compose {
            window.next_frame.scene.insert_primitive(RasterTile {
                order: 0,
                bounds: scaled_bounds,
                content_mask: ContentMask {
                    bounds: scaled_bounds,
                },
                cache_id: cache.id(),
                key: self.miss.key.value(),
                revision: self.miss.revision.value(),
                gutter: self.gutter.0 as u32,
            });
        }
    }
}
