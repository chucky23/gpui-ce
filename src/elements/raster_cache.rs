use std::mem;

use crate::{
    AnyElement, App, Bounds, ContentMask, DevicePixels, Element, GlobalElementId,
    InspectorElementId, IntoElement, LayoutId, Pixels, RasterTile, RasterTileHit, RasterTileMiss,
    RasterTileUpdate, Size, Style, StyleRefinement, Styled, Window,
};
use refineable::Refineable as _;

/// Composes a renderer-resident tile without rebuilding its detailed child tree.
pub fn cached_raster_tile(hit: RasterTileHit) -> CachedRasterTile {
    CachedRasterTile {
        hit,
        style: StyleRefinement::default(),
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
            texture_width: self.hit.texture_width,
            texture_height: self.hit.texture_height,
            gutter: self.hit.gutter,
        });
    }
}

impl Styled for CachedRasterTile {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

/// An element whose child is captured into a renderer-owned raster tile.
pub struct RasterizeTile {
    miss: RasterTileMiss,
    texture_size: Size<DevicePixels>,
    gutter: DevicePixels,
    child: Option<AnyElement>,
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
        self.child.as_mut().unwrap().paint(window, cx);
        let mut tile_scene = mem::replace(&mut window.next_frame.scene, parent_scene);
        tile_scene.finish();

        let scale_factor = window.scale_factor();
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
        window.next_frame.scene.insert_primitive(RasterTile {
            order: 0,
            bounds: scaled_bounds,
            content_mask: ContentMask {
                bounds: scaled_bounds,
            },
            cache_id: cache.id(),
            key: self.miss.key.value(),
            revision: self.miss.revision.value(),
            texture_width: self.texture_size.width.0 as u32,
            texture_height: self.texture_size.height.0 as u32,
            gutter: self.gutter.0 as u32,
        });
    }
}
