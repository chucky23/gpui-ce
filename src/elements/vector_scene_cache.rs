use std::sync::{
    Arc,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

use refineable::Refineable as _;

use crate::{
    AnyElement, App, Bounds, Element, ElementId, GlobalElementId, InspectorElementId, IntoElement,
    LayoutId, PaintOperation, Pixels, Point, Style, StyleRefinement, Styled, Window,
};

static NEXT_VECTOR_SCENE_CACHE_ID: AtomicU64 = AtomicU64::new(1);

/// Владелец одного независимо повторяемого диапазона векторной сцены.
#[derive(Clone, Debug)]
pub struct VectorSceneCacheHandle {
    inner: Arc<VectorSceneCacheInner>,
}

#[derive(Debug)]
struct VectorSceneCacheInner {
    id: u64,
    hits: AtomicU64,
    misses: AtomicU64,
    unsupported: AtomicU64,
    primitive_count: AtomicUsize,
}

impl VectorSceneCacheHandle {
    /// Создаёт пустое пространство кэша с независимой статистикой.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(VectorSceneCacheInner {
                id: NEXT_VECTOR_SCENE_CACHE_ID.fetch_add(1, Ordering::Relaxed),
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
                unsupported: AtomicU64::new(0),
                primitive_count: AtomicUsize::new(0),
            }),
        }
    }

    /// Возвращает накопленную статистику повторения сцены.
    pub fn stats(&self) -> VectorSceneCacheStats {
        VectorSceneCacheStats {
            hits: self.inner.hits.load(Ordering::Relaxed),
            misses: self.inner.misses.load(Ordering::Relaxed),
            unsupported: self.inner.unsupported.load(Ordering::Relaxed),
            primitive_count: self.inner.primitive_count.load(Ordering::Relaxed),
        }
    }
}

impl Default for VectorSceneCacheHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// Ревизия содержимого, не зависящая от положения камеры.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct VectorSceneRevision(pub u64);

/// Полная область логической сцены, представленная захваченными операциями.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VectorSceneCoverage(pub Bounds<Pixels>);

impl VectorSceneCoverage {
    /// Проверяет, что текущая область просмотра целиком входит в покрытие кэша.
    pub fn contains(self, viewport: Bounds<Pixels>) -> bool {
        viewport.left() >= self.0.left()
            && viewport.top() >= self.0.top()
            && viewport.right() <= self.0.right()
            && viewport.bottom() <= self.0.bottom()
    }
}

/// Счётчики одного пространства векторной сцены.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VectorSceneCacheStats {
    /// Число кадров, в которых подробное дерево не строилось.
    pub hits: u64,
    /// Число подробных построений сцены.
    pub misses: u64,
    /// Число построений с примитивом, который нельзя безопасно повторить.
    pub unsupported: u64,
    /// Число операций в последнем успешно захваченном диапазоне.
    pub primitive_count: usize,
}

/// Повторяет ранее захваченные векторные операции с новым переносом камеры.
///
/// При изменении `revision`, выходе `viewport` за `coverage` или наличии
/// неподдерживаемого примитива подробный `child` строится синхронно в том же кадре.
pub fn cached_vector_scene_translation(
    handle: VectorSceneCacheHandle,
    revision: VectorSceneRevision,
    coverage: VectorSceneCoverage,
    viewport: Bounds<Pixels>,
    translation: Point<Pixels>,
    child: impl IntoElement,
) -> CachedVectorSceneTranslation {
    CachedVectorSceneTranslation {
        handle,
        revision,
        coverage,
        viewport,
        translation,
        child: Some(child.into_any_element()),
        style: StyleRefinement::default(),
    }
}

/// Граница элемента для повторения векторной сцены при переносе.
pub struct CachedVectorSceneTranslation {
    handle: VectorSceneCacheHandle,
    revision: VectorSceneRevision,
    coverage: VectorSceneCoverage,
    viewport: Bounds<Pixels>,
    translation: Point<Pixels>,
    child: Option<AnyElement>,
    style: StyleRefinement,
}

struct VectorSceneState {
    revision: VectorSceneRevision,
    coverage: VectorSceneCoverage,
    captured_translation: Point<Pixels>,
    scale_factor_bits: u32,
    operations: Vec<PaintOperation>,
    ready: bool,
}

impl Default for VectorSceneState {
    fn default() -> Self {
        Self {
            revision: VectorSceneRevision::default(),
            coverage: VectorSceneCoverage::default(),
            captured_translation: Point::default(),
            scale_factor_bits: 0,
            operations: Vec::new(),
            ready: false,
        }
    }
}

/// Внутреннее состояние одного prepaint-прохода кэшируемой сцены.
pub struct VectorScenePrepaintState {
    hit: bool,
    child: Option<AnyElement>,
    prepaint_safe: bool,
    cache_id_available: bool,
}

impl IntoElement for CachedVectorSceneTranslation {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for CachedVectorSceneTranslation {
    type RequestLayoutState = Style;
    type PrepaintState = VectorScenePrepaintState;

    fn id(&self) -> Option<ElementId> {
        Some(ElementId::NamedInteger(
            "vector-scene-cache".into(),
            self.handle.inner.id,
        ))
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
        (window.request_layout(style.clone(), None, cx), style)
    }

    fn prepaint(
        &mut self,
        id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Style,
        window: &mut Window,
        cx: &mut App,
    ) -> VectorScenePrepaintState {
        let scale_factor_bits = window.scale_factor().to_bits();
        let hit = id.is_some_and(|id| {
            window.with_element_state::<VectorSceneState, _>(id, |state, _window| {
                let state = state.unwrap_or_default();
                let hit = state.ready
                    && state.revision == self.revision
                    && state.scale_factor_bits == scale_factor_bits
                    && state.coverage.contains(self.viewport);
                (hit, state)
            })
        });
        if hit {
            self.handle.inner.hits.fetch_add(1, Ordering::Relaxed);
            return VectorScenePrepaintState {
                hit: true,
                child: None,
                prepaint_safe: true,
                cache_id_available: true,
            };
        }

        self.handle.inner.misses.fetch_add(1, Ordering::Relaxed);
        let hitboxes_before = window.next_frame.hitboxes.len();
        let deferred_before = window.next_frame.deferred_draws.len();
        let mut child = self.child.take();
        if let Some(child) = child.as_mut() {
            child.layout_as_root(bounds.size.into(), window, cx);
            child.prepaint_at(bounds.origin, window, cx);
        }
        VectorScenePrepaintState {
            hit: false,
            child,
            prepaint_safe: hitboxes_before == window.next_frame.hitboxes.len()
                && deferred_before == window.next_frame.deferred_draws.len(),
            cache_id_available: id.is_some(),
        }
    }

    fn paint(
        &mut self,
        id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Style,
        prepaint: &mut VectorScenePrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        if prepaint.hit {
            let Some(id) = id else {
                return;
            };
            let scale_factor = window.scale_factor();
            window.with_element_state::<VectorSceneState, _>(id, |state, window| {
                let state = state.unwrap_or_default();
                let delta = Point::new(
                    (self.translation.x - state.captured_translation.x).scale(scale_factor),
                    (self.translation.y - state.captured_translation.y).scale(scale_factor),
                );
                window
                    .next_frame
                    .scene
                    .replay_vector_translation(&state.operations, delta);
                ((), state)
            });
            return;
        }

        let scene_start = window.next_frame.scene.len();
        let mouse_listeners_before = window.next_frame.mouse_listeners.len();
        let input_handlers_before = window.next_frame.input_handlers.len();
        let cursor_styles_before = window.next_frame.cursor_styles.len();
        let Some(child) = prepaint.child.as_mut() else {
            return;
        };
        child.paint(window, cx);
        let scene_end = window.next_frame.scene.len();
        let operations = prepaint
            .prepaint_safe
            .then(|| {
                window
                    .next_frame
                    .scene
                    .clone_vector_paint(scene_start..scene_end)
            })
            .flatten()
            .filter(|_| {
                mouse_listeners_before == window.next_frame.mouse_listeners.len()
                    && input_handlers_before == window.next_frame.input_handlers.len()
                    && cursor_styles_before == window.next_frame.cursor_styles.len()
            });
        let supported = prepaint.cache_id_available && operations.is_some();
        let operation_count = operations.as_ref().map_or(0, Vec::len);
        if !supported {
            self.handle
                .inner
                .unsupported
                .fetch_add(1, Ordering::Relaxed);
        }
        self.handle
            .inner
            .primitive_count
            .store(operation_count, Ordering::Relaxed);
        let scale_factor_bits = window.scale_factor().to_bits();
        let Some(id) = id else {
            return;
        };
        window.with_element_state::<VectorSceneState, _>(id, |state, _window| {
            let mut state = state.unwrap_or_default();
            state.revision = self.revision;
            state.coverage = self.coverage;
            state.captured_translation = self.translation;
            state.scale_factor_bits = scale_factor_bits;
            state.operations = operations.unwrap_or_default();
            state.ready = supported;
            ((), state)
        });
    }
}

impl Styled for CachedVectorSceneTranslation {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

#[cfg(test)]
mod tests {
    use crate::{bounds, point, px, size};

    use super::VectorSceneCoverage;

    #[test]
    fn coverage_uses_inclusive_outer_edges() {
        let coverage =
            VectorSceneCoverage(bounds(point(px(-100.), px(-50.)), size(px(300.), px(200.))));
        assert!(coverage.contains(bounds(point(px(-100.), px(-50.)), size(px(300.), px(200.)),)));
        assert!(!coverage.contains(bounds(point(px(-101.), px(-50.)), size(px(300.), px(200.)),)));
    }
}
