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

/// Преобразование камеры, с которым была построена либо должна быть показана сцена.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorSceneTransform {
    /// Равномерный масштаб относительно координат подробной сцены.
    pub scale: f32,
    /// Перенос после применения масштаба.
    pub translation: Point<Pixels>,
}

impl VectorSceneTransform {
    /// Создаёт преобразование камеры; допустимость проверяется перед повторением.
    pub fn new(scale: f32, translation: Point<Pixels>) -> Self {
        Self { scale, translation }
    }

    fn is_valid(self) -> bool {
        self.scale.is_finite()
            && self.scale > 0.0
            && self.translation.x.0.is_finite()
            && self.translation.y.0.is_finite()
    }

    fn relative_to(self, captured: Self) -> Option<(f32, Point<Pixels>)> {
        if !self.is_valid() || !captured.is_valid() {
            return None;
        }
        let scale = self.scale / captured.scale;
        let translation = Point::new(
            self.translation.x - captured.translation.x * scale,
            self.translation.y - captured.translation.y * scale,
        );
        (scale.is_finite()
            && scale > 0.0
            && translation.x.0.is_finite()
            && translation.y.0.is_finite())
        .then_some((scale, translation))
    }
}

impl Default for VectorSceneTransform {
    fn default() -> Self {
        Self {
            scale: 1.0,
            translation: Point::default(),
        }
    }
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
    cached_vector_scene_transform(
        handle,
        revision,
        coverage,
        viewport,
        VectorSceneTransform::new(1.0, translation),
        child,
    )
}

/// Повторяет ранее захваченные векторные операции с новым масштабом и переносом камеры.
///
/// При изменении `revision`, выходе `viewport` за `coverage`, недопустимом
/// преобразовании или наличии неподдерживаемого примитива подробный `child`
/// строится синхронно в том же кадре.
pub fn cached_vector_scene_transform(
    handle: VectorSceneCacheHandle,
    revision: VectorSceneRevision,
    coverage: VectorSceneCoverage,
    viewport: Bounds<Pixels>,
    transform: VectorSceneTransform,
    child: impl IntoElement,
) -> CachedVectorSceneTranslation {
    CachedVectorSceneTranslation {
        handle,
        revision,
        coverage,
        viewport,
        transform,
        child: Some(child.into_any_element()),
        style: StyleRefinement::default(),
    }
}

/// Граница элемента для повторения векторной сцены при преобразовании камеры.
pub struct CachedVectorSceneTranslation {
    handle: VectorSceneCacheHandle,
    revision: VectorSceneRevision,
    coverage: VectorSceneCoverage,
    viewport: Bounds<Pixels>,
    transform: VectorSceneTransform,
    child: Option<AnyElement>,
    style: StyleRefinement,
}

struct VectorSceneState {
    revision: VectorSceneRevision,
    coverage: VectorSceneCoverage,
    captured_transform: VectorSceneTransform,
    scale_factor_bits: u32,
    operations: Vec<PaintOperation>,
    ready: bool,
}

impl Default for VectorSceneState {
    fn default() -> Self {
        Self {
            revision: VectorSceneRevision::default(),
            coverage: VectorSceneCoverage::default(),
            captured_transform: VectorSceneTransform::default(),
            scale_factor_bits: 0,
            operations: Vec::new(),
            ready: false,
        }
    }
}

/// Внутреннее состояние одного prepaint-прохода кэшируемой сцены.
pub struct VectorScenePrepaintState {
    hit: bool,
    relative_scale: f32,
    relative_translation: Point<Pixels>,
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
        let relative_transform = id.and_then(|id| {
            window.with_element_state::<VectorSceneState, _>(id, |state, _window| {
                let state = state.unwrap_or_default();
                let relative = (state.ready
                    && state.revision == self.revision
                    && state.scale_factor_bits == scale_factor_bits
                    && state.coverage.contains(self.viewport))
                .then(|| self.transform.relative_to(state.captured_transform))
                .flatten();
                (relative, state)
            })
        });
        if let Some((relative_scale, relative_translation)) = relative_transform {
            self.handle.inner.hits.fetch_add(1, Ordering::Relaxed);
            return VectorScenePrepaintState {
                hit: true,
                relative_scale,
                relative_translation,
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
            relative_scale: 1.0,
            relative_translation: Point::default(),
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
            let relative_scale = prepaint.relative_scale;
            let relative_translation = prepaint.relative_translation;
            window.with_element_state::<VectorSceneState, _>(id, |state, window| {
                let state = state.unwrap_or_default();
                let translation = Point::new(
                    relative_translation.x.scale(scale_factor),
                    relative_translation.y.scale(scale_factor),
                );
                window.next_frame.scene.replay_vector_transform(
                    &state.operations,
                    relative_scale,
                    translation,
                );
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
        let supported =
            prepaint.cache_id_available && self.transform.is_valid() && operations.is_some();
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
            state.captured_transform = self.transform;
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

    use super::{VectorSceneCoverage, VectorSceneTransform};

    #[test]
    fn coverage_uses_inclusive_outer_edges() {
        let coverage =
            VectorSceneCoverage(bounds(point(px(-100.), px(-50.)), size(px(300.), px(200.))));
        assert!(coverage.contains(bounds(point(px(-100.), px(-50.)), size(px(300.), px(200.)),)));
        assert!(!coverage.contains(bounds(point(px(-101.), px(-50.)), size(px(300.), px(200.)),)));
    }

    #[test]
    fn relative_transform_maps_captured_screen_coordinates_to_current_ones() {
        let captured = VectorSceneTransform::new(0.5, point(px(20.), px(-10.)));
        let current = VectorSceneTransform::new(0.8, point(px(-5.), px(30.)));
        let (scale, translation) = current.relative_to(captured).expect("valid transform");

        assert!((scale - 1.6).abs() < f32::EPSILON);
        assert!((translation.x.0 + 37.0).abs() < f32::EPSILON);
        assert!((translation.y.0 - 46.0).abs() < f32::EPSILON);
    }

    #[test]
    fn invalid_transform_cannot_be_replayed() {
        let valid = VectorSceneTransform::default();
        assert!(
            VectorSceneTransform::new(0.0, point(px(0.), px(0.)))
                .relative_to(valid)
                .is_none()
        );
        assert!(
            VectorSceneTransform::new(1.0, point(px(f32::NAN), px(0.)))
                .relative_to(valid)
                .is_none()
        );
    }
}
