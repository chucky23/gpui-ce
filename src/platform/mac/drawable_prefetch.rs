//! Заблаговременное получение drawable во вспомогательном потоке.
//!
//! `CAMetalLayer.nextDrawable()` блокирует вызывающий поток, пока конвейер показа не вернёт
//! свободный буфер. Под нагрузкой WindowServer это до 15 мс, и главный поток пропускает такт
//! DisplayLink и задерживает ввод. Здесь блокируется вспомогательный поток, а главный берёт
//! готовый drawable без ожидания или откладывает показ до сигнала готовности.
//!
//! Логика очереди отделена от Metal: источник drawable и сигнал главному потоку передаются
//! замыканиями, поэтому её можно проверить без окна.

use std::{
    mem,
    sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock},
    thread,
};

/// Переменная окружения, включающая заблаговременное получение drawable. По умолчанию
/// выключено: одна сборка меряется в обоих режимах.
pub(crate) const ASYNC_DRAWABLE_ENV: &str = "GPUI_ASYNC_DRAWABLE";

pub(crate) fn async_drawable_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os(ASYNC_DRAWABLE_ENV).is_some_and(|value| value == "1"))
}

/// Итог неблокирующей попытки взять drawable.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DrawableTake<D> {
    Ready(D),
    /// Вспомогательный поток ещё ждёт буфер. После его получения вызывается сигнал готовности.
    NotReady,
    /// Источник вернул пустой результат, пока показ ждал (у `nextDrawable` истёк тайм-аут или
    /// у слоя нулевой размер). Новое получение запрошено без сигнала готовности: иначе слой без
    /// буферов зациклил бы повторы показа. Как и прежде, показ повторит следующее изменение
    /// сцены.
    Unavailable,
}

struct PrefetchState<D> {
    ready: Option<D>,
    acquire_requested: bool,
    acquiring: bool,
    /// Источник вернул пустой результат, пока его ждали. Пустой результат фонового получения
    /// не запоминается: к следующему показу он устарел бы.
    last_acquire_failed: bool,
    /// Главный поток отложил показ и ждёт сигнала готовности.
    presentation_waiting: bool,
    /// `take_blocking` ждёт результата на условной переменной.
    blocking_waiting: bool,
    shutdown: bool,
}

impl<D> PrefetchState<D> {
    fn request_acquire(&mut self) {
        if self.ready.is_none() && !self.acquiring {
            self.acquire_requested = true;
        }
    }
}

struct PrefetchShared<D> {
    state: Mutex<PrefetchState<D>>,
    changed: Condvar,
}

impl<D> PrefetchShared<D> {
    fn lock(&self) -> MutexGuard<'_, PrefetchState<D>> {
        // Состояние меняется короткими участками без паники внутри, поэтому отравленный
        // мьютекс не оставляет его несогласованным.
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Держит не больше одного заранее взятого drawable. Источник вызывается только из одного
/// вспомогательного потока, поэтому `nextDrawable` слоя никогда не выполняется параллельно.
pub(crate) struct DrawablePrefetcher<D: Send + 'static> {
    shared: Arc<PrefetchShared<D>>,
}

impl<D: Send + 'static> DrawablePrefetcher<D> {
    /// Запускает вспомогательный поток. Первый drawable запрашивает первая попытка взять его:
    /// при создании окна у слоя ещё нет размера.
    ///
    /// `acquire` блокируется, пока буфер не освободится. `notify_ready` вызывается из
    /// вспомогательного потока, когда отложенный показ можно повторить.
    pub(crate) fn spawn(
        thread_name: &str,
        acquire: impl Fn() -> Option<D> + Send + 'static,
        notify_ready: impl Fn() + Send + 'static,
    ) -> std::io::Result<Self> {
        let shared = Arc::new(PrefetchShared {
            state: Mutex::new(PrefetchState {
                ready: None,
                acquire_requested: false,
                acquiring: false,
                last_acquire_failed: false,
                presentation_waiting: false,
                blocking_waiting: false,
                shutdown: false,
            }),
            changed: Condvar::new(),
        });
        thread::Builder::new().name(thread_name.into()).spawn({
            let shared = shared.clone();
            move || run_worker(&shared, acquire, notify_ready)
        })?;
        Ok(Self { shared })
    }

    /// Берёт готовый drawable без ожидания. Drawable, для которого `usable` ложно (размер слоя
    /// сменился после взятия), выбрасывается, и запрашивается новый.
    pub(crate) fn try_take(&self, usable: impl Fn(&D) -> bool) -> DrawableTake<D> {
        let mut state = self.shared.lock();
        let stale = match state.ready.take() {
            Some(drawable) if usable(&drawable) => {
                state.request_acquire();
                drop(state);
                self.shared.changed.notify_all();
                return DrawableTake::Ready(drawable);
            }
            stale => stale,
        };
        let failed = mem::take(&mut state.last_acquire_failed);
        state.request_acquire();
        state.presentation_waiting = !failed;
        drop(state);
        self.shared.changed.notify_all();
        drop(stale);
        if failed {
            DrawableTake::Unavailable
        } else {
            DrawableTake::NotReady
        }
    }

    /// Ждёт drawable, пригодный по `usable`. Нужен синхронным кадрам
    /// (`presentsWithTransaction`), которые нельзя отложить. `None` — источник вернул пустой
    /// результат.
    pub(crate) fn take_blocking(&self, usable: impl Fn(&D) -> bool) -> Option<D> {
        let mut state = self.shared.lock();
        loop {
            match state.ready.take() {
                Some(drawable) if usable(&drawable) => {
                    state.request_acquire();
                    drop(state);
                    self.shared.changed.notify_all();
                    return Some(drawable);
                }
                Some(stale) => {
                    state.request_acquire();
                    drop(state);
                    self.shared.changed.notify_all();
                    drop(stale);
                    state = self.shared.lock();
                    continue;
                }
                None => {}
            }
            if mem::take(&mut state.last_acquire_failed) {
                state.blocking_waiting = false;
                return None;
            }
            state.request_acquire();
            state.blocking_waiting = true;
            self.shared.changed.notify_all();
            state = self
                .shared
                .changed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.blocking_waiting = false;
        }
    }
}

impl<D: Send + 'static> Drop for DrawablePrefetcher<D> {
    fn drop(&mut self) {
        let mut state = self.shared.lock();
        state.shutdown = true;
        let ready = state.ready.take();
        drop(state);
        self.shared.changed.notify_all();
        drop(ready);
    }
}

fn run_worker<D>(
    shared: &PrefetchShared<D>,
    acquire: impl Fn() -> Option<D>,
    notify_ready: impl Fn(),
) {
    loop {
        let mut state = shared.lock();
        while !state.acquire_requested && !state.shutdown {
            state = shared
                .changed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        if state.shutdown {
            return;
        }
        state.acquire_requested = false;
        state.acquiring = true;
        drop(state);

        let drawable = acquire();

        let mut state = shared.lock();
        state.acquiring = false;
        if state.shutdown {
            drop(state);
            drop(drawable);
            return;
        }
        match drawable {
            Some(drawable) => state.ready = Some(drawable),
            None => {
                state.last_acquire_failed = state.presentation_waiting || state.blocking_waiting
            }
        }
        let notify = mem::take(&mut state.presentation_waiting);
        drop(state);
        shared.changed.notify_all();
        if notify {
            notify_ready();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc,
        },
        time::Duration,
    };

    const TIMEOUT: Duration = Duration::from_secs(5);

    /// Источник, выдающий drawable только по команде теста: так видно, что главный поток
    /// не ждёт, пока источник заблокирован.
    struct ScriptedSource {
        acquire_started: mpsc::Receiver<()>,
        release: mpsc::Sender<Option<u32>>,
        ready_signals: mpsc::Receiver<()>,
        acquire_calls: Arc<AtomicUsize>,
    }

    fn scripted_prefetcher() -> (DrawablePrefetcher<u32>, ScriptedSource) {
        let (started_sender, acquire_started) = mpsc::channel();
        let (release, release_receiver) = mpsc::channel::<Option<u32>>();
        let (ready_sender, ready_signals) = mpsc::channel();
        let acquire_calls = Arc::new(AtomicUsize::new(0));
        let release_receiver = Mutex::new(release_receiver);
        let prefetcher = DrawablePrefetcher::spawn(
            "test-drawable-prefetch",
            {
                let acquire_calls = acquire_calls.clone();
                move || {
                    acquire_calls.fetch_add(1, Ordering::SeqCst);
                    started_sender.send(()).ok();
                    release_receiver
                        .lock()
                        .ok()?
                        .recv_timeout(TIMEOUT)
                        .ok()
                        .flatten()
                }
            },
            move || {
                ready_sender.send(()).ok();
            },
        )
        .expect("spawn prefetch thread");
        (
            prefetcher,
            ScriptedSource {
                acquire_started,
                release,
                ready_signals,
                acquire_calls,
            },
        )
    }

    impl ScriptedSource {
        fn wait_acquire_started(&self) {
            self.acquire_started
                .recv_timeout(TIMEOUT)
                .expect("source acquisition started");
        }

        fn complete(&self, drawable: Option<u32>) {
            self.release.send(drawable).expect("source is waiting");
        }

        fn wait_ready_signal(&self) {
            self.ready_signals
                .recv_timeout(TIMEOUT)
                .expect("ready signal delivered");
        }

        fn assert_no_ready_signal(&self) {
            assert!(
                self.ready_signals
                    .recv_timeout(Duration::from_millis(50))
                    .is_err(),
                "unexpected ready signal"
            );
        }
    }

    fn wait_until_ready(prefetcher: &DrawablePrefetcher<u32>) {
        let state = prefetcher.shared.lock();
        let (_state, timeout) = prefetcher
            .shared
            .changed
            .wait_timeout_while(state, TIMEOUT, |state| state.ready.is_none())
            .expect("prefetch state lock");
        assert!(!timeout.timed_out(), "drawable was not prefetched");
    }

    /// Первая попытка откладывает показ, источник выдаёт `drawable`, приходит сигнал.
    fn prime(prefetcher: &DrawablePrefetcher<u32>, source: &ScriptedSource, drawable: u32) {
        assert_eq!(prefetcher.try_take(|_| true), DrawableTake::NotReady);
        source.wait_acquire_started();
        source.complete(Some(drawable));
        source.wait_ready_signal();
    }

    #[test]
    fn first_take_requests_drawable_and_signals_when_ready() {
        let (prefetcher, source) = scripted_prefetcher();
        assert!(
            source
                .acquire_started
                .recv_timeout(Duration::from_millis(50))
                .is_err(),
            "source was called before the first take"
        );

        assert_eq!(prefetcher.try_take(|_| true), DrawableTake::NotReady);
        source.wait_acquire_started();
        assert_eq!(prefetcher.try_take(|_| true), DrawableTake::NotReady);

        source.complete(Some(1));
        source.wait_ready_signal();
        assert_eq!(prefetcher.try_take(|_| true), DrawableTake::Ready(1));
    }

    #[test]
    fn prefetched_drawable_is_taken_without_signal_and_next_one_is_requested() {
        let (prefetcher, source) = scripted_prefetcher();
        prime(&prefetcher, &source, 1);
        assert_eq!(prefetcher.try_take(|_| true), DrawableTake::Ready(1));

        source.wait_acquire_started();
        source.complete(Some(2));
        wait_until_ready(&prefetcher);
        source.assert_no_ready_signal();
        assert_eq!(prefetcher.try_take(|_| true), DrawableTake::Ready(2));
        source.wait_acquire_started();
        assert_eq!(source.acquire_calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn at_most_one_drawable_is_held_in_advance() {
        let (prefetcher, source) = scripted_prefetcher();
        prime(&prefetcher, &source, 1);

        assert!(
            source
                .acquire_started
                .recv_timeout(Duration::from_millis(50))
                .is_err(),
            "source acquired a second drawable while one was held"
        );
        assert_eq!(source.acquire_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stale_drawable_is_discarded_and_replaced() {
        let (prefetcher, source) = scripted_prefetcher();
        prime(&prefetcher, &source, 1);

        assert_eq!(
            prefetcher.try_take(|drawable| *drawable == 2),
            DrawableTake::NotReady
        );
        source.wait_acquire_started();
        source.complete(Some(2));
        source.wait_ready_signal();
        assert_eq!(
            prefetcher.try_take(|drawable| *drawable == 2),
            DrawableTake::Ready(2)
        );
    }

    #[test]
    fn failed_acquisition_is_reported_once_without_signal_loop() {
        let (prefetcher, source) = scripted_prefetcher();
        assert_eq!(prefetcher.try_take(|_| true), DrawableTake::NotReady);
        source.wait_acquire_started();
        source.complete(None);
        source.wait_ready_signal();

        assert_eq!(prefetcher.try_take(|_| true), DrawableTake::Unavailable);
        source.wait_acquire_started();
        source.complete(None);
        source.assert_no_ready_signal();

        assert_eq!(prefetcher.try_take(|_| true), DrawableTake::NotReady);
        source.wait_acquire_started();
        source.complete(Some(3));
        source.wait_ready_signal();
        assert_eq!(prefetcher.try_take(|_| true), DrawableTake::Ready(3));
    }

    #[test]
    fn blocking_take_waits_for_usable_drawable() {
        let (prefetcher, source) = scripted_prefetcher();
        prime(&prefetcher, &source, 1);

        thread::scope(|scope| {
            let taken = scope.spawn(|| prefetcher.take_blocking(|drawable| *drawable == 2));
            source.wait_acquire_started();
            source.complete(Some(2));
            assert_eq!(taken.join().expect("blocking take thread"), Some(2));
        });
        source.assert_no_ready_signal();
    }

    #[test]
    fn blocking_take_reports_failed_acquisition() {
        let (prefetcher, source) = scripted_prefetcher();
        thread::scope(|scope| {
            let taken = scope.spawn(|| prefetcher.take_blocking(|_| true));
            source.wait_acquire_started();
            source.complete(None);
            assert_eq!(taken.join().expect("blocking take thread"), None);
        });
    }

    #[test]
    fn drop_releases_held_drawable_and_stops_worker() {
        struct Tracked(Arc<AtomicUsize>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicUsize::new(0));
        let acquired = Arc::new(AtomicUsize::new(0));
        let prefetcher = DrawablePrefetcher::spawn(
            "test-drawable-prefetch",
            {
                let dropped = dropped.clone();
                let acquired = acquired.clone();
                move || {
                    acquired.fetch_add(1, Ordering::SeqCst);
                    Some(Tracked(dropped.clone()))
                }
            },
            || {},
        )
        .expect("spawn prefetch thread");
        assert!(matches!(
            prefetcher.try_take(|_| false),
            DrawableTake::NotReady
        ));
        {
            let state = prefetcher.shared.lock();
            let (_state, timeout) = prefetcher
                .shared
                .changed
                .wait_timeout_while(state, TIMEOUT, |state| state.ready.is_none())
                .expect("prefetch state lock");
            assert!(!timeout.timed_out());
        }
        drop(prefetcher);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        thread::sleep(Duration::from_millis(50));
        assert_eq!(acquired.load(Ordering::SeqCst), 1);
    }
}
