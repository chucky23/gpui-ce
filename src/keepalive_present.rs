//! Удержание показа: секунду после ввода окно показывает текущую сцену на каждом такте, чтобы
//! дисплей не снижал частоту.
//!
//! При двух drawable корневого слоя такой повтор почти всегда ждёт в `nextDrawable`, пока
//! прошлый показ не сменит кадр на экране, то есть до 8 мс главного потока на такте 8,3 мс.
//! Ввод, который AppKit ещё не передал окну, ждёт за этим повтором, и окно на этом такте не
//! грязное. Когда главный поток занят без пауз, каждый следующий такт DisplayLink уже стоит в
//! очереди к концу предыдущего, и порядок «такт, ввод» закрепляется: новая сцена строится через
//! такт, а промежуточный уходит на ожидание drawable ради старой сцены (режим полу-частоты).
//! Повтор, который ждал бы drawable, ничего не даёт дисплею: в очереди показа уже стоит кадр,
//! и дисплей получит его на ближайшем обновлении. Поэтому по умолчанию такой повтор
//! пропускается.
//!
//! Переменная `GPUI_KEEPALIVE_PRESENT` оставлена для сравнения:
//!
//! * `nonblocking` или без переменной — повтор пропускается, если свободного drawable сейчас нет;
//! * `block` — прежнее поведение: повтор ждёт свободный drawable.
//!
//! Режим касается только повтора ради удержания. Показ новой или ещё не показанной сцены и
//! повтор кадра, отброшенного Metal, идут прежним путём.

use std::sync::LazyLock;

pub(crate) const KEEPALIVE_PRESENT_ENV: &str = "GPUI_KEEPALIVE_PRESENT";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum KeepalivePresentMode {
    /// Повтор ради удержания ждёт свободный drawable. Прежнее поведение, только для сравнения.
    Block,
    /// Повтор ради удержания пропускается, если показ ждал бы свободный drawable.
    #[default]
    NonBlocking,
}

impl KeepalivePresentMode {
    pub(crate) fn global() -> KeepalivePresentMode {
        static MODE: LazyLock<KeepalivePresentMode> = LazyLock::new(|| {
            let value = std::env::var(KEEPALIVE_PRESENT_ENV).ok();
            match KeepalivePresentMode::parse(value.as_deref()) {
                Ok(mode) => {
                    if mode != KeepalivePresentMode::default() {
                        log::info!("keepalive present: {mode:?}");
                    }
                    mode
                }
                Err(error) => {
                    // Опечатка не должна молча менять замер: режим по умолчанию, причина в логе.
                    let mode = KeepalivePresentMode::default();
                    log::error!("keepalive present mode reset to {mode:?}: {error}");
                    mode
                }
            }
        });
        *MODE
    }

    pub(crate) fn parse(value: Option<&str>) -> Result<KeepalivePresentMode, String> {
        match value.map(str::trim).unwrap_or_default() {
            "" => Ok(KeepalivePresentMode::default()),
            "block" => Ok(KeepalivePresentMode::Block),
            "nonblocking" => Ok(KeepalivePresentMode::NonBlocking),
            other => Err(format!("{KEEPALIVE_PRESENT_ENV}: unknown mode {other:?}")),
        }
    }

    /// Показать ли текущую сцену без перестройки. `required` — сцена ещё не показана или Metal
    /// отбросил её последний кадр. `keepalive` — идёт удержание показа после ввода.
    /// `presentation_would_block` — показ сейчас ждал бы свободный drawable.
    pub(crate) fn should_present(
        self,
        required: bool,
        keepalive: bool,
        presentation_would_block: bool,
    ) -> bool {
        required
            || (keepalive
                && !(self == KeepalivePresentMode::NonBlocking && presentation_would_block))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_keepalive_never_waits_for_drawable() {
        // Без переменной повтор старой сцены не должен занимать главный поток ожиданием
        // drawable: иначе ввод ждёт за ним и новая сцена строится через такт.
        for value in [None, Some(""), Some("  ")] {
            let mode = KeepalivePresentMode::parse(value).unwrap();
            assert_eq!(mode, KeepalivePresentMode::NonBlocking);
            assert!(!mode.should_present(false, true, true));
            assert!(mode.should_present(false, true, false));
        }
        assert_eq!(
            KeepalivePresentMode::default(),
            KeepalivePresentMode::NonBlocking
        );
    }

    #[test]
    fn block_keeps_previous_behavior_for_comparison() {
        assert_eq!(
            KeepalivePresentMode::parse(Some(" block ")),
            Ok(KeepalivePresentMode::Block)
        );
        let mode = KeepalivePresentMode::Block;
        for required in [false, true] {
            for keepalive in [false, true] {
                for would_block in [false, true] {
                    assert_eq!(
                        mode.should_present(required, keepalive, would_block),
                        required || keepalive,
                    );
                }
            }
        }
    }

    #[test]
    fn invalid_mode_is_rejected() {
        assert!(KeepalivePresentMode::parse(Some("non-blocking")).is_err());
        assert!(KeepalivePresentMode::parse(Some("1")).is_err());
    }

    #[test]
    fn nonblocking_skips_only_keepalive_that_would_wait() {
        let mode = KeepalivePresentMode::parse(Some("nonblocking")).unwrap();
        assert_eq!(mode, KeepalivePresentMode::NonBlocking);
        assert!(!mode.should_present(false, true, true));
        assert!(mode.should_present(false, true, false));
        assert!(!mode.should_present(false, false, false));
        // Непоказанная сцена и отброшенный кадр показываются, даже если придётся ждать.
        assert!(mode.should_present(true, true, true));
        assert!(mode.should_present(true, false, true));
    }
}
