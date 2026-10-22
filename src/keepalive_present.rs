//! Удержание показа: секунду после ввода окно показывает текущую сцену на каждом такте, чтобы
//! дисплей не снижал частоту.
//!
//! При двух drawable корневого слоя такой повтор почти всегда ждёт в `nextDrawable`, пока
//! прошлый показ не сменит кадр на экране, то есть 4–8 мс главного потока на каждом такте.
//! Работа, которая строит новую сцену после действия пользователя, стоит за этим ожиданием.
//! Режим задаёт переменная `GPUI_KEEPALIVE_PRESENT`:
//!
//! * `block` или без переменной — прежнее поведение: повтор ждёт свободный drawable;
//! * `nonblocking` — повтор пропускается, если свободного drawable сейчас нет. В очереди
//!   показа тогда уже стоит кадр, и дисплей получит его на ближайшем обновлении.
//!
//! Режим касается только повтора ради удержания. Показ новой или ещё не показанной сцены и
//! повтор кадра, отброшенного Metal, идут прежним путём.

use std::sync::LazyLock;

pub(crate) const KEEPALIVE_PRESENT_ENV: &str = "GPUI_KEEPALIVE_PRESENT";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum KeepalivePresentMode {
    /// Повтор ради удержания ждёт свободный drawable.
    #[default]
    Block,
    /// Повтор ради удержания пропускается, если показ ждал бы свободный drawable.
    NonBlocking,
}

impl KeepalivePresentMode {
    pub(crate) fn global() -> KeepalivePresentMode {
        static MODE: LazyLock<KeepalivePresentMode> = LazyLock::new(|| {
            let value = std::env::var(KEEPALIVE_PRESENT_ENV).ok();
            match KeepalivePresentMode::parse(value.as_deref()) {
                Ok(mode) => {
                    if mode != KeepalivePresentMode::Block {
                        log::info!("keepalive present: {mode:?}");
                    }
                    mode
                }
                Err(error) => {
                    // Опечатка не должна молча менять замер: режим прежний, причина в логе.
                    log::error!("keepalive present mode reset to block: {error}");
                    KeepalivePresentMode::Block
                }
            }
        });
        *MODE
    }

    pub(crate) fn parse(value: Option<&str>) -> Result<KeepalivePresentMode, String> {
        match value.map(str::trim).unwrap_or_default() {
            "" | "block" => Ok(KeepalivePresentMode::Block),
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
    fn unset_or_block_keeps_previous_behavior() {
        assert_eq!(
            KeepalivePresentMode::parse(None),
            Ok(KeepalivePresentMode::Block)
        );
        assert_eq!(
            KeepalivePresentMode::parse(Some(" block ")),
            Ok(KeepalivePresentMode::Block)
        );
        assert_eq!(
            KeepalivePresentMode::parse(Some("")),
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
