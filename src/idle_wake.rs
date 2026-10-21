//! Удержание дисплея от пониженной частоты после ввода: опыт для ProMotion на macOS.
//!
//! Без новых кадров WindowServer опускает частоту дисплея. Первые кадры после такого покоя
//! Metal отбрасывает (`presentedTime` = 0), и первый показ опаздывает на 25–50 мс. Режимы
//! включаются переменной `GPUI_IDLE_WAKE`, без неё поведение прежнее:
//!
//! * `link` — на macOS 14+ после ввода работает `NSView.displayLink` с
//!   `preferredFrameRateRange` 120 Гц, кадров он не рисует, только просит частоту;
//! * `heartbeat` — после секунды обычного удержания показ текущей сцены повторяется раз в
//!   `GPUI_IDLE_WAKE_HEARTBEAT_MS` (по умолчанию 100 мс).
//!
//! Оба режима действуют `GPUI_IDLE_WAKE_HOLD_MS` после последнего ввода (по умолчанию 10 000 мс)
//! и сочетаются через запятую: `link,heartbeat`.

use std::{sync::LazyLock, time::Duration};

pub(crate) const IDLE_WAKE_ENV: &str = "GPUI_IDLE_WAKE";
pub(crate) const IDLE_WAKE_HOLD_ENV: &str = "GPUI_IDLE_WAKE_HOLD_MS";
pub(crate) const IDLE_WAKE_HEARTBEAT_ENV: &str = "GPUI_IDLE_WAKE_HEARTBEAT_MS";

/// Обычное удержание показа после ввода, действующее и без режимов.
pub(crate) const KEEPALIVE_AFTER_INPUT: Duration = Duration::from_secs(1);
const DEFAULT_HOLD: Duration = Duration::from_secs(10);
const DEFAULT_HEARTBEAT: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct IdleWakeConfig {
    /// Просить у системы частоту дисплея через `NSView.displayLink` (macOS 14+).
    pub link: bool,
    /// Период повторного показа текущей сцены после секунды обычного удержания.
    pub heartbeat: Option<Duration>,
    /// Сколько после последнего ввода действуют включённые режимы.
    pub hold: Duration,
}

impl IdleWakeConfig {
    pub(crate) fn global() -> IdleWakeConfig {
        static CONFIG: LazyLock<IdleWakeConfig> = LazyLock::new(|| {
            let mode = std::env::var(IDLE_WAKE_ENV).ok();
            let hold = std::env::var(IDLE_WAKE_HOLD_ENV).ok();
            let heartbeat = std::env::var(IDLE_WAKE_HEARTBEAT_ENV).ok();
            match IdleWakeConfig::parse(mode.as_deref(), hold.as_deref(), heartbeat.as_deref()) {
                Ok(config) => {
                    if config.is_enabled() {
                        log::info!("idle wake: {config:?}");
                    }
                    config
                }
                Err(error) => {
                    // Опечатка не должна молча менять замер: режим выключен, причина в логе.
                    log::error!("idle wake disabled: {error}");
                    IdleWakeConfig::default()
                }
            }
        });
        *CONFIG
    }

    pub(crate) fn parse(
        mode: Option<&str>,
        hold_ms: Option<&str>,
        heartbeat_ms: Option<&str>,
    ) -> Result<IdleWakeConfig, String> {
        let mut link = false;
        let mut heartbeat = false;
        for token in mode.unwrap_or_default().split(',').map(str::trim) {
            match token {
                "" | "off" => {}
                "link" => link = true,
                "heartbeat" => heartbeat = true,
                other => return Err(format!("{IDLE_WAKE_ENV}: unknown mode {other:?}")),
            }
        }
        let hold = parse_positive_ms(IDLE_WAKE_HOLD_ENV, hold_ms)?.unwrap_or(DEFAULT_HOLD);
        let heartbeat_period =
            parse_positive_ms(IDLE_WAKE_HEARTBEAT_ENV, heartbeat_ms)?.unwrap_or(DEFAULT_HEARTBEAT);
        Ok(IdleWakeConfig {
            link,
            heartbeat: heartbeat.then_some(heartbeat_period),
            hold: if link || heartbeat {
                hold
            } else {
                Duration::ZERO
            },
        })
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.link || self.heartbeat.is_some()
    }

    /// Нужен ли на этом такте повтор показа ради удержания частоты: секунду после ввода на
    /// каждом такте, а с `heartbeat` ещё и редкими повторами до конца `hold`. Пропустить ли
    /// повтор, который ждал бы drawable, решает `KeepalivePresentMode`.
    pub(crate) fn keepalive_due(&self, since_input: Duration, since_present: Duration) -> bool {
        since_input < KEEPALIVE_AFTER_INPUT || self.heartbeat_due(since_input, since_present)
    }

    /// Пора ли повторить показ текущей сцены, чтобы дисплей не ушёл в пониженную частоту.
    pub(crate) fn heartbeat_due(&self, since_input: Duration, since_present: Duration) -> bool {
        self.heartbeat
            .is_some_and(|period| since_input < self.hold && since_present >= period)
    }
}

fn parse_positive_ms(name: &str, value: Option<&str>) -> Result<Option<Duration>, String> {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    match value.parse::<u64>() {
        Ok(milliseconds) if milliseconds > 0 => Ok(Some(Duration::from_millis(milliseconds))),
        _ => Err(format!(
            "{name}: expected a positive number of milliseconds, got {value:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_variables_keep_previous_behavior() {
        let config = IdleWakeConfig::parse(None, None, None).unwrap();
        assert_eq!(config, IdleWakeConfig::default());
        assert!(!config.is_enabled());
        assert!(!config.heartbeat_due(Duration::from_millis(1500), Duration::from_secs(5)));
        let off = IdleWakeConfig::parse(Some("off"), Some("5000"), Some("50")).unwrap();
        assert_eq!(off, IdleWakeConfig::default());
    }

    #[test]
    fn modes_combine_and_take_defaults() {
        let link = IdleWakeConfig::parse(Some("link"), None, None).unwrap();
        assert!(link.link);
        assert_eq!(link.heartbeat, None);
        assert_eq!(link.hold, DEFAULT_HOLD);

        let both =
            IdleWakeConfig::parse(Some(" link , heartbeat "), Some("3000"), Some("40")).unwrap();
        assert!(both.link);
        assert_eq!(both.heartbeat, Some(Duration::from_millis(40)));
        assert_eq!(both.hold, Duration::from_millis(3000));
    }

    #[test]
    fn invalid_values_are_rejected() {
        assert!(IdleWakeConfig::parse(Some("links"), None, None).is_err());
        assert!(IdleWakeConfig::parse(Some("link"), Some("0"), None).is_err());
        assert!(IdleWakeConfig::parse(Some("heartbeat"), None, Some("-5")).is_err());
        assert!(IdleWakeConfig::parse(Some("heartbeat"), None, Some("1.5")).is_err());
    }

    #[test]
    fn heartbeat_repeats_only_inside_hold_after_period() {
        let config = IdleWakeConfig::parse(Some("heartbeat"), Some("2000"), Some("100")).unwrap();
        assert!(!config.heartbeat_due(Duration::from_millis(1200), Duration::from_millis(99)));
        assert!(config.heartbeat_due(Duration::from_millis(1200), Duration::from_millis(100)));
        assert!(config.heartbeat_due(Duration::from_millis(1999), Duration::from_millis(250)));
        assert!(!config.heartbeat_due(Duration::from_millis(2000), Duration::from_millis(250)));
        let link_only = IdleWakeConfig::parse(Some("link"), Some("2000"), None).unwrap();
        assert!(!link_only.heartbeat_due(Duration::from_millis(1200), Duration::from_secs(1)));
    }

    #[test]
    fn without_variables_keepalive_is_one_second_after_input() {
        // Без переменных удержание то же, что до режимов: только секунда после ввода.
        let config = IdleWakeConfig::parse(None, None, None).unwrap();
        for since_present in [Duration::ZERO, Duration::from_millis(100), Duration::from_secs(5)] {
            assert!(config.keepalive_due(Duration::ZERO, since_present));
            assert!(config.keepalive_due(Duration::from_millis(999), since_present));
            assert!(!config.keepalive_due(Duration::from_secs(1), since_present));
            assert!(!config.keepalive_due(Duration::from_secs(3), since_present));
        }
        let link = IdleWakeConfig::parse(Some("link"), None, None).unwrap();
        assert!(!link.keepalive_due(Duration::from_secs(3), Duration::from_secs(1)));
    }

    #[test]
    fn heartbeat_extends_keepalive_but_never_waits_for_drawable() {
        use crate::keepalive_present::KeepalivePresentMode;
        let config = IdleWakeConfig::parse(Some("heartbeat"), None, None).unwrap();
        let since_input = Duration::from_millis(1500);
        assert!(!config.keepalive_due(since_input, Duration::from_millis(50)));
        let keepalive = config.keepalive_due(since_input, Duration::from_millis(100));
        assert!(keepalive);
        // Повтор heartbeat при режиме по умолчанию пропускается, если ждал бы drawable.
        let mode = KeepalivePresentMode::parse(None).unwrap();
        assert!(mode.should_present(false, keepalive, false));
        assert!(!mode.should_present(false, keepalive, true));
        assert!(!config.keepalive_due(Duration::from_secs(10), Duration::from_secs(1)));
    }
}
