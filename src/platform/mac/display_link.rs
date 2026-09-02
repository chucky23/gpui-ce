use crate::{
    dispatch_get_main_queue,
    dispatch_sys::{
        _dispatch_source_type_data_add, dispatch_resume, dispatch_set_context,
        dispatch_source_cancel, dispatch_source_create, dispatch_source_merge_data,
        dispatch_source_set_event_handler_f, dispatch_source_t, dispatch_suspend,
    },
};
use anyhow::Result;
use core_graphics::display::CGDirectDisplayID;
use std::ffi::c_void;
#[cfg(feature = "frame-trace")]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use util::ResultExt;

#[cfg(feature = "frame-trace")]
struct DisplayLinkCallbackState {
    frame_requests: dispatch_source_t,
    display_id: CGDirectDisplayID,
    snapshot_version: AtomicU64,
    latest_callback_host_time: AtomicU64,
    latest_current_host_time: AtomicU64,
    latest_output_host_time: AtomicU64,
    latest_video_refresh_period: AtomicU64,
    latest_video_time_scale: AtomicU64,
    latest_rate_scalar_bits: AtomicU64,
    latest_current_valid: AtomicBool,
    latest_output_valid: AtomicBool,
    latest_refresh_valid: AtomicBool,
    latest_tick_sequence_id: AtomicU64,
    callback_count: AtomicU64,
}

#[cfg(feature = "frame-trace")]
#[derive(Clone, Copy)]
struct DisplayLinkTraceSnapshot {
    callback_host_time: u64,
    current_host_time_raw: u64,
    output_host_time_raw: u64,
    video_refresh_period: i64,
    video_time_scale: i32,
    rate_scalar: f64,
    current_valid: bool,
    output_valid: bool,
    refresh_valid: bool,
    sequence: u64,
    callback_count: u64,
}

#[cfg(feature = "frame-trace")]
fn frame_trace_delivery(
    display_id: CGDirectDisplayID,
    snapshot: DisplayLinkTraceSnapshot,
    last_delivered_callback_count: u64,
    main_queue_delivery_time_ns: u64,
) -> Option<crate::frame_trace::FrameTraceDisplayTick> {
    if snapshot.callback_count == last_delivered_callback_count {
        return None;
    }
    let refresh_period_ns = snapshot
        .refresh_valid
        .then(|| {
            (snapshot.video_refresh_period as f64 * 1_000_000_000.0
                / snapshot.video_time_scale as f64
                / snapshot.rate_scalar)
                .round() as u64
        })
        .unwrap_or_default();
    let mut flags = 0;
    if !snapshot.current_valid {
        flags |= crate::frame_trace::FLAG_DISPLAY_CURRENT_INVALID;
    }
    if !snapshot.output_valid {
        flags |= crate::frame_trace::FLAG_DISPLAY_TARGET_INVALID;
    }
    if !snapshot.refresh_valid || refresh_period_ns == 0 {
        flags |= crate::frame_trace::FLAG_DISPLAY_REFRESH_INVALID;
    }
    Some(crate::frame_trace::FrameTraceDisplayTick {
        display_id,
        sequence: snapshot.sequence,
        worker_callback_time_ns: crate::frame_trace::mach_ticks_to_ns(snapshot.callback_host_time),
        current_host_time_raw: snapshot.current_host_time_raw,
        output_host_time_raw: snapshot.output_host_time_raw,
        video_refresh_period: snapshot.video_refresh_period,
        video_time_scale: snapshot.video_time_scale,
        rate_scalar: snapshot.rate_scalar,
        refresh_period_ns,
        main_queue_delivery_time_ns,
        coalesced_count: snapshot
            .callback_count
            .saturating_sub(last_delivered_callback_count),
        flags,
    })
}

pub struct DisplayLink {
    display_link: Option<sys::DisplayLink>,
    frame_requests: dispatch_source_t,
    #[cfg(feature = "frame-trace")]
    callback_state: &'static DisplayLinkCallbackState,
    #[cfg(feature = "frame-trace")]
    last_delivered_callback_count: u64,
}

impl DisplayLink {
    pub fn new(
        display_id: CGDirectDisplayID,
        data: *mut c_void,
        callback: unsafe extern "C" fn(*mut c_void),
    ) -> Result<DisplayLink> {
        unsafe extern "C" fn display_link_callback(
            _display_link_out: *mut sys::CVDisplayLink,
            _current_time: *const sys::CVTimeStamp,
            _output_time: *const sys::CVTimeStamp,
            _flags_in: i64,
            _flags_out: *mut i64,
            frame_requests: *mut c_void,
        ) -> i32 {
            unsafe {
                #[cfg(feature = "frame-trace")]
                let frame_requests = {
                    let state = &*(frame_requests as *const DisplayLinkCallbackState);
                    let current_time = _current_time.as_ref();
                    let output_time = _output_time.as_ref();
                    // SAFETY: mach_absolute_time has no preconditions.
                    let callback_host_time = mach2::mach_time::mach_absolute_time();
                    let current_valid = current_time
                        .is_some_and(|time| time.flags & sys::kCVTimeStampHostTimeValid != 0);
                    let output_valid = output_time
                        .is_some_and(|time| time.flags & sys::kCVTimeStampHostTimeValid != 0);
                    let refresh_valid = output_time.is_some_and(|time| {
                        time.flags & sys::kCVTimeStampVideoRefreshPeriodValid != 0
                            && time.flags & sys::kCVTimeStampRateScalarValid != 0
                            && time.video_refresh_period > 0
                            && time.video_time_scale > 0
                            && time.rate_scalar.is_finite()
                            && time.rate_scalar > 0.0
                    });
                    // CoreVideo invokes one DisplayLink's output callback serially. This
                    // single-writer seqlock publishes a coherent payload to the main thread.
                    state.snapshot_version.fetch_add(1, Ordering::SeqCst);
                    state
                        .latest_callback_host_time
                        .store(callback_host_time, Ordering::SeqCst);
                    state.latest_current_host_time.store(
                        current_time.map_or(0, |time| time.host_time),
                        Ordering::SeqCst,
                    );
                    state.latest_output_host_time.store(
                        output_time.map_or(0, |time| time.host_time),
                        Ordering::SeqCst,
                    );
                    state.latest_video_refresh_period.store(
                        output_time.map_or(0, |time| time.video_refresh_period as u64),
                        Ordering::SeqCst,
                    );
                    state.latest_video_time_scale.store(
                        output_time.map_or(0, |time| time.video_time_scale as u64),
                        Ordering::SeqCst,
                    );
                    state.latest_rate_scalar_bits.store(
                        output_time.map_or(0, |time| time.rate_scalar.to_bits()),
                        Ordering::SeqCst,
                    );
                    state
                        .latest_current_valid
                        .store(current_valid, Ordering::SeqCst);
                    state
                        .latest_output_valid
                        .store(output_valid, Ordering::SeqCst);
                    state
                        .latest_refresh_valid
                        .store(refresh_valid, Ordering::SeqCst);
                    state.latest_tick_sequence_id.store(
                        crate::frame_trace::next_display_tick_sequence_id(),
                        Ordering::SeqCst,
                    );
                    state.callback_count.fetch_add(1, Ordering::SeqCst);
                    state.snapshot_version.fetch_add(1, Ordering::SeqCst);
                    state.frame_requests
                };
                #[cfg(not(feature = "frame-trace"))]
                let frame_requests = frame_requests as dispatch_source_t;
                dispatch_source_merge_data(frame_requests, 1);
                0
            }
        }

        unsafe {
            let frame_requests = dispatch_source_create(
                &_dispatch_source_type_data_add,
                0,
                0,
                dispatch_get_main_queue(),
            );
            dispatch_set_context(
                crate::dispatch_sys::dispatch_object_t {
                    _ds: frame_requests,
                },
                data,
            );
            dispatch_source_set_event_handler_f(frame_requests, Some(callback));

            #[cfg(feature = "frame-trace")]
            // DisplayLink itself is intentionally leaked on drop because CoreVideo may still
            // access it from its worker thread. Its callback context must have the same lifetime.
            let callback_state = Box::leak(Box::new(DisplayLinkCallbackState {
                frame_requests,
                display_id,
                snapshot_version: AtomicU64::new(0),
                latest_callback_host_time: AtomicU64::new(0),
                latest_current_host_time: AtomicU64::new(0),
                latest_output_host_time: AtomicU64::new(0),
                latest_video_refresh_period: AtomicU64::new(0),
                latest_video_time_scale: AtomicU64::new(0),
                latest_rate_scalar_bits: AtomicU64::new(0),
                latest_current_valid: AtomicBool::new(false),
                latest_output_valid: AtomicBool::new(false),
                latest_refresh_valid: AtomicBool::new(false),
                latest_tick_sequence_id: AtomicU64::new(0),
                callback_count: AtomicU64::new(0),
            }));
            #[cfg(feature = "frame-trace")]
            let display_link_context = callback_state as *const _ as *mut c_void;
            #[cfg(not(feature = "frame-trace"))]
            let display_link_context = frame_requests as *mut c_void;

            let display_link =
                sys::DisplayLink::new(display_id, display_link_callback, display_link_context)?;

            Ok(Self {
                display_link: Some(display_link),
                frame_requests,
                #[cfg(feature = "frame-trace")]
                callback_state,
                #[cfg(feature = "frame-trace")]
                last_delivered_callback_count: 0,
            })
        }
    }

    pub fn start(&mut self) -> Result<()> {
        unsafe {
            dispatch_resume(crate::dispatch_sys::dispatch_object_t {
                _ds: self.frame_requests,
            });
            self.display_link.as_mut().unwrap().start()?;
        }
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        unsafe {
            dispatch_suspend(crate::dispatch_sys::dispatch_object_t {
                _ds: self.frame_requests,
            });
            self.display_link.as_mut().unwrap().stop()?;
        }
        Ok(())
    }

    #[cfg(feature = "frame-trace")]
    pub(crate) fn take_trace_delivery(
        &mut self,
    ) -> Option<crate::frame_trace::FrameTraceDisplayTick> {
        let version_before = self.callback_state.snapshot_version.load(Ordering::SeqCst);
        if version_before & 1 != 0 {
            return None;
        }
        let callback_host_time = self
            .callback_state
            .latest_callback_host_time
            .load(Ordering::SeqCst);
        let current_host_time_raw = self
            .callback_state
            .latest_current_host_time
            .load(Ordering::SeqCst);
        let output_host_time_raw = self
            .callback_state
            .latest_output_host_time
            .load(Ordering::SeqCst);
        let video_refresh_period = self
            .callback_state
            .latest_video_refresh_period
            .load(Ordering::SeqCst) as i64;
        let video_time_scale = self
            .callback_state
            .latest_video_time_scale
            .load(Ordering::SeqCst) as i32;
        let rate_scalar = f64::from_bits(
            self.callback_state
                .latest_rate_scalar_bits
                .load(Ordering::SeqCst),
        );
        let current_valid = self
            .callback_state
            .latest_current_valid
            .load(Ordering::SeqCst);
        let output_valid = self
            .callback_state
            .latest_output_valid
            .load(Ordering::SeqCst);
        let refresh_valid = self
            .callback_state
            .latest_refresh_valid
            .load(Ordering::SeqCst);
        let sequence = self
            .callback_state
            .latest_tick_sequence_id
            .load(Ordering::SeqCst);
        let callback_count = self.callback_state.callback_count.load(Ordering::SeqCst);
        let version_after = self.callback_state.snapshot_version.load(Ordering::SeqCst);
        if version_before != version_after {
            return None;
        }
        let snapshot = DisplayLinkTraceSnapshot {
            callback_host_time,
            current_host_time_raw,
            output_host_time_raw,
            video_refresh_period,
            video_time_scale,
            rate_scalar,
            current_valid,
            output_valid,
            refresh_valid,
            sequence,
            callback_count,
        };
        let tick = frame_trace_delivery(
            self.callback_state.display_id,
            snapshot,
            self.last_delivered_callback_count,
            crate::frame_trace::monotonic_time_ns(),
        )?;
        self.last_delivered_callback_count = callback_count;
        Some(tick)
    }
}

impl Drop for DisplayLink {
    fn drop(&mut self) {
        self.stop().log_err();
        // We see occasional segfaults on the CVDisplayLink thread.
        //
        // It seems possible that this happens because CVDisplayLinkRelease releases the CVDisplayLink
        // on the main thread immediately, but the background thread that CVDisplayLink uses for timers
        // is still accessing it.
        //
        // We might also want to upgrade to CADisplayLink, but that requires dropping old macOS support.
        std::mem::forget(self.display_link.take());
        unsafe {
            dispatch_source_cancel(self.frame_requests);
        }
    }
}

#[cfg(all(test, feature = "frame-trace"))]
mod frame_trace_tests {
    use super::*;

    fn snapshot(callback_count: u64) -> DisplayLinkTraceSnapshot {
        DisplayLinkTraceSnapshot {
            callback_host_time: 10,
            current_host_time_raw: 20,
            output_host_time_raw: 30,
            video_refresh_period: 1,
            video_time_scale: 60,
            rate_scalar: 1.0,
            current_valid: true,
            output_valid: true,
            refresh_valid: true,
            sequence: 40,
            callback_count,
        }
    }

    #[test]
    fn frame_trace_delivery_reports_coalesced_callbacks_without_reusing_state() {
        let tick = frame_trace_delivery(7, snapshot(4), 1, 50).unwrap();
        assert_eq!(tick.display_id, 7);
        assert_eq!(tick.sequence, 40);
        assert_eq!(tick.coalesced_count, 3);
        assert_eq!(tick.main_queue_delivery_time_ns, 50);
        assert_eq!(tick.refresh_period_ns, 16_666_667);
        assert!(frame_trace_delivery(7, snapshot(4), 4, 51).is_none());
    }

    #[test]
    fn frame_trace_delivery_marks_current_target_and_refresh_independently_invalid() {
        let mut invalid = snapshot(1);
        invalid.current_valid = false;
        invalid.output_valid = false;
        invalid.refresh_valid = false;
        let tick = frame_trace_delivery(9, invalid, 0, 60).unwrap();
        assert_ne!(
            tick.flags & crate::frame_trace::FLAG_DISPLAY_CURRENT_INVALID,
            0
        );
        assert_ne!(
            tick.flags & crate::frame_trace::FLAG_DISPLAY_TARGET_INVALID,
            0
        );
        assert_ne!(
            tick.flags & crate::frame_trace::FLAG_DISPLAY_REFRESH_INVALID,
            0
        );
        assert_eq!(tick.refresh_period_ns, 0);
        assert_eq!(tick.current_host_time_raw, 20);
        assert_eq!(tick.output_host_time_raw, 30);
    }
}

mod sys {
    //! Derived from display-link crate under the following license:
    //! <https://github.com/BrainiumLLC/display-link/blob/master/LICENSE-MIT>
    //! Apple docs: [CVDisplayLink](https://developer.apple.com/documentation/corevideo/cvdisplaylinkoutputcallback?language=objc)
    #![allow(dead_code, non_upper_case_globals)]

    use anyhow::Result;
    use core_graphics::display::CGDirectDisplayID;
    use foreign_types::{ForeignType, foreign_type};
    use std::{
        ffi::c_void,
        fmt::{self, Debug, Formatter},
    };

    #[derive(Debug)]
    pub enum CVDisplayLink {}

    foreign_type! {
        pub unsafe type DisplayLink {
            type CType = CVDisplayLink;
            fn drop = CVDisplayLinkRelease;
            fn clone = CVDisplayLinkRetain;
        }
    }

    impl Debug for DisplayLink {
        fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
            formatter
                .debug_tuple("DisplayLink")
                .field(&self.as_ptr())
                .finish()
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(crate) struct CVTimeStamp {
        pub version: u32,
        pub video_time_scale: i32,
        pub video_time: i64,
        pub host_time: u64,
        pub rate_scalar: f64,
        pub video_refresh_period: i64,
        pub smpte_time: CVSMPTETime,
        pub flags: u64,
        pub reserved: u64,
    }

    pub type CVTimeStampFlags = u64;

    pub const kCVTimeStampVideoTimeValid: CVTimeStampFlags = 1 << 0;
    pub const kCVTimeStampHostTimeValid: CVTimeStampFlags = 1 << 1;
    pub const kCVTimeStampSMPTETimeValid: CVTimeStampFlags = 1 << 2;
    pub const kCVTimeStampVideoRefreshPeriodValid: CVTimeStampFlags = 1 << 3;
    pub const kCVTimeStampRateScalarValid: CVTimeStampFlags = 1 << 4;
    pub const kCVTimeStampTopField: CVTimeStampFlags = 1 << 16;
    pub const kCVTimeStampBottomField: CVTimeStampFlags = 1 << 17;
    pub const kCVTimeStampVideoHostTimeValid: CVTimeStampFlags =
        kCVTimeStampVideoTimeValid | kCVTimeStampHostTimeValid;
    pub const kCVTimeStampIsInterlaced: CVTimeStampFlags =
        kCVTimeStampTopField | kCVTimeStampBottomField;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub(crate) struct CVSMPTETime {
        pub subframes: i16,
        pub subframe_divisor: i16,
        pub counter: u32,
        pub time_type: u32,
        pub flags: u32,
        pub hours: i16,
        pub minutes: i16,
        pub seconds: i16,
        pub frames: i16,
    }

    pub type CVSMPTETimeType = u32;

    pub const kCVSMPTETimeType24: CVSMPTETimeType = 0;
    pub const kCVSMPTETimeType25: CVSMPTETimeType = 1;
    pub const kCVSMPTETimeType30Drop: CVSMPTETimeType = 2;
    pub const kCVSMPTETimeType30: CVSMPTETimeType = 3;
    pub const kCVSMPTETimeType2997: CVSMPTETimeType = 4;
    pub const kCVSMPTETimeType2997Drop: CVSMPTETimeType = 5;
    pub const kCVSMPTETimeType60: CVSMPTETimeType = 6;
    pub const kCVSMPTETimeType5994: CVSMPTETimeType = 7;

    pub type CVSMPTETimeFlags = u32;

    pub const kCVSMPTETimeValid: CVSMPTETimeFlags = 1 << 0;
    pub const kCVSMPTETimeRunning: CVSMPTETimeFlags = 1 << 1;

    pub type CVDisplayLinkOutputCallback = unsafe extern "C" fn(
        display_link_out: *mut CVDisplayLink,
        // A pointer to the current timestamp. This represents the timestamp when the callback is called.
        current_time: *const CVTimeStamp,
        // A pointer to the output timestamp. This represents the timestamp for when the frame will be displayed.
        output_time: *const CVTimeStamp,
        // Unused
        flags_in: i64,
        // Unused
        flags_out: *mut i64,
        // A pointer to app-defined data.
        display_link_context: *mut c_void,
    ) -> i32;

    #[link(name = "CoreFoundation", kind = "framework")]
    #[link(name = "CoreVideo", kind = "framework")]
    #[allow(improper_ctypes, unknown_lints, clippy::duplicated_attributes)]
    unsafe extern "C" {
        pub fn CVDisplayLinkCreateWithActiveCGDisplays(
            display_link_out: *mut *mut CVDisplayLink,
        ) -> i32;
        pub fn CVDisplayLinkSetCurrentCGDisplay(
            display_link: &mut DisplayLinkRef,
            display_id: u32,
        ) -> i32;
        pub fn CVDisplayLinkSetOutputCallback(
            display_link: &mut DisplayLinkRef,
            callback: CVDisplayLinkOutputCallback,
            user_info: *mut c_void,
        ) -> i32;
        pub fn CVDisplayLinkStart(display_link: &mut DisplayLinkRef) -> i32;
        pub fn CVDisplayLinkStop(display_link: &mut DisplayLinkRef) -> i32;
        pub fn CVDisplayLinkRelease(display_link: *mut CVDisplayLink);
        pub fn CVDisplayLinkRetain(display_link: *mut CVDisplayLink) -> *mut CVDisplayLink;
    }

    impl DisplayLink {
        /// Apple docs: [CVDisplayLinkCreateWithCGDisplay](https://developer.apple.com/documentation/corevideo/1456981-cvdisplaylinkcreatewithcgdisplay?language=objc)
        pub unsafe fn new(
            display_id: CGDirectDisplayID,
            callback: CVDisplayLinkOutputCallback,
            user_info: *mut c_void,
        ) -> Result<Self> {
            unsafe {
                let mut display_link: *mut CVDisplayLink = 0 as _;

                let code = CVDisplayLinkCreateWithActiveCGDisplays(&mut display_link);
                anyhow::ensure!(code == 0, "could not create display link, code: {}", code);

                let mut display_link = DisplayLink::from_ptr(display_link);

                let code = CVDisplayLinkSetOutputCallback(&mut display_link, callback, user_info);
                anyhow::ensure!(code == 0, "could not set output callback, code: {}", code);

                let code = CVDisplayLinkSetCurrentCGDisplay(&mut display_link, display_id);
                anyhow::ensure!(
                    code == 0,
                    "could not assign display to display link, code: {}",
                    code
                );

                Ok(display_link)
            }
        }
    }

    impl DisplayLinkRef {
        /// Apple docs: [CVDisplayLinkStart](https://developer.apple.com/documentation/corevideo/1457193-cvdisplaylinkstart?language=objc)
        pub unsafe fn start(&mut self) -> Result<()> {
            unsafe {
                let code = CVDisplayLinkStart(self);
                anyhow::ensure!(code == 0, "could not start display link, code: {}", code);
                Ok(())
            }
        }

        /// Apple docs: [CVDisplayLinkStop](https://developer.apple.com/documentation/corevideo/1457281-cvdisplaylinkstop?language=objc)
        pub unsafe fn stop(&mut self) -> Result<()> {
            unsafe {
                let code = CVDisplayLinkStop(self);
                anyhow::ensure!(code == 0, "could not stop display link, code: {}", code);
                Ok(())
            }
        }
    }
}
