//! Bounded, allocation-free recording for input-to-presentation investigations.

use serde::Serialize;
use std::{
    cell::{Cell, UnsafeCell},
    hint::spin_loop,
    marker::PhantomData,
    mem::MaybeUninit,
    rc::Rc,
    sync::{
        OnceLock,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
};

const WRITING_SEQUENCE_BIT: u64 = 1 << 63;
const SEQUENCE_MASK: u64 = !WRITING_SEQUENCE_BIT;
const WRITER_STOP_BIT: usize = 1 << (usize::BITS - 1);
const WRITER_COUNT_MASK: usize = !WRITER_STOP_BIT;

/// Input originated from an AppKit event with a valid native timestamp.
pub const FLAG_PHYSICAL_INPUT: u64 = 1 << 0;
/// Input was produced by a deterministic in-process driver.
pub const FLAG_SYNTHETIC_INPUT: u64 = 1 << 1;
/// A command buffer was submitted after its current DisplayLink target.
pub const FLAG_MISSED_DISPLAY_TARGET: u64 = 1 << 2;
/// Metal сообщил, что drawable не был показан или соответствующий кадр был отброшен.
pub const FLAG_PRESENTATION_TIMESTAMP_INVALID: u64 = 1 << 3;
/// The submitted frame reuses a scene built by an earlier logical frame.
pub const FLAG_REUSED_SCENE: u64 = 1 << 4;
/// A DisplayLink callback did not provide a valid target host time.
pub const FLAG_DISPLAY_TARGET_INVALID: u64 = 1 << 5;
/// Metal did not report a valid absolute GPU execution interval.
pub const FLAG_GPU_TIMESTAMPS_INVALID: u64 = 1 << 6;
/// A presentation callback observed an already-empty diagnostic queue counter.
pub const FLAG_QUEUE_DEPTH_UNDERFLOW: u64 = 1 << 7;
/// The drawable passed to the presentation callback differs from the submitted drawable.
pub const FLAG_DRAWABLE_ID_MISMATCH: u64 = 1 << 8;
/// The drawable was presented more than one millisecond after its DisplayLink target.
pub const FLAG_PRESENTED_AFTER_DISPLAY_TARGET: u64 = 1 << 9;
/// An invalidation request found the window already dirty and was coalesced.
pub const FLAG_INVALIDATION_ALREADY_DIRTY: u64 = 1 << 10;
/// An invalidation request arrived while GPUI was already drawing the window.
pub const FLAG_INVALIDATION_DURING_DRAW: u64 = 1 << 11;

static TRACE: OnceLock<FrameTraceBuffer> = OnceLock::new();
static NEXT_INPUT_SEQUENCE_ID: AtomicU64 = AtomicU64::new(0);
static LATEST_INPUT_SEQUENCE_ID: AtomicU64 = AtomicU64::new(0);
static NEXT_LOGICAL_FRAME_ID: AtomicU64 = AtomicU64::new(0);
static NEXT_RENDERER_INSTANCE_ID: AtomicU64 = AtomicU64::new(0);
static NEXT_DISPLAY_TICK_SEQUENCE_ID: AtomicU64 = AtomicU64::new(0);
static LATEST_DISPLAY_TARGET_NS: AtomicU64 = AtomicU64::new(0);
static LATEST_DISPLAY_TICK_SEQUENCE: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static CURRENT_APPKIT_INPUT_SEQUENCE_ID: Cell<u64> = const { Cell::new(0) };
}

/// One low-cardinality event in the input-to-presentation trace.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameTraceEventKind {
    /// The bounded trace was enabled outside the measured frame.
    TraceStarted,
    /// The authoritative workload measurement window began.
    MeasurementWindowStarted,
    /// AppKit or the deterministic driver produced an input command.
    InputReceived,
    /// A native or synthetic event reached the GPUI platform callback.
    PlatformInputDelivered,
    /// Canvas began handling the correlated input command.
    CanvasInputStarted,
    /// Canvas completed the correlated input command.
    CanvasInputCompleted,
    /// Canvas requested a render for a low-cardinality reason.
    CanvasRenderRequested,
    /// GPUI accepted an entity notification from application code.
    EntityNotified,
    /// GPUI received a request to mark a window or one of its views dirty.
    WindowInvalidated,
    /// The main-thread DisplayLink dispatch source delivered one or more ticks.
    DisplayLinkDelivered,
    /// GPUI began rebuilding a dirty logical window frame.
    LogicalFrameStarted,
    /// Canvas began constructing its element tree.
    CanvasRenderStarted,
    /// Canvas completed construction of its element tree.
    CanvasRenderCompleted,
    /// GPUI began prepainting the full window tree.
    WindowPrepaintStarted,
    /// GPUI began prepainting the Canvas root element.
    CanvasPrepaintStarted,
    /// GPUI completed prepainting the Canvas root element.
    CanvasPrepaintCompleted,
    /// GPUI completed prepainting the full window tree.
    WindowPrepaintCompleted,
    /// GPUI completed painting the Canvas root element.
    CanvasPaintCompleted,
    /// GPUI completed painting the full window tree.
    WindowPaintCompleted,
    /// GPUI completed the logical window-frame rebuild.
    LogicalFrameCompleted,
    /// GPUI requested platform presentation before acquiring the platform-window lock.
    PlatformDrawRequested,
    /// The macOS renderer gained the platform-window lock and entered draw.
    PlatformDrawStarted,
    /// The renderer returned from `nextDrawable`.
    DrawableAcquired,
    /// Metal created the command buffer used by the root drawable.
    CommandBufferCreated,
    /// CPU submitted the command buffer for the root drawable.
    CommandBufferSubmitted,
    /// Metal returned from the root command buffer's `commit` call.
    CommandBufferCommitReturned,
    /// Metal reported that the command buffer was scheduled.
    GpuScheduled,
    /// Metal reported command-buffer completion and absolute GPU times.
    GpuCompleted,
    /// Core Animation reported the drawable's actual presentation timestamp.
    DrawablePresented,
    /// Core Animation завершил drawable без показа на экране.
    DrawableDropped,
    /// The authoritative workload measurement window ended.
    MeasurementWindowCompleted,
    /// Recording stopped before the snapshot was allocated and serialized.
    TraceStopped,
}

/// Stable input class recorded without strings or heap payloads.
#[derive(Clone, Copy, Debug, Serialize)]
#[repr(u64)]
#[serde(rename_all = "snake_case")]
pub enum FrameTraceInputKind {
    /// Primary or auxiliary pointer press.
    PointerDown = 1,
    /// Pointer movement with or without a pressed button.
    PointerMove = 2,
    /// Primary or auxiliary pointer release.
    PointerUp = 3,
    /// Scroll-wheel or trackpad scrolling.
    Scroll = 4,
    /// Trackpad magnification.
    Magnify = 5,
    /// Keyboard press.
    KeyDown = 6,
    /// Keyboard release.
    KeyUp = 7,
    /// Modifier-state change.
    ModifiersChanged = 8,
    /// Other input not used by the Canvas investigation.
    Other = 255,
}

/// Gesture phase recorded without platform-owned event objects.
#[derive(Clone, Copy, Debug, Serialize)]
#[repr(u64)]
#[serde(rename_all = "snake_case")]
pub enum FrameTraceInputPhase {
    /// The platform event has no meaningful gesture phase.
    None = 0,
    /// Gesture beginning.
    Started = 1,
    /// Gesture update.
    Moved = 2,
    /// Gesture completion.
    Ended = 3,
}

/// A fixed-size event copied into the trace buffer.
///
/// `value_0` through `value_2` are event-specific numeric payloads. The offline
/// assembler interprets them by `kind`; the hot path never allocates a tagged payload.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct FrameTraceEvent {
    /// Monotonic order in which writers reserved trace slots.
    pub event_sequence_id: u64,
    /// Event type.
    pub kind: FrameTraceEventKind,
    /// Event time in nanoseconds on the system host-time clock.
    pub timestamp_ns: u64,
    /// Time at which a callback observed an event whose timestamp is retrospective.
    pub callback_observed_ns: u64,
    /// Native event time on the system host-time clock, when available.
    pub physical_input_time_ns: u64,
    /// Correlated input sequence, or zero when no input owns the event.
    pub input_sequence_id: u64,
    /// Native input that caused a synthesized event, or zero for an original event.
    pub parent_input_sequence_id: u64,
    /// Stable numeric [`FrameTraceInputKind`], or zero for non-input events.
    pub input_kind: u64,
    /// Stable numeric [`FrameTraceInputPhase`], or zero for non-input events.
    pub input_phase: u64,
    /// Raw `NSEventType`, or zero for synthetic and non-input events.
    pub native_event_type: u64,
    /// Raw AppKit momentum phase, or zero when unavailable.
    pub input_momentum_phase: u64,
    /// AppKit event number for event types that define it, or zero otherwise.
    pub native_event_number: u64,
    /// Stable application-defined render-request reason, or zero otherwise.
    pub request_reason: u64,
    /// Monotonic Canvas render-request count observed at this event.
    pub request_count: u64,
    /// Render requests coalesced into the correlated logical frame.
    pub coalesced_request_count: u64,
    /// Render requests explicitly dropped before the correlated logical frame.
    pub dropped_request_count: u64,
    /// GPUI logical frame identifier, or zero before a frame is assigned.
    pub logical_frame_id: u64,
    /// Renderer instance identifier, or zero before renderer entry.
    pub renderer_instance_id: u64,
    /// Renderer-local submitted frame identifier, or zero before submission.
    pub renderer_frame_id: u64,
    /// CAMetalDrawable raw identifier plus one, or zero before drawable acquisition.
    pub drawable_id: u64,
    /// Stable pointer identity of the root Metal command buffer.
    pub command_buffer_id: u64,
    /// DisplayLink target host time associated with the event, when available.
    pub target_display_time_ns: u64,
    /// DisplayLink tick sequence associated with the event, when available.
    pub display_tick_sequence: u64,
    /// Number of DisplayLink callbacks represented by a main-thread delivery.
    pub coalesced_display_tick_count: u64,
    /// Number of root drawable presentations in flight after this event.
    pub presentation_queue_depth: u64,
    /// Time spent waiting for `nextDrawable`, when measured.
    pub next_drawable_wait_ns: u64,
    /// Absolute Metal `GPUStartTime`, when known.
    pub gpu_start_time_ns: u64,
    /// Absolute Metal `GPUEndTime`, when known.
    pub gpu_end_time_ns: u64,
    /// Event flags such as physical-input, missed-target, or invalid-presentation.
    pub flags: u64,
}

impl FrameTraceEvent {
    /// Creates a zero-correlated event at the current host time.
    pub fn now(kind: FrameTraceEventKind) -> Self {
        Self {
            event_sequence_id: 0,
            kind,
            timestamp_ns: monotonic_time_ns(),
            callback_observed_ns: 0,
            physical_input_time_ns: 0,
            input_sequence_id: 0,
            parent_input_sequence_id: 0,
            input_kind: 0,
            input_phase: 0,
            native_event_type: 0,
            input_momentum_phase: 0,
            native_event_number: 0,
            request_reason: 0,
            request_count: 0,
            coalesced_request_count: 0,
            dropped_request_count: 0,
            logical_frame_id: 0,
            renderer_instance_id: 0,
            renderer_frame_id: 0,
            drawable_id: 0,
            command_buffer_id: 0,
            target_display_time_ns: 0,
            display_tick_sequence: 0,
            coalesced_display_tick_count: 0,
            presentation_queue_depth: 0,
            next_drawable_wait_ns: 0,
            gpu_start_time_ns: 0,
            gpu_end_time_ns: 0,
            flags: 0,
        }
    }
}

/// Fixed input metadata copied before AppKit continues routing the native event.
#[derive(Clone, Copy, Debug)]
pub struct FrameTraceInput {
    /// Timestamp at the `NSApplication.sendEvent` boundary or synthetic source.
    pub received_time_ns: u64,
    /// Native event timestamp on the host clock, or zero for synthetic input.
    pub physical_time_ns: u64,
    /// Stable input class.
    pub kind: FrameTraceInputKind,
    /// Stable logical gesture phase.
    pub phase: FrameTraceInputPhase,
    /// Raw `NSEventType`, or zero for synthetic input.
    pub native_event_type: u64,
    /// Raw AppKit momentum phase.
    pub momentum_phase: u64,
    /// Valid AppKit event number, or zero when that selector is not applicable.
    pub native_event_number: u64,
    /// Native input that caused a synthesized event.
    pub parent_input_sequence_id: u64,
    /// Origin and validity flags.
    pub flags: u64,
}

/// Immutable trace allocated only after recording has stopped.
#[derive(Debug, Serialize)]
pub struct FrameTraceSnapshot {
    /// Fixed event capacity used for the run.
    pub capacity: usize,
    /// Oldest events overwritten after the trace wrapped.
    pub overwritten_events: u64,
    /// Older writers discarded after a newer writer wrapped to the same slot first.
    pub superseded_writes: u64,
    /// Writers discarded instead of waiting for a concurrently owned slot.
    pub contended_writes: u64,
    /// Events in writer-reservation order.
    pub events: Vec<FrameTraceEvent>,
}

struct FrameTraceSlot {
    sequence: AtomicU64,
    event: UnsafeCell<MaybeUninit<FrameTraceEvent>>,
}

// Each accepted writer receives a unique slot and snapshotting waits for all writers to finish.
unsafe impl Sync for FrameTraceSlot {}

struct FrameTraceBuffer {
    slots: Box<[FrameTraceSlot]>,
    writer_state: AtomicUsize,
    snapshot_claimed: AtomicBool,
    next_event_sequence_id: AtomicU64,
    published_events: AtomicU64,
    superseded_writes: AtomicU64,
    contended_writes: AtomicU64,
}

impl FrameTraceBuffer {
    fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        slots.resize_with(capacity, || FrameTraceSlot {
            sequence: AtomicU64::new(0),
            event: UnsafeCell::new(MaybeUninit::uninit()),
        });
        Self {
            slots: slots.into_boxed_slice(),
            writer_state: AtomicUsize::new(0),
            snapshot_claimed: AtomicBool::new(false),
            next_event_sequence_id: AtomicU64::new(1),
            published_events: AtomicU64::new(0),
            superseded_writes: AtomicU64::new(0),
            contended_writes: AtomicU64::new(0),
        }
    }

    fn record(&self, event: FrameTraceEvent) -> bool {
        let mut writer_state = self.writer_state.load(Ordering::Acquire);
        loop {
            if writer_state & WRITER_STOP_BIT != 0
                || writer_state & WRITER_COUNT_MASK == WRITER_COUNT_MASK
            {
                return false;
            }
            match self.writer_state.compare_exchange_weak(
                writer_state,
                writer_state + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => writer_state = observed,
            }
        }

        let published = self.write_registered(event);
        self.writer_state.fetch_sub(1, Ordering::Release);
        published
    }

    fn write_registered(&self, mut event: FrameTraceEvent) -> bool {
        let sequence = self.next_event_sequence_id.fetch_add(1, Ordering::Relaxed);
        event.event_sequence_id = sequence;
        let slot = &self.slots[(sequence as usize - 1) % self.slots.len()];
        let previous = slot.sequence.load(Ordering::Acquire);
        if previous & WRITING_SEQUENCE_BIT != 0 {
            self.contended_writes.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        if previous >= sequence {
            self.superseded_writes.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        if slot
            .sequence
            .compare_exchange(
                previous,
                sequence | WRITING_SEQUENCE_BIT,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            self.contended_writes.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        // SAFETY: the writing sequence bit grants this writer exclusive slot access.
        unsafe { (*slot.event.get()).write(event) };
        slot.sequence.store(sequence, Ordering::Release);
        self.published_events.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn stop_and_snapshot(&self) -> Option<FrameTraceSnapshot> {
        if self
            .snapshot_claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return None;
        }
        self.writer_state
            .fetch_or(WRITER_STOP_BIT, Ordering::AcqRel);
        while self.writer_state.load(Ordering::Acquire) & WRITER_COUNT_MASK != 0 {
            spin_loop();
        }
        let _ = self.write_registered(FrameTraceEvent::now(FrameTraceEventKind::TraceStopped));
        let published_events = self.published_events.load(Ordering::Acquire);
        let mut events = Vec::with_capacity(
            usize::try_from(published_events)
                .unwrap_or(usize::MAX)
                .min(self.slots.len()),
        );
        for slot in &self.slots {
            let sequence = slot.sequence.load(Ordering::Acquire) & SEQUENCE_MASK;
            if sequence == 0 {
                continue;
            }
            // SAFETY: recording is disabled, all writers exited, and a nonzero sequence is
            // published only after the corresponding event was fully initialized.
            events.push(unsafe { *(*slot.event.get()).assume_init_ref() });
        }
        events.sort_unstable_by_key(|event| event.event_sequence_id);
        Some(FrameTraceSnapshot {
            capacity: self.slots.len(),
            overwritten_events: published_events.saturating_sub(events.len() as u64),
            superseded_writes: self.superseded_writes.load(Ordering::Acquire),
            contended_writes: self.contended_writes.load(Ordering::Acquire),
            events,
        })
    }
}

/// Enables one bounded trace for the lifetime of the current process.
///
/// Allocation occurs here, before measurement. A zero capacity is rejected.
pub fn start(capacity: usize) -> bool {
    if capacity == 0 || TRACE.set(FrameTraceBuffer::new(capacity)).is_err() {
        return false;
    }
    record(FrameTraceEvent::now(FrameTraceEventKind::TraceStarted));
    true
}

/// Records one preconstructed POD event without locking or allocating.
pub fn record(event: FrameTraceEvent) {
    if let Some(trace) = TRACE.get() {
        let _ = trace.record(event);
    }
}

/// Stops recording, waits for in-progress writers, and allocates the serialized snapshot input.
pub fn stop_and_snapshot() -> Option<FrameTraceSnapshot> {
    let trace = TRACE.get()?;
    trace.stop_and_snapshot()
}

/// Returns whether the one-shot trace is currently accepting events.
pub fn is_enabled() -> bool {
    TRACE
        .get()
        .is_some_and(|trace| trace.writer_state.load(Ordering::Acquire) & WRITER_STOP_BIT == 0)
}

/// Records a native or synthetic input and returns its unique sequence identifier.
pub fn record_input(input: FrameTraceInput) -> u64 {
    if !is_enabled() {
        return 0;
    }
    let input_sequence_id = NEXT_INPUT_SEQUENCE_ID.fetch_add(1, Ordering::Relaxed) + 1;
    let mut event = FrameTraceEvent::now(FrameTraceEventKind::InputReceived);
    event.timestamp_ns = input.received_time_ns;
    event.input_sequence_id = input_sequence_id;
    event.parent_input_sequence_id = input.parent_input_sequence_id;
    event.physical_input_time_ns = input.physical_time_ns;
    event.input_kind = input.kind as u64;
    event.input_phase = input.phase as u64;
    event.native_event_type = input.native_event_type;
    event.input_momentum_phase = input.momentum_phase;
    event.native_event_number = input.native_event_number;
    event.flags = input.flags;
    let Some(trace) = TRACE.get() else {
        return 0;
    };
    if trace.record(event) {
        LATEST_INPUT_SEQUENCE_ID.store(input_sequence_id, Ordering::Release);
        input_sequence_id
    } else {
        0
    }
}

/// Returns the input sequence currently being routed by `NSApplication.sendEvent`.
pub fn current_appkit_input_sequence_id() -> u64 {
    CURRENT_APPKIT_INPUT_SEQUENCE_ID.with(Cell::get)
}

/// Restores the previously routed input identifier when the current callback returns.
#[must_use = "the input scope must remain alive while the correlated callback is routed"]
pub struct CurrentAppKitInputScope {
    previous_input_sequence_id: u64,
    not_send_or_sync: PhantomData<Rc<()>>,
}

impl Drop for CurrentAppKitInputScope {
    fn drop(&mut self) {
        CURRENT_APPKIT_INPUT_SEQUENCE_ID
            .with(|current| current.set(self.previous_input_sequence_id));
    }
}

/// Routes one native or synthetic input through nested platform callbacks on this thread.
pub fn enter_current_appkit_input_sequence_id(input_sequence_id: u64) -> CurrentAppKitInputScope {
    let previous_input_sequence_id =
        CURRENT_APPKIT_INPUT_SEQUENCE_ID.with(|current| current.replace(input_sequence_id));
    CurrentAppKitInputScope {
        previous_input_sequence_id,
        not_send_or_sync: PhantomData,
    }
}

/// Returns the most recently delivered process-wide input sequence.
pub fn latest_input_sequence_id() -> u64 {
    LATEST_INPUT_SEQUENCE_ID.load(Ordering::Acquire)
}

/// Records delivery of a previously identified input to the GPUI platform callback.
pub fn record_platform_input_delivery(input_sequence_id: u64) {
    if input_sequence_id == 0 {
        return;
    }
    let mut event = FrameTraceEvent::now(FrameTraceEventKind::PlatformInputDelivered);
    event.input_sequence_id = input_sequence_id;
    record(event);
}

/// Allocates the next process-wide logical GPUI frame identifier.
pub(crate) fn next_logical_frame_id() -> u64 {
    NEXT_LOGICAL_FRAME_ID.fetch_add(1, Ordering::Relaxed) + 1
}

/// Allocates a process-wide renderer instance identifier.
pub(crate) fn next_renderer_instance_id() -> u64 {
    NEXT_RENDERER_INSTANCE_ID.fetch_add(1, Ordering::Relaxed) + 1
}

/// Allocates a process-wide DisplayLink tick identifier that survives link restarts.
pub(crate) fn next_display_tick_sequence_id() -> u64 {
    NEXT_DISPLAY_TICK_SEQUENCE_ID.fetch_add(1, Ordering::Relaxed) + 1
}

/// Updates and records the latest DisplayLink target delivered on the main thread.
pub(crate) fn record_display_link_delivery(
    target_display_time_ns: u64,
    display_tick_sequence: u64,
    coalesced_tick_count: u64,
    flags: u64,
) {
    LATEST_DISPLAY_TARGET_NS.store(target_display_time_ns, Ordering::Release);
    LATEST_DISPLAY_TICK_SEQUENCE.store(display_tick_sequence, Ordering::Release);
    let mut event = FrameTraceEvent::now(FrameTraceEventKind::DisplayLinkDelivered);
    event.target_display_time_ns = target_display_time_ns;
    event.display_tick_sequence = display_tick_sequence;
    event.coalesced_display_tick_count = coalesced_tick_count;
    event.flags = flags;
    record(event);
}

/// Prevents transaction-driven frames from inheriting a stale DisplayLink deadline.
pub(crate) fn invalidate_display_link_context() {
    LATEST_DISPLAY_TARGET_NS.store(0, Ordering::Release);
    LATEST_DISPLAY_TICK_SEQUENCE.store(0, Ordering::Release);
}

/// Returns the latest main-thread DisplayLink target in host-time nanoseconds.
pub fn latest_display_target_ns() -> u64 {
    LATEST_DISPLAY_TARGET_NS.load(Ordering::Acquire)
}

/// Returns the latest main-thread DisplayLink sequence.
pub fn latest_display_tick_sequence() -> u64 {
    LATEST_DISPLAY_TICK_SEQUENCE.load(Ordering::Acquire)
}

/// Converts a floating-point host-clock timestamp to nanoseconds.
pub(crate) fn host_seconds_to_ns(seconds: f64) -> u64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    let nanoseconds = seconds * 1_000_000_000.0;
    if nanoseconds >= u64::MAX as f64 {
        u64::MAX
    } else {
        nanoseconds.round() as u64
    }
}

/// Encodes Metal's zero-based drawable identifier while preserving zero as "not acquired".
pub(crate) fn encode_drawable_id(raw_drawable_id: u64) -> u64 {
    raw_drawable_id.saturating_add(1)
}

/// Converts raw mach host ticks to nanoseconds.
#[cfg(target_os = "macos")]
pub(crate) fn mach_ticks_to_ns(ticks: u64) -> u64 {
    use mach2::mach_time::{mach_timebase_info, mach_timebase_info_data_t};

    static TIMEBASE: OnceLock<mach_timebase_info_data_t> = OnceLock::new();
    let timebase = TIMEBASE.get_or_init(|| {
        let mut value = mach_timebase_info_data_t { numer: 0, denom: 0 };
        // SAFETY: value is a valid out pointer for mach_timebase_info.
        let result = unsafe { mach_timebase_info(&mut value) };
        if result != 0 {
            return mach_timebase_info_data_t { numer: 0, denom: 0 };
        }
        value
    });
    if timebase.denom == 0 {
        return 0;
    }
    let nanoseconds = u128::from(ticks) * u128::from(timebase.numer) / u128::from(timebase.denom);
    u64::try_from(nanoseconds).unwrap_or(u64::MAX)
}

/// Returns current system host time in nanoseconds.
#[cfg(target_os = "macos")]
pub fn monotonic_time_ns() -> u64 {
    // SAFETY: mach_absolute_time has no preconditions.
    mach_ticks_to_ns(unsafe { mach2::mach_time::mach_absolute_time() })
}

/// Returns a process-relative monotonic time outside macOS diagnostic builds.
#[cfg(not(target_os = "macos"))]
pub fn monotonic_time_ns() -> u64 {
    use std::time::Instant;

    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    u64::try_from(ORIGIN.get_or_init(Instant::now).elapsed().as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn bounded_trace_retains_the_newest_events_after_wrapping() {
        let trace = FrameTraceBuffer::new(2);
        trace.record(FrameTraceEvent::now(FrameTraceEventKind::TraceStarted));
        trace.record(FrameTraceEvent::now(FrameTraceEventKind::TraceStopped));
        trace.record(FrameTraceEvent::now(
            FrameTraceEventKind::LogicalFrameStarted,
        ));
        let snapshot = trace.stop_and_snapshot().unwrap();
        assert_eq!(snapshot.events.len(), 2);
        assert_eq!(snapshot.overwritten_events, 2);
        assert_eq!(snapshot.superseded_writes, 0);
        assert_eq!(snapshot.contended_writes, 0);
        assert_eq!(snapshot.events[0].event_sequence_id, 3);
        assert_eq!(snapshot.events[1].event_sequence_id, 4);
    }

    #[test]
    fn stop_waits_for_registered_writers_and_publishes_the_last_marker() {
        let trace = Arc::new(FrameTraceBuffer::new(50_000));
        let barrier = Arc::new(Barrier::new(5));
        let writers = (0..4)
            .map(|_| {
                let trace = trace.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..10_000 {
                        if !trace.record(FrameTraceEvent::now(
                            FrameTraceEventKind::LogicalFrameStarted,
                        )) {
                            break;
                        }
                    }
                })
            })
            .collect::<Vec<_>>();

        barrier.wait();
        let snapshot = trace.stop_and_snapshot().unwrap();
        for writer in writers {
            writer.join().unwrap();
        }

        assert_eq!(
            snapshot.events.last().map(|event| event.kind),
            Some(FrameTraceEventKind::TraceStopped)
        );
        assert!(
            snapshot
                .events
                .windows(2)
                .all(|pair| pair[0].event_sequence_id < pair[1].event_sequence_id)
        );
        assert!(!trace.record(FrameTraceEvent::now(
            FrameTraceEventKind::LogicalFrameStarted,
        )));
        assert!(trace.stop_and_snapshot().is_none());
    }
}
