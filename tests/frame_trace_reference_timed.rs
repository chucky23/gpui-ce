#![cfg(feature = "frame-trace")]

use gpui::frame_trace::{self, FrameTraceCaptureLevel, FrameTraceEvent, FrameTraceEventKind};

// Трасса одноразовая на процесс, поэтому уровень проверяется в отдельном тестовом бинарнике.
#[test]
fn reference_timed_capture_keeps_reference_events_and_timing_fields() {
    assert!(frame_trace::start_with_level(
        16,
        FrameTraceCaptureLevel::ReferenceTimed,
        9
    ));
    assert!(frame_trace::is_reference_timing_enabled());
    assert!(!frame_trace::is_detailed_enabled());

    for kind in [
        FrameTraceEventKind::LogicalFrameStarted,
        FrameTraceEventKind::DrawableAcquired,
        FrameTraceEventKind::GpuCompleted,
        FrameTraceEventKind::MeasurementWindowStarted,
    ] {
        frame_trace::record(FrameTraceEvent::now(kind));
    }
    let mut submitted = FrameTraceEvent::now(FrameTraceEventKind::CommandBufferSubmitted);
    submitted.logical_frame_id = 3;
    submitted.logical_frame_started_ns = 100;
    submitted.next_drawable_wait_ns = 20;
    frame_trace::record(submitted);
    let mut presented = submitted;
    presented.kind = FrameTraceEventKind::DrawablePresented;
    presented.next_drawable_wait_ns = 0;
    presented.gpu_start_time_ns = 200;
    presented.gpu_end_time_ns = 300;
    frame_trace::record(presented);

    let Some(snapshot) = frame_trace::stop_and_snapshot() else {
        panic!("reference-timed trace must produce one snapshot");
    };
    assert!(!frame_trace::is_reference_timing_enabled());
    assert_eq!(
        snapshot
            .events
            .iter()
            .map(|event| event.kind)
            .collect::<Vec<_>>(),
        vec![
            FrameTraceEventKind::TraceStarted,
            FrameTraceEventKind::CommandBufferSubmitted,
            FrameTraceEventKind::DrawablePresented,
            FrameTraceEventKind::TraceStopped,
        ]
    );
    let submitted = snapshot.events[1];
    assert_eq!(submitted.logical_frame_started_ns, 100);
    assert_eq!(submitted.next_drawable_wait_ns, 20);
    let presented = snapshot.events[2];
    assert_eq!(presented.gpu_start_time_ns, 200);
    assert_eq!(presented.gpu_end_time_ns, 300);

    let Ok(serde_json::Value::Object(json)) = serde_json::to_value(&snapshot) else {
        panic!("snapshot must serialize to a JSON object");
    };
    let events = json["events"].as_array().cloned().unwrap_or_default();
    assert_eq!(events.len(), 4);
    assert!(events[0].get("logical_frame_started_ns").is_none());
    assert_eq!(
        events[1]["logical_frame_started_ns"],
        serde_json::json!(100)
    );
}
