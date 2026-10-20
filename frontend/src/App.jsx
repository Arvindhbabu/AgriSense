import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./styles.css";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const STEP_LABELS = {
  "case.started": "Analysis session created",
  "diagnosis.started": "Examining crop image",
  "diagnosis.completed": "Diagnosis candidate evaluated",
  "advisory.started": "Building crop guidance",
  "advisory.skipped": "Safety gate requested review",
  "advisory.completed": "Guidance stage complete",
  "market.started": "Checking market data",
  "market.completed": "Market evidence checked",
  "climate.started": "Checking climate context",
  "climate.completed": "Climate evidence checked",
  "case.completed": "Analysis completed",
};
const readableStep = (step) => STEP_LABELS[step] || String(step || "system").replaceAll(".", " ").replaceAll("_", " ");
const formatConfidence = (value) => {
  const num = Number(value);
  return Number.isFinite(num) ? String(Math.round(Math.max(0, Math.min(1, num)) * 100)) + "%" : "Not reported";
};
const evidenceLabel = (section) => {
  const status = section?.evidence?.status || "unavailable";
  if (status === "available_with_provenance") return "Provenance supplied";
  if (status === "unverified") return "Unverified source";
  return "Not connected / unavailable";
};

export default function App() {
  const [file, setFile] = useState(null);
  const [cropType, setCropType] = useState("wheat");
  const [userId, setUserId] = useState("farmer_web");
  const [location, setLocation] = useState("");
  const [marketRegion, setMarketRegion] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [connection, setConnection] = useState("checking");
  const [sessionId, setSessionId] = useState("");
  const [events, setEvents] = useState([]);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef(null);
  const socketRef = useRef(null);

  const refreshHistory = useCallback(async () => {
    try {
      const response = await fetch(API_BASE + "/history?limit=8");
      if (!response.ok) return;
      const data = await response.json();
      setHistory(Array.isArray(data.items) ? data.items : []);
    } catch {}
  }, []);

  useEffect(() => {
    let alive = true;
    fetch(API_BASE + "/")
      .then((response) => { if (!response.ok) throw new Error("API not ready"); return response.json(); })
      .then(() => { if (alive) setConnection("online"); })
      .catch(() => { if (alive) setConnection("offline"); });
    refreshHistory();
    return () => { alive = false; socketRef.current?.close(); };
  }, [refreshHistory]);

  useEffect(() => {
    if (!file) { setPreviewUrl(""); return undefined; }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const latestStep = useMemo(() => {
    const last = [...events].reverse().find((item) => item.type === "progress");
    return last ? readableStep(last.step) : "Ready for image";
  }, [events]);

  const selectFile = (candidate) => {
    if (!candidate) return;
    if (!candidate.type.startsWith("image/")) { setError("Choose a PNG, JPG, or WEBP image."); return; }
    if (candidate.size > 10 * 1024 * 1024) { setError("Image exceeds the current 10 MB upload limit."); return; }
    setError("");
    setResult(null);
    setFile(candidate);
  };

  const startAnalysis = async (event) => {
    event.preventDefault();
    if (!file) { setError("Choose a crop image before starting."); return; }
    socketRef.current?.close();
    setBusy(true);
    setError("");
    setResult(null);
    setSessionId("");
    setEvents([{ type: "local", text: "Uploading image and opening an analysis session…" }]);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("crop_type", cropType);
      formData.append("user_id", userId.trim() || "farmer_web");
      formData.append("location", location.trim());
      formData.append("market_region", marketRegion.trim());
      const response = await fetch(API_BASE + "/upload-start", { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not start analysis.");
      setSessionId(payload.session_id);

      const wsBase = API_BASE.replace(/^http/, "ws");
      const socket = new WebSocket(wsBase + (payload.websocket_url || ("/ws/" + payload.session_id)));
      socketRef.current = socket;
      socket.onopen = () => {
        setConnection("online");
        setEvents((current) => [...current, { type: "local", text: "Live event stream connected." }]);
      };
      socket.onmessage = (messageEvent) => {
        let message;
        try { message = JSON.parse(messageEvent.data); }
        catch { setEvents((current) => [...current, { type: "local", text: "Received an unreadable server event." }]); return; }
        if (message.type === "progress") {
          setEvents((current) => [...current, message]);
        } else if (message.type === "result") {
          setResult(message.payload);
          setEvents((current) => [...current, { type: "local", text: "Report received and saved." }]);
          refreshHistory();
        } else if (message.type === "error") {
          setError(message.text || "Analysis failed.");
          setEvents((current) => [...current, message]);
        } else if (message.type === "finished") {
          setBusy(false);
          socket.close();
          refreshHistory();
        }
      };
      socket.onerror = () => {
        setError("Live connection failed. Check that the real-time API is running.");
        setBusy(false);
      };
      socket.onclose = () => { socketRef.current = null; };
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start analysis.");
      setBusy(false);
    }
  };

  const openHistoryCase = async (caseId) => {
    try {
      setError("");
      const response = await fetch(API_BASE + "/cases/" + encodeURIComponent(caseId));
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not load the report.");
      setResult(data);
      setSessionId("");
      setEvents([{ type: "local", text: "Loaded a saved report from local history." }]);
    } catch (err) { setError(err instanceof Error ? err.message : "Could not load report."); }
  };

  const downloadReport = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "agrisense-" + (result.case_id || "report") + ".json";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const diagnosis = result?.diagnosis || {};
  const gate = result?.diagnosis_gate || {};
  const reviewRequired = gate.review_required ?? true;
  const confidence = gate.reported_confidence ?? diagnosis.confidence;
  const eventItems = events.length ? events.slice(-8) : [
    { type: "placeholder", step: "diagnosis.started", text: "Waiting for an image to start the pipeline" },
    { type: "placeholder", step: "advisory.started", text: "Advice depends on the diagnosis gate" },
    { type: "placeholder", step: "market.started", text: "Market adapter status will appear here" },
    { type: "placeholder", step: "climate.started", text: "Weather adapter status will appear here" },
  ];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup"><div className="brand-mark">A<span>+</span></div><div><strong>AgriSense</strong><small>FIELD INTELLIGENCE</small></div></div>
        <div className="sidebar-label">WORKSPACE</div>
        <a className="nav-item active" href="#overview"><span className="nav-icon">◈</span> Crop intelligence</a>
        <a className="nav-item" href="#analysis"><span className="nav-icon">⌁</span> New assessment</a>
        <a className="nav-item" href="#history"><span className="nav-icon">↺</span> Case history</a>
        <div className="sidebar-bottom">
          <div className="system-card"><span className={"status-dot " + (connection === "online" ? "green" : connection === "offline" ? "red" : "")} /><div><strong>API connection</strong><small>{connection === "online" ? "Connected locally" : connection === "offline" ? "Waiting for backend" : "Checking service…"}</small></div></div>
          <p>Decision support for healthier crops and better-informed farming.</p><span className="build-label">PROTOTYPE · RESEARCH BUILD</span>
        </div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> <b>Crop intelligence</b></div><div className="topbar-right"><span className={"connection-pill " + connection}><i />{connection === "online" ? "System online" : connection === "offline" ? "API offline" : "Connecting"}</span><div className="avatar">AS</div></div></header>

        <section className="welcome-row">
          <div><div className="eyebrow"><span className="eyebrow-line" /> PRECISION AGRICULTURE · MULTI-AGENT AI</div><h1>See the signals.<br /><em>Grow with confidence.</em></h1><p className="intro-copy">An evidence-aware workspace for crop image triage, climate context, and market signals — with uncertainty made visible instead of hidden.</p></div>
          <div className="season-card"><div className="season-icon">✳</div><div><small>INTELLIGENCE MODE</small><strong>Evidence-gated</strong><span>Provisional insights · Human review when uncertain</span></div></div>
        </section>

        <section className="metric-grid">
          <article className="metric-card"><div className="metric-head"><span>ANALYSIS STATUS</span><span className="metric-icon">⌁</span></div><strong>{busy ? "In progress" : result ? "Report ready" : "Standby"}</strong><p>{busy ? latestStep : "Ready for your next crop assessment"}</p><div className="metric-bar"><span style={{ width: busy ? "68%" : result ? "100%" : "12%" }} /></div></article>
          <article className="metric-card"><div className="metric-head"><span>SAVED CASES</span><span className="metric-icon">▤</span></div><strong>{history.length}</strong><p>Recent reports available in this instance</p><div className="mini-bars"><i /><i /><i /><i /><i /><i /><i /></div></article>
          <article className="metric-card"><div className="metric-head"><span>SAFETY POSTURE</span><span className="metric-icon">◇</span></div><strong>Uncertainty-aware</strong><p>Weak signals trigger review instead of treatment claims</p><div className="safety-line"><span /> Human-in-the-loop</div></article>
        </section>

        {error && <div className="alert-banner"><strong>Action needed</strong><span>{error}</span><button type="button" onClick={() => setError("")} aria-label="Dismiss error">×</button></div>}

        <section className="workspace-grid" id="analysis">
          <article className="panel upload-panel">
            <div className="panel-header"><div><span className="section-kicker">01 / INPUT</span><h2>Start a crop assessment</h2><p>Upload a clear leaf image to begin a live multi-agent analysis.</p></div><span className="panel-step">01</span></div>
            <form onSubmit={startAnalysis}>
              <input ref={fileRef} type="file" accept="image/*" className="visually-hidden" onChange={(e) => selectFile(e.target.files?.[0])} />
              <div className={"dropzone " + (dragging ? "dragging" : "") + (file ? " has-file" : "")} onDragEnter={(e) => { e.preventDefault(); setDragging(true); }} onDragOver={(e) => e.preventDefault()} onDragLeave={() => setDragging(false)} onDrop={(e) => { e.preventDefault(); setDragging(false); selectFile(e.dataTransfer.files?.[0]); }} onClick={() => fileRef.current?.click()} role="button" tabIndex={0} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") fileRef.current?.click(); }}>
                {previewUrl ? <img className="image-preview" src={previewUrl} alt="Selected crop preview" /> : <div className="upload-graphic"><span>↑</span><i /><i /><i /></div>}
                <strong>{file ? file.name : "Drop your crop image here"}</strong><p>{file ? String(Math.round(file.size / 1024)) + " KB · Ready to analyze" : "or click to browse your device"}</p><small>PNG, JPG, WEBP · Max 10 MB</small>
              </div>
              <div className="form-row">
                <label>Crop type<select value={cropType} onChange={(e) => setCropType(e.target.value)}><option value="wheat">Wheat</option><option value="rice">Rice</option><option value="tomato">Tomato</option><option value="potato">Potato</option><option value="maize">Maize</option><option value="cotton">Cotton</option><option value="other">Other / unspecified</option></select></label>
                <label>Farmer / session ID<input value={userId} onChange={(e) => setUserId(e.target.value)} maxLength={120} placeholder="farmer_web" /></label>
              </div>
              <div className="form-row form-row-extra">
                <label>Farm location for weather<input value={location} onChange={(e) => setLocation(e.target.value)} maxLength={200} placeholder="City or latitude,longitude" /></label>
                <label>Market state (optional)<input value={marketRegion} onChange={(e) => setMarketRegion(e.target.value)} maxLength={100} placeholder="Tamil Nadu" /></label>
              </div>
              <div className="upload-footer"><div className="privacy-note"><span>▣</span><p>Image is processed by your configured local API.<br /><b>Not a confirmed agricultural diagnosis.</b></p></div><button className="primary-button" type="submit" disabled={busy || connection === "offline"}>{busy ? <><span className="spinner" /> Analyzing…</> : <>Run assessment <span>↗</span></>}</button></div>
            </form>
          </article>

          <article className="panel pipeline-panel">
            <div className="panel-header"><div><span className="section-kicker">02 / LIVE PIPELINE</span><h2>Agent activity</h2><p>Follow each stage as events arrive from the backend.</p></div><span className={"live-tag " + (busy ? "is-live" : "")}><i /> {busy ? "LIVE" : "EVENT LOG"}</span></div>
            <div className="pipeline-list">
              {eventItems.map((item, index) => (
                <div className={"pipeline-event " + (item.type === "progress" ? "received" : "")} key={String(item.step || item.text) + "-" + index}>
                  <div className="event-rail"><span className={item.step?.endsWith("completed") || item.step === "case.completed" ? "done" : item.type === "placeholder" ? "muted" : "event-dot"} /></div>
                  <div className="event-content"><strong>{item.type === "local" ? item.text : readableStep(item.step || "system")}</strong><small>{item.text && item.type !== "local" ? item.text : item.meta?.message || item.meta?.decision_status || item.meta?.evidence_status || (item.type === "placeholder" ? item.text : item.meta ? JSON.stringify(item.meta) : "Backend event received")}</small></div>
                  <span className="event-status">{item.type === "placeholder" ? "WAITING" : item.type === "local" ? "INFO" : "RECEIVED"}</span>
                </div>
              ))}
            </div>
            <div className="pipeline-footer"><span><i /> Progress stream</span><span>{events.filter((item) => item.type === "progress").length} events</span></div>
          </article>
        </section>

        <section className="results-section">
          <div className="section-title-row"><div><span className="section-kicker">03 / DECISION REPORT</span><h2>Assessment insights</h2></div>{result && <button className="secondary-button" onClick={downloadReport} type="button">↓ Export JSON</button>}</div>
          {!result ? <div className="empty-results"><div className="empty-mark">◈</div><div><strong>Your crop report will appear here</strong><p>Run an assessment to see the model output, uncertainty gate, advice, and evidence-source status.</p></div><span>AWAITING ANALYSIS</span></div> : <>
            <div className="result-grid">
              <article className="result-card diagnosis-card">
                <div className="result-card-top"><span>DIAGNOSIS CANDIDATE</span><span className={"review-pill " + (reviewRequired ? "review" : "provisional")}>{reviewRequired ? "Review required" : "Provisional"}</span></div>
                <h3>{String(diagnosis.label || "Unknown").replaceAll("_", " ")}</h3><p>{gate.message || "Treat this output as a candidate, not a confirmed diagnosis."}</p>
                <div className="confidence-row"><span>Reported confidence</span><strong>{formatConfidence(confidence)}</strong></div><div className="confidence-track"><span style={{ width: formatConfidence(confidence) }} /></div>
                <small className="muted-caption">{gate.confidence_calibrated ? "Calibration status: provided by model" : "Confidence has not been independently calibrated."}</small>
                {gate.reasons?.length > 0 && <div className="reason-list">{gate.reasons.map((reason) => <span key={reason}>• {reason.replaceAll("_", " ")}</span>)}</div>}
              </article>
              <article className="result-card advisory-card"><div className="result-card-top"><span>FIELD GUIDANCE</span><span className="neutral-pill">{String(result.advisory_mode || "Advisory").replaceAll("_", " ")}</span></div><h3>Recommended next step</h3><p className="advice-text">{result.advice || "No advice was returned."}</p><div className="safety-callout"><span>!</span><p>{result.safety_note || "Verify recommendations with a qualified local agricultural source before treatment."}</p></div></article>
            </div>
            <div className="evidence-grid">
              <article className="evidence-card"><div className="evidence-icon">☼</div><div className="evidence-body"><span>CLIMATE CONTEXT</span><strong>{result.climate?.risk || "Unknown"}</strong><p>{evidenceLabel(result.climate)}</p><small>Source: {result.climate?.evidence?.source || "Not supplied"}</small></div><span className="evidence-symbol">↗</span></article>
              <article className="evidence-card"><div className="evidence-icon">₹</div><div className="evidence-body"><span>MARKET CONTEXT</span><strong>{result.market?.avg_price_per_kg != null ? "₹" + result.market.avg_price_per_kg + " / kg" : "Price unavailable"}</strong><p>{evidenceLabel(result.market)}</p><small>Crop: {result.market?.crop || cropType} · Region: {result.market?.region || "Not supplied"}</small></div><span className="evidence-symbol">↗</span></article>
            </div>
            <div className="report-meta"><span>Case ID <code>{result.case_id || "not supplied"}</code></span><span>Pipeline {result.run_metadata?.pipeline || "version unspecified"}</span><span>Duration {result.run_metadata?.duration_ms != null ? String(result.run_metadata.duration_ms) + " ms" : "not reported"}</span></div>
          </>}
        </section>

        <section className="history-section" id="history">
          <div className="section-title-row"><div><span className="section-kicker">04 / AUDIT TRAIL</span><h2>Recent cases</h2></div><button className="text-button" onClick={refreshHistory} type="button">Refresh history ↻</button></div>
          {history.length === 0 ? <div className="history-empty">No saved cases yet. Completed reports will be listed here.</div> : <div className="history-table">
            <div className="history-row history-head"><span>CASE / DIAGNOSIS</span><span>DECISION GATE</span><span>CREATED</span><span /></div>
            {history.map((item) => <button className="history-row history-item" key={item.case_id} onClick={() => openHistoryCase(item.case_id)} type="button"><span><strong>{String(item.diagnosis || "Unknown").replaceAll("_", " ")}</strong><small>{item.case_id}</small></span><span><i className={"gate-dot " + (item.decision_status === "review_required" ? "amber" : "green")} />{item.decision_status || "Unknown"}</span><span>{item.generated_at ? new Date(item.generated_at).toLocaleString() : "Time unavailable"}</span><span className="open-case">Open ↗</span></button>)}
          </div>}
        </section>

        <footer className="page-footer"><span>AGRISENSE <b>·</b> EVIDENCE-GROUNDED CROP INTELLIGENCE</span><span>Research prototype — validate with local agricultural experts</span></footer>
      </main>
    </div>
  );
}
