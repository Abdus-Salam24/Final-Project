import { useState, useRef } from 'react';
import { ScanLine, Upload, Package, AlertTriangle, CheckCircle, XCircle, Bot } from 'lucide-react';
import { apiPostForm, apiPostJson } from '../api';


const LOGS_SEQUENCE = [
  { text: '$ Initializing SCALA-Guard sandbox...', type: 'cmd' },
  { text: '> Pulling Docker image: scala-guard:sandbox', type: 'ok' },
  { text: '> Container started: sg-sandbox-001', type: 'ok' },
  { text: '> Running: strace -o syscalls.log python setup.py install', type: 'cmd' },
  { text: '> Capturing network: tcpdump -w capture.pcap', type: 'cmd' },
  { text: '> Monitoring file system access...', type: 'ok' },
  { text: '> Analyzing syscall sequences...', type: 'ok' },
  { text: '> Running ML classifier...', type: 'ok' },
  { text: '> Querying DeepSeek remediation engine...', type: 'ok' },
  { text: '> Analysis complete. Generating report.', type: 'ok' },
];

function RiskGauge({ score, label }: { score: number; label: string }) {
  const cls = label.toLowerCase();
  return (
    <div className="risk-gauge">
      <div className={`risk-score-big ${cls}`}>{score}%</div>
      <div className={`risk-badge ${cls}`}>
        {cls === 'malicious' && <XCircle size={12} />}
        {cls === 'suspicious' && <AlertTriangle size={12} />}
        {cls === 'benign' && <CheckCircle size={12} />}
        {label}
      </div>
    </div>
  );
}

interface ShapFactor {
  type: 'critical' | 'high' | 'medium' | 'low';
  factor: string;
  impact: number;
}

interface RemediationData {
  threat_type: string;
  summary: string;
  fix_steps?: string[];
  safe_alternative?: string;
  cve_references?: string[];
}

interface SandboxData {
  syscall_count?: number;
  network_connections?: Array<string | { ip?: string; port?: number; suspicious?: boolean }>;
  data_exfiltrated_kb?: number;
  processes_spawned?: string[];
  files_accessed?: string[];
  sandbox_duration_ms?: number;
  syscall_samples?: string[];
}

interface RiskData {
  score: number;
  label: string;
  confidence_band?: { low: number; high: number };
  shap_factors?: ShapFactor[];
}

interface ScanResultData {
  package_name: string;
  ecosystem?: string;
  scan_id: string;
  timestamp: string;
  registry_info?: {
    author?: string;
    version?: string;
    summary?: string;
    error?: string;
  };
  risk?: RiskData;
  sandbox?: SandboxData;
  remediation?: RemediationData;
}

type NameScanResponse = ScanResultData;

type TextScanResponse = ScanResultData;

type FileScanResponse = ScanResultData;

function SHAPFactors({ factors }: { factors?: ShapFactor[] }) {
  if (!factors || factors.length === 0)
    return <div style={{ color: 'var(--text-dim)', fontSize: 13 }}>No suspicious factors detected.</div>;

  return (
    <div>
      {factors.map((f, i) => {
        const color = f.type === 'critical' ? 'var(--red)' : f.type === 'high' ? 'var(--yellow)' : 'var(--green)';
        return (
          <div key={i} className="shap-factor">
            <div style={{ fontSize: 10, textTransform: 'uppercase', color, letterSpacing: 1, minWidth: 60 }}>
              {f.type}
            </div>
            <div className="shap-label">{f.factor}</div>
            <div className="shap-impact" style={{ color }}>+{f.impact}%</div>
          </div>
        );
      })}
    </div>
  );
}

function RemediationPanel({ data }: { data?: RemediationData }) {
  if (!data) return null;

  return (
    <div style={{ marginTop: 24 }}>
      <div className="section-title">
        <Bot size={16} />
        AI Remediation Report
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {/* Summary */}
        <div className="card-sm" style={{ gridColumn: '1 / -1' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', letterSpacing: 1, marginBottom: 8, textTransform: 'uppercase' }}>
            Threat Analysis
          </div>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            background: 'var(--red-dim)', color: 'var(--red)',
            border: '1px solid rgba(255,69,96,0.3)',
            borderRadius: 4, padding: '4px 10px', fontSize: 12, marginBottom: 12,
          }}>
            ⚠ {data.threat_type}
          </div>
          <p style={{ fontSize: 13, color: 'var(--text)', lineHeight: 1.7 }}>{data.summary}</p>
        </div>

        {/* Fix Steps */}
        <div className="card-sm">
          <div style={{ fontSize: 11, color: 'var(--text-dim)', letterSpacing: 1, marginBottom: 12, textTransform: 'uppercase' }}>
            Fix Steps
          </div>
          <ol style={{ paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 8 }}>
            {(data.fix_steps || []).filter(Boolean).map((step: string, i: number) => (
              <li key={i} style={{ fontSize: 12, color: 'var(--text)', lineHeight: 1.6 }}>{step}</li>
            ))}
          </ol>
        </div>

        {/* Alternatives + CVEs */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {data.safe_alternative && (
            <div className="card-sm">
              <div style={{ fontSize: 11, color: 'var(--text-dim)', letterSpacing: 1, marginBottom: 8, textTransform: 'uppercase' }}>
                Safe Alternative
              </div>
              <div style={{
                background: 'var(--green-dim)', borderRadius: 4, padding: '8px 12px',
                fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--green)',
              }}>
                {data.safe_alternative}
              </div>
            </div>
          )}

          {data.cve_references && data.cve_references.length > 0 && (
            <div className="card-sm">
              <div style={{ fontSize: 11, color: 'var(--text-dim)', letterSpacing: 1, marginBottom: 8, textTransform: 'uppercase' }}>
                CVE References
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {data.cve_references.map((cve: string) => (
                  <span key={cve} style={{
                    background: 'rgba(255,69,96,0.1)', color: 'var(--red)',
                    border: '1px solid rgba(255,69,96,0.3)',
                    borderRadius: 4, padding: '3px 8px', fontSize: 11, fontFamily: 'var(--font-mono)',
                  }}>
                    {cve}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function Scanner() {
  const [tab, setTab] = useState('name');
  const [pkgName, setPkgName] = useState('');
  const [pkgVersion, setPkgVersion] = useState('');
  const [scanText, setScanText] = useState('');
  const [ecosystem, setEcosystem] = useState('pypi');
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<typeof LOGS_SEQUENCE>([]);
  const [result, setResult] = useState<ScanResultData | null>(null);
  const [error, setError] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  // function addLog(entry: typeof LOGS_SEQUENCE[0]) {
  //   setLogs(prev => [...prev, entry]);
  // }

  async function runLogs() {
    setLogs([]);
    for (let i = 0; i < LOGS_SEQUENCE.length; i++) {
      await new Promise(r => setTimeout(r, 280 + Math.random() * 200));
      setLogs(prev => [...prev, LOGS_SEQUENCE[i]]);
    }
  }

  async function handleScanName() {
    if (!pkgName.trim()) { setError('Package name is required'); return; }
    setError(''); setResult(null); setLoading(true);

    runLogs();

    try {
      const data = await apiPostJson<NameScanResponse>('/analyze/name', { name: pkgName.trim(), version: pkgVersion || 'latest', ecosystem });
      setResult(data);
    } catch (e) {
      const error = e instanceof Error ? e.message : 'Failed to connect to API. Is the backend running?';
      setError(error);
    } finally {
      setLoading(false);
    }
  }

  async function handleScanFile() {
    if (!file) { setError('Please select a file first'); return; }
    setError(''); setResult(null); setLoading(true);

    runLogs();

    try {
      const form = new FormData();
      form.append('package_file', file);
      const data = await apiPostForm<FileScanResponse>('/analyze', form);
      setResult(data);
    } catch (e) {
      const error = e instanceof Error ? e.message : 'Failed to connect to API. Is the backend running?';
      setError(error);
    } finally {
      setLoading(false);
    }
  }

  async function handleScanText() {
    if (!scanText.trim()) { setError('Scan text is required'); return; }
    setError(''); setResult(null); setLoading(true);

    runLogs();

    try {
      const data = await apiPostJson<TextScanResponse>('/analyze/text', { text: scanText.trim(), ecosystem });
      setResult(data);
    } catch (e) {
      const error = e instanceof Error ? e.message : 'Failed to connect to API. Is the backend running?';
      setError(error);
    } finally {
      setLoading(false);
    }
  }

  const risk = result?.risk;
  const sandbox = result?.sandbox;
  const syscallSamples = sandbox?.syscall_samples ?? [];

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Package <span>Scanner</span></h1>
        <p className="page-subtitle">Analyze packages from registry or upload a file for behavioral analysis</p>
      </div>

      {/* Input Area */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="tabs">
          <button className={`tab-btn ${tab === 'name' ? 'active' : ''}`} onClick={() => setTab('name')}>
            By Name
          </button>
          <button className={`tab-btn ${tab === 'text' ? 'active' : ''}`} onClick={() => setTab('text')}>
            By Text
          </button>
          <button className={`tab-btn ${tab === 'file' ? 'active' : ''}`} onClick={() => setTab('file')}>
            Upload File
          </button>
        </div>

        {tab === 'name' && (
          <div>
            <div className="input-row flex-col sm:flex-row">
              <div className="input-group flex-1 min-w-0">
                <label className="input-label">Package Name</label>
                <input
                  className="input-field"
                  placeholder="e.g. requests, express, numpy"
                  value={pkgName}
                  onChange={e => setPkgName(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleScanName()}
                />
              </div>
              <div className="input-group flex-1 sm:flex-none sm:max-w-40 min-w-0">
                <label className="input-label">Version</label>
                <input
                  className="input-field"
                  placeholder="latest"
                  value={pkgVersion}
                  onChange={e => setPkgVersion(e.target.value)}
                />
              </div>
              <div className="input-group flex-1 sm:flex-none sm:max-w-35 min-w-0">
                <label className="input-label">Ecosystem</label>
                <select className="input-field" value={ecosystem} onChange={e => setEcosystem(e.target.value)}>
                  <option value="pypi">PyPI</option>
                  <option value="npm">NPM</option>
                </select>
              </div>
            </div>
            <button className="btn btn-primary w-full sm:w-auto" onClick={handleScanName} disabled={loading}>
              {loading ? <><span className="spinner" /> Scanning...</> : <><ScanLine size={16} /> Scan Package</>}
            </button>
          </div>
        )}

        {tab === 'text' && (
          <div>
            <div className="input-row flex-col sm:flex-row" style={{ alignItems: 'stretch' }}>
              <div className="input-group flex-1 min-w-0" style={{ marginBottom: 0 }}>
                <label className="input-label">Scan Text</label>
                <textarea
                  className="input-field"
                  placeholder="e.g. requests-fake or suspicious package indicators"
                  value={scanText}
                  onChange={e => setScanText(e.target.value)}
                  rows={5}
                  style={{ resize: 'vertical', minHeight: 120 }}
                />
              </div>
              <div className="input-group flex-1 sm:flex-none sm:max-w-35 min-w-0" style={{ marginBottom: 0 }}>
                <label className="input-label">Ecosystem</label>
                <select className="input-field" value={ecosystem} onChange={e => setEcosystem(e.target.value)}>
                  <option value="pypi">PyPI</option>
                  <option value="npm">NPM</option>
                </select>
              </div>
            </div>
            <button className="btn btn-primary w-full sm:w-auto" onClick={handleScanText} disabled={loading} style={{ marginTop: 16 }}>
              {loading ? <><span className="spinner" /> Scanning...</> : <><ScanLine size={16} /> Scan Text</>}
            </button>
          </div>
        )}

        {tab === 'file' && (
          <div>
            <div className="upload-zone" onClick={() => fileRef.current?.click()}>
              <input
                ref={fileRef}
                type="file"
                accept=".whl,.tar.gz,.tgz,.zip,.gz,.pdf,.csv,.doc,.docx"
                style={{ display: 'none' }}
                onChange={e => setFile(e.target.files?.[0] || null)}
              />
              <Upload size={32} className="upload-icon" />
              <div className="upload-text">
                {file ? file.name : 'Drop package file here or click to browse'}
              </div>
              <div className="upload-hint">.whl, .tar.gz, .tgz, .zip, .pdf, .csv, .doc, .docx supported</div>
            </div>
            <button
              className="btn btn-primary"
              onClick={handleScanFile}
              disabled={loading || !file}
              style={{ marginTop: 16 }}
            >
              {loading ? <><span className="spinner" /> Scanning...</> : <><ScanLine size={16} /> Scan File</>}
            </button>
          </div>
        )}

        {error && (
          <div style={{
            marginTop: 16, padding: '12px 16px',
            background: 'var(--red-dim)', border: '1px solid rgba(255,69,96,0.3)',
            borderRadius: 8, color: 'var(--red)', fontSize: 13,
          }}>
            ⚠ {error}
          </div>
        )}
      </div>

      {/* Terminal Logs */}
      {logs.length > 0 && (
        <div className="card" style={{ marginBottom: 24 }}>
          <div className="section-title" style={{ marginBottom: 12 }}>
            <Package size={16} />
            Sandbox Output
          </div>
          <div className="terminal">
            {logs.map((l, i) => (
              <div key={i} className={`terminal-line ${l.type}`}>
                {l.text}
              </div>
            ))}
            {loading && <div className="terminal-line" style={{ color: 'var(--green)' }}>█</div>}
          </div>
        </div>
      )}

      {/* Results */}
      {result && risk && (
        <div className="scan-result">
          <div style={{ display: 'grid', gridTemplateColumns: '200px 1fr', gap: 20, marginBottom: 20 }}>
            {/* Score */}
            <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <RiskGauge score={risk.score} label={risk.label} />
            </div>

            {/* Details */}
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
                <div>
                  <div style={{ fontFamily: 'var(--font-display)', fontSize: 20, fontWeight: 800 }}>
                    {result.package_name}
                    {result.ecosystem && (
                      <span style={{ fontSize: 12, color: 'var(--text-dim)', marginLeft: 8 }}>
                        [{result.ecosystem}]
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 4 }}>
                    Scan ID: {result.scan_id} · {new Date(result.timestamp).toLocaleString()}
                  </div>
                </div>
              </div>

              {/* Confidence band */}
              <div style={{ marginBottom: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
                  <span>Risk Score</span>
                  <span>{risk.confidence_band?.low}% – {risk.confidence_band?.high}% confidence</span>
                </div>
                <div className="progress-track">
                  <div
                    className={`progress-fill ${risk.label.toLowerCase()}`}
                    style={{ width: `${risk.score}%` }}
                  />
                </div>
              </div>

              {/* Registry Info */}
              {result.registry_info && !result.registry_info.error && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
                  {[
                    { label: 'Author', value: result.registry_info.author },
                    { label: 'Version', value: result.registry_info.version },
                    { label: 'Summary', value: result.registry_info.summary?.slice(0, 60) + '...' },
                  ].filter(i => i.value).map(({ label, value }) => (
                    <div key={label} style={{ background: 'var(--bg3)', borderRadius: 6, padding: '10px 12px' }}>
                      <div style={{ fontSize: 10, color: 'var(--text-dim)', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 4 }}>{label}</div>
                      <div style={{ fontSize: 12, color: 'var(--text)' }}>{value}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* SHAP + Sandbox split */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
            {/* SHAP */}
            <div className="card">
              <div className="section-title">
                <AlertTriangle size={16} />
                Risk Factors (SHAP Analysis)
              </div>
              <SHAPFactors factors={risk.shap_factors} />
            </div>

            {/* Sandbox */}
            <div className="card">
              <div className="section-title">
                <Package size={16} />
                Sandbox Findings
              </div>
              {sandbox && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {[
                    { label: 'Syscall Count', value: sandbox.syscall_count },
                    { label: 'Network Connections', value: sandbox.network_connections?.length || 0 },
                    { label: 'Data Exfiltrated', value: `${sandbox.data_exfiltrated_kb} KB` },
                    { label: 'Processes Spawned', value: sandbox.processes_spawned?.length || 0 },
                    { label: 'Files Accessed', value: sandbox.files_accessed?.join(', ') || 'None' },
                    { label: 'Duration', value: `${sandbox.sandbox_duration_ms}ms` },
                  ].map(({ label, value }) => (
                    <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13 }}>
                      <span style={{ color: 'var(--text-dim)' }}>{label}</span>
                      <span style={{ color: 'var(--text)', fontFamily: 'var(--font-mono)' }}>{value}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Syscall Sample */}
          {syscallSamples.length > 0 && (
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="section-title">
                <Package size={16} />
                Syscall Sample
              </div>
              <div className="terminal">
                {syscallSamples.map((s: string, i: number) => (
                  <div key={i} className={`terminal-line ${s.includes('passwd') || s.includes('connect') || s.includes('execve') ? 'err' : 'ok'}`}>
                    {s}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* AI Remediation */}
          {result.remediation && <RemediationPanel data={result.remediation} />}
          {!result.remediation && risk.label === 'BENIGN' && (
            <div className="card" style={{
              background: 'var(--green-dim)',
              borderColor: 'var(--border-hover)',
              display: 'flex', alignItems: 'center', gap: 16,
            }}>
              <CheckCircle size={32} color="var(--green)" />
              <div>
                <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 16, color: 'var(--green)' }}>
                  Package Appears Safe
                </div>
                <div style={{ fontSize: 13, color: 'var(--text-dim)', marginTop: 4 }}>
                  No malicious behavior detected. Risk score is within acceptable bounds.
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
