import { useState, useRef, type ReactNode } from 'react';
import { Layers, Upload, ScanLine, AlertTriangle, CheckCircle, XCircle, Download } from 'lucide-react';
import { apiPostForm } from '../api';

interface ScanResult {
  scan_id?: string;
  filename: string;
  ecosystem?: string;
  timestamp?: string;
  summary: {
    total: number;
    malicious: number;
    suspicious: number;
    benign: number;
  };
  results: Array<{
    package: string;
    label: string;
    risk_score: number;
    top_factor: string;
    needs_action?: boolean;
  }>;
}

type BatchApiResponse = Partial<ScanResult> & {
  data?: Partial<ScanResult>;
};

export default function BatchAudit() {
  const [file, setFile] = useState<File | null>(null);
  const [ecosystem, setEcosystem] = useState('pypi');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState('');
  const [progress, setProgress] = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);

  async function handleScan() {
    if (!file) {
      setError('Please upload a requirements.txt, package.json, file');
      return;
    }

    setError('');
    setResult(null);
    setLoading(true);
    setProgress(0);

    const interval = setInterval(() => {
      setProgress(p => Math.min(p + 8, 90));
    }, 250);

    try {
      const form = new FormData();
      form.append('package_file', file);
      form.append('file', file);
      form.append('ecosystem', ecosystem);

      const raw = await apiPostForm<BatchApiResponse>('/analyze/batch', form, {
        fallbackPaths: ['/analyze/batch', '/api/analyze/batch'],
      });

        const payload = raw.data ?? raw;
        const results = Array.isArray(payload.results) ? payload.results : [];
        const summary = payload.summary ?? {
          total: results.length,
          malicious: results.filter(r => r.label === 'MALICIOUS').length,
          suspicious: results.filter(r => r.label === 'SUSPICIOUS').length,
          benign: results.filter(r => r.label === 'BENIGN').length,
        };

        setResult({
          scan_id: payload.scan_id,
          filename: payload.filename || file.name,
          ecosystem: payload.ecosystem || ecosystem,
          timestamp: payload.timestamp,
          summary,
          results,
        });
      setProgress(100);
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Failed to connect to API. Is the backend running?';
      setError(message);
    } finally {
      clearInterval(interval);
      setLoading(false);
    }
  }

  function downloadReport() {
    if (!result) return;
    const lines = [
      `SCALA-Guard Batch Report — ${new Date().toISOString()}`,
      `File: ${result.filename}`,
      `Summary: ${result.summary.total} packages | ${result.summary.malicious} Malicious | ${result.summary.suspicious} Suspicious | ${result.summary.benign} Benign`,
      '',
      ...result.results.map(r =>
        `[${r.label.padEnd(10)}] ${r.package.padEnd(30)} ${r.risk_score}%  ${r.top_factor}`
      ),
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'scala-guard-report.txt';
    a.click();
  }

  const labelIcon: Record<string, ReactNode> = {
    MALICIOUS: <XCircle size={14} color="var(--red)" />,
    SUSPICIOUS: <AlertTriangle size={14} color="var(--yellow)" />,
    BENIGN: <CheckCircle size={14} color="var(--green)" />,
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Batch <span>Audit</span></h1>
        <p className="page-subtitle">Upload requirements.txt, package.json, scan via /analyze/batch</p>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="upload-zone" onClick={() => fileRef.current?.click()}>
          <input
            ref={fileRef}
            type="file"
            accept=".txt,.json,.csv"
            style={{ display: 'none' }}
            onChange={e => setFile(e.target.files?.[0] || null)}
          />
          <Upload size={40} className="upload-icon" />
          <div className="upload-text">
            {file ? file.name : 'Upload all type of files(csv,json,txt,zip,xlsx,docs etc)'}
          </div>
          <div className="upload-hint">
            Backend will analyze all dependencies in one batch run
          </div>
        </div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 16, flexWrap: 'wrap' }}>
          <div className="input-group" style={{ maxWidth: 180, marginBottom: 0 }}>
            <label className="input-label">Ecosystem</label>
            <select className="input-field" value={ecosystem} onChange={e => setEcosystem(e.target.value)}>
              <option value="pypi">PyPI (Python)</option>
              <option value="npm">NPM (Node.js)</option>
            </select>
          </div>
          <button
            className="btn btn-primary"
            onClick={handleScan}
            disabled={loading || !file}
            style={{ alignSelf: 'flex-end' }}
          >
            {loading ? <><span className="spinner" /> Scanning {Math.round(progress)}%...</> : <><ScanLine size={16} /> Start Batch Scan</>}
          </button>
        </div>

        {loading && (
          <div style={{ marginTop: 16 }}>
            <div className="progress-track" style={{ height: 8 }}>
              <div className="progress-fill benign" style={{ width: `${progress}%`, transition: 'width 0.3s ease' }} />
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 6 }}>
              Running backend /analyze/batch...
            </div>
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

      {result && (
        <div className="scan-result">
          <div className="stat-grid" style={{ marginBottom: 20 }}>
            <div className="stat-card">
              <div className="stat-value">{result.summary.total}</div>
              <div className="stat-label">Packages Scanned</div>
            </div>
            <div className="stat-card danger">
              <div className="stat-value" style={{ color: 'var(--red)' }}>{result.summary.malicious}</div>
              <div className="stat-label">Malicious</div>
            </div>
            <div className="stat-card warn">
              <div className="stat-value" style={{ color: 'var(--yellow)' }}>{result.summary.suspicious}</div>
              <div className="stat-label">Suspicious</div>
            </div>
            <div className="stat-card">
              <div className="stat-value" style={{ color: 'var(--green)' }}>{result.summary.benign}</div>
              <div className="stat-label">Benign</div>
            </div>
          </div>

          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <div className="section-title" style={{ marginBottom: 0 }}>
                <Layers size={16} />
                Scan Results — {result.filename}
              </div>
              <button className="btn btn-outline" onClick={downloadReport} style={{ fontSize: 12 }}>
                <Download size={14} />
                Export Report
              </button>
            </div>

            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Package</th>
                  <th>Risk Score</th>
                  <th>Status</th>
                  <th>Top Risk Factor</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {result.results.map((r, i) => (
                  <tr key={r.package}>
                    <td style={{ color: 'var(--text-dim)', fontSize: 12 }}>{i + 1}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{r.package}</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div className="progress-track" style={{ width: 80 }}>
                          <div
                            className={`progress-fill ${r.label.toLowerCase()}`}
                            style={{ width: `${r.risk_score}%` }}
                          />
                        </div>
                        <span style={{ fontSize: 12, minWidth: 36, color: r.label === 'MALICIOUS' ? 'var(--red)' : r.label === 'SUSPICIOUS' ? 'var(--yellow)' : 'var(--green)' }}>
                          {r.risk_score}%
                        </span>
                      </div>
                    </td>
                    <td>
                      <span className={`risk-badge ${r.label.toLowerCase()}`}>
                        {labelIcon[r.label]}
                        {r.label}
                      </span>
                    </td>
                    <td style={{ fontSize: 12, color: 'var(--text-dim)', maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {r.top_factor}
                    </td>
                    <td>
                      {r.needs_action ? (
                        <span style={{
                          fontSize: 11, color: 'var(--red)',
                          background: 'var(--red-dim)', borderRadius: 4, padding: '3px 8px',
                          border: '1px solid rgba(255,69,96,0.3)',
                        }}>
                          Remove
                        </span>
                      ) : (
                        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>✓ OK</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
