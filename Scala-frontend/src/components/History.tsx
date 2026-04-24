import { useEffect, useState, type ReactNode } from 'react';
import { Clock, RefreshCw, Trash2, XCircle, AlertTriangle, CheckCircle } from 'lucide-react';
import { apiFetch, apiGetJson } from '../api';

interface RiskFactor {
  factor: string;
  impact: number;
  type: 'critical' | 'high' | 'low';
}

interface RiskInfo {
  score?: number;
  label: string;
  confidence_band?: { low: number; high: number };
  shap_factors?: RiskFactor[];
}


interface ScanRecord {
  scan_id: string;
  package_name?: string;
  filename?: string;
  ecosystem?: string;
  risk?: RiskInfo;
  timestamp: string;
  summary?: { malicious: number; benign: number };
  remediation?: { summary: string; safe_alternative?: string };
}

export default function History() {
  const [history, setHistory] = useState<ScanRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<ScanRecord | null>(null);

  async function load() {
    setLoading(true);
    try {
      const data = await apiGetJson<{ scans?: ScanRecord[] }>('/history');
      setHistory(data.scans || []);
    } catch {
      setHistory([]);
    } finally {
      setLoading(false);
    }
  }

  async function clearHistory() {
    if (!window.confirm('Clear all scan history?')) return;
    await apiFetch('/history', { method: 'DELETE' });
    setHistory([]); setSelected(null);
  }

  useEffect(() => { load(); }, []);

  const labelIcon: Record<string, ReactNode> = {
    MALICIOUS: <XCircle size={14} color="var(--red)" />,
    SUSPICIOUS: <AlertTriangle size={14} color="var(--yellow)" />,
    BENIGN: <CheckCircle size={14} color="var(--green)" />,
  };

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="page-title">Scan <span>History</span></h1>
          <p className="page-subtitle">{history.length} total scans recorded</p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-outline" onClick={load}><RefreshCw size={14} /></button>
          <button className="btn btn-danger" onClick={clearHistory}><Trash2 size={14} /> Clear</button>
        </div>
      </div>

      {loading ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: 'var(--text-dim)', padding: 32 }}>
          <span className="spinner" /> Loading...
        </div>
      ) : history.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '64px 32px' }}>
          <Clock size={48} color="var(--text-dim)" style={{ margin: '0 auto 16px' }} />
          <div style={{ fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 700, marginBottom: 8 }}>
            No scans yet
          </div>
          <div style={{ color: 'var(--text-dim)', fontSize: 13 }}>
            Your scan history will appear here.
          </div>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: selected ? '1fr 1fr' : '1fr', gap: 20 }}>
          {/* Table */}
          <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Package</th>
                  <th>Ecosystem</th>
                  <th>Risk</th>
                  <th>Label</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {history.map(scan => (
                  <tr
                    key={scan.scan_id}
                    onClick={() => setSelected(scan)}
                    style={{
                      cursor: 'pointer',
                      background: selected?.scan_id === scan.scan_id ? 'var(--green-dim)' : undefined,
                    }}
                  >
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                      {scan.package_name || scan.filename}
                    </td>
                    <td style={{ fontSize: 12, color: 'var(--text-dim)' }}>
                      {scan.ecosystem || '—'}
                    </td>
                    <td>
                      {scan.risk ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <div className="progress-track" style={{ width: 60 }}>
                            <div
                              className={`progress-fill ${scan.risk.label.toLowerCase()}`}
                              style={{ width: `${scan.risk.score}%` }}
                            />
                          </div>
                          <span style={{ fontSize: 12 }}>{scan.risk.score}%</span>
                        </div>
                      ) : (
                        <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>Batch</span>
                      )}
                    </td>
                    <td>
                      {scan.risk ? (
                        <span className={`risk-badge ${scan.risk.label.toLowerCase()}`}>
                          {labelIcon[scan.risk.label]}
                          {scan.risk.label}
                        </span>
                      ) : (
                        <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>
                          {scan.summary?.malicious}M / {scan.summary?.benign}B
                        </span>
                      )}
                    </td>
                    <td style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                      {new Date(scan.timestamp).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Detail Panel */}
          {selected && (
            <div className="card scan-result" style={{ position: 'sticky', top: 32, maxHeight: '80vh', overflowY: 'auto' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 18 }}>
                  {selected.package_name}
                </div>
                <button
                  onClick={() => setSelected(null)}
                  style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer', fontSize: 18 }}
                >
                  ×
                </button>
              </div>

              {selected.risk && (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 16 }}>
                    <div style={{
                      fontFamily: 'var(--font-display)',
                      fontSize: 48, fontWeight: 800,
                      color: selected.risk.label === 'MALICIOUS' ? 'var(--red)' :
                             selected.risk.label === 'SUSPICIOUS' ? 'var(--yellow)' : 'var(--green)',
                    }}>
                      {selected.risk.score}%
                    </div>
                    <div>
                      <span className={`risk-badge ${selected.risk.label.toLowerCase()}`}>
                        {selected.risk.label}
                      </span>
                      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 6 }}>
                        Confidence: {selected.risk.confidence_band?.low}–{selected.risk.confidence_band?.high}%
                      </div>
                    </div>
                  </div>

                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: 11, color: 'var(--text-dim)', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 10 }}>
                      Risk Factors
                    </div>
                    {(selected.risk.shap_factors || []).map((f: RiskFactor, i: number) => (
                      <div key={i} style={{ fontSize: 12, color: 'var(--text)', padding: '6px 0', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-dim)' }}>{f.factor}</span>
                        <span style={{ color: f.type === 'critical' ? 'var(--red)' : 'var(--yellow)' }}>+{f.impact}%</span>
                      </div>
                    ))}
                  </div>
                </>
              )}

              {selected.remediation && (
                <div>
                  <div style={{ fontSize: 11, color: 'var(--text-dim)', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 8 }}>
                    AI Remediation
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text)', lineHeight: 1.7, marginBottom: 12 }}>
                    {selected.remediation.summary}
                  </div>
                  {selected.remediation.safe_alternative && (
                    <div style={{
                      background: 'var(--green-dim)', borderRadius: 6, padding: '10px 12px',
                      fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--green)',
                    }}>
                      ✓ Use: {selected.remediation.safe_alternative}
                    </div>
                  )}
                </div>
              )}

              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 16 }}>
                Scan ID: {selected.scan_id}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
