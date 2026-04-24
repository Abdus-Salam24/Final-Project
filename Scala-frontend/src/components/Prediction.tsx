import React, { useEffect, useMemo, useState } from 'react';
import { SlidersHorizontal, FileText, Upload, Sparkles, ShieldAlert, CheckCircle2 } from 'lucide-react';
import { API as DEFAULT_API, apiGetJson, apiPostForm, apiPostJson } from '../api';

interface PredictionResultItem {
  index: number;
  prediction: string;
  confidence: number | null;
  receivedFeatures?: number;
  usedFeatures?: number;
  droppedExtraCount?: number;
}

interface PredictionResponse {
  results?: PredictionResultItem[];
  features?: string[];
}

interface BackendPredictionResponse {
  label?: string;
  risk_score?: number;
  confidence?: number;
  received_features?: number;
  used_features?: number;
}

interface CsvPredictionItem {
  row: number;
  label: string;
  risk_score?: number;
  received_features?: number;
  used_features?: number;
  dropped_extra_values?: number[];
}

interface CsvPredictionResponse {
  filename?: string;
  total_predicted?: number;
  total_skipped?: number;
  skipped_rows?: number[];
  predictions?: CsvPredictionItem[];
}

interface PredictionMeta {
  filename?: string;
  totalPredicted?: number;
  totalSkipped?: number;
  skippedRows?: number[];
}

type InputMethod = 'manual' | 'csv' | 'file';

function encodeStringToken(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash * 31 + input.charCodeAt(i)) >>> 0;
  }
  // Keep encoded values in a bounded range for model stability.
  return hash % 100000;
}

function toNumericArray(values: string[]): number[] {
  return values.map((raw) => {
    const value = raw.trim();
    if (!value) {
      return 0;
    }

    const numeric = Number(value);
    if (!Number.isNaN(numeric) && Number.isFinite(numeric)) {
      return numeric;
    }

    const lower = value.toLowerCase();
    if (lower === 'true') return 1;
    if (lower === 'false') return 0;
    if (lower === 'null' || lower === 'undefined' || lower === 'na' || lower === 'n/a') {
      return 0;
    }

    return encodeStringToken(value);
  });
}

function normalizePredictionPayload(json: unknown): { items: PredictionResultItem[]; meta: PredictionMeta | null } {
  const payload = json as PredictionResponse & BackendPredictionResponse & CsvPredictionResponse;

  if (Array.isArray(payload.results)) {
    return { items: payload.results, meta: null };
  }

  if (Array.isArray(payload.predictions)) {
    const items = payload.predictions.map((item, idx) => ({
      index: item.row ?? idx + 1,
      prediction: item.label,
      confidence: item.risk_score ?? null,
      receivedFeatures: item.received_features,
      usedFeatures: item.used_features,
      droppedExtraCount: item.dropped_extra_values?.length ?? 0,
    }));

    return {
      items,
      meta: {
        filename: payload.filename,
        totalPredicted: payload.total_predicted,
        totalSkipped: payload.total_skipped,
        skippedRows: payload.skipped_rows,
      },
    };
  }

  if (payload.label) {
    return {
      items: [
        {
          index: 1,
          prediction: payload.label,
          confidence: payload.confidence ?? payload.risk_score ?? null,
          receivedFeatures: payload.received_features,
          usedFeatures: payload.used_features,
        },
      ],
      meta: null,
    };
  }

  return { items: [], meta: null };
}

const Prediction: React.FC = () => {
  const [features, setFeatures] = useState<string[]>([]);
  const [values, setValues] = useState<Record<string, string>>({});
  const [csvRow, setCsvRow] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResultItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [resultMeta, setResultMeta] = useState<PredictionMeta | null>(null);
  const [loadingFeatures, setLoadingFeatures] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [activeMethod, setActiveMethod] = useState<InputMethod>('manual');

  useEffect(() => {
    let mounted = true;
    setLoadingFeatures(true);
    apiGetJson<PredictionResponse>('/features')
      .then((j: PredictionResponse) => { if (mounted) setFeatures(j.features || []) })
      .catch((error: unknown) => { if (mounted) setError('Failed to load features: ' + String(error)); })
      .finally(()=>{ if (mounted) setLoadingFeatures(false) });
    return ()=>{ mounted = false; };
  }, []);

  const summary = useMemo(() => {
    const items = result || [];
    if (items.length === 0) {
      return { total: 0, malicious: 0, benign: 0, avgConfidence: 0 };
    }

    const malicious = items.filter((item) => item.prediction.toUpperCase() === 'MALICIOUS').length;
    const benign = items.filter((item) => item.prediction.toUpperCase() !== 'MALICIOUS').length;
    const confidenceValues = items
      .map((item) => item.confidence)
      .filter((value): value is number => value !== null);
    const avgConfidence = confidenceValues.length
      ? Math.round(confidenceValues.reduce((acc, n) => acc + n, 0) / confidenceValues.length)
      : 0;

    return { total: items.length, malicious, benign, avgConfidence };
  }, [result]);

  const selectMethod = (method: InputMethod) => {
    setActiveMethod(method);
    setError(null);
    setResult(null);
    setResultMeta(null);

    if (method !== 'manual') {
      setValues({});
    }
    if (method !== 'csv') {
      setCsvRow('');
    }
    if (method !== 'file') {
      setFile(null);
    }
  };

  const handleChange = (k: string, v: string) => {
    setValues((s) => ({ ...s, [k]: v }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setResultMeta(null);
    setSubmitting(true);

    try {
      let resp: Response;

      if (activeMethod === 'file') {
        if (!file) {
          throw new Error('Please upload a CSV file before prediction.');
        }
        const fd = new FormData();
        fd.append('file', file);
        resp = await apiPostForm('/predict/csv', fd);
      } else if (activeMethod === 'csv') {
        if (!csvRow.trim()) {
          throw new Error('Please provide a CSV row before prediction.');
        }
        const csvValues = csvRow.split(',').map((v) => v.trim()).filter(Boolean);
        const numericData = toNumericArray(csvValues);
        resp = await apiPostJson('/predict', { data: numericData });
      } else if (activeMethod === 'manual') {
        if (features.length === 0) throw new Error('Feature list not loaded');
        const vals: string[] = features.map((f) => (values[f] || '').toString().trim());
        if (vals.some((v) => v === '')) throw new Error('Please fill all feature fields');
        const numericData = toNumericArray(vals);
        resp = await apiPostJson('/predict', { data: numericData });
      } else {
        throw new Error('Please select an input method.');
      }

      const json = await resp.json();
      const normalized = normalizePredictionPayload(json);
      setResult(normalized.items);
      setResultMeta(normalized.meta);
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      setError(message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
        <div>
          <h1 className="page-title">ML <span>Prediction</span></h1>
          <p className="page-subtitle">Run risk prediction using manual features, CSV row, or CSV file upload.</p>
        </div>
        <div className="card-sm" style={{ minWidth: 230 }}>
          <div style={{ fontSize: 11, letterSpacing: 1, textTransform: 'uppercase', color: 'var(--text-dim)', marginBottom: 6 }}>
            Active Endpoint
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-bright)', fontSize: 12, wordBreak: 'break-all' }}>
            {DEFAULT_API}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.35fr 1fr', gap: 20 }}>
        <div className="card">
          <div style={{ marginBottom: 14 }}>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
              Input Method
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 10 }}>
              <button
                type="button"
                className={`btn ${activeMethod === 'manual' ? 'btn-primary' : 'btn-outline'}`}
                onClick={() => selectMethod('manual')}
              >
                <SlidersHorizontal size={14} /> Manual
              </button>
              <button
                type="button"
                className={`btn ${activeMethod === 'csv' ? 'btn-primary' : 'btn-outline'}`}
                onClick={() => selectMethod('csv')}
              >
                <FileText size={14} /> CSV Row
              </button>
              <button
                type="button"
                className={`btn ${activeMethod === 'file' ? 'btn-primary' : 'btn-outline'}`}
                onClick={() => selectMethod('file')}
              >
                <Upload size={14} /> CSV File
              </button>
            </div>
          </div>

          {loadingFeatures && activeMethod === 'manual' && (
            <div style={{ marginBottom: 16, color: 'var(--text-dim)', fontSize: 13 }}>
              Loading feature definitions...
            </div>
          )}

          {error && (
            <div style={{
              marginBottom: 16,
              padding: '10px 12px',
              borderRadius: 8,
              border: '1px solid rgba(239,68,68,.35)',
              background: 'var(--red-dim)',
              color: 'var(--red)',
              fontSize: 13,
            }}>
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit}>
            {activeMethod === 'manual' && (
              <div>
                <div className="input-row" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
                  {features.map((f) => (
                    <div key={f} className="input-group" style={{ marginBottom: 8 }}>
                      <label className="input-label">{f}</label>
                      <input
                        className="input-field"
                        type="number"
                        step="any"
                        value={values[f] || ''}
                        onChange={(e) => handleChange(f, e.target.value)}
                        placeholder="Enter numeric value"
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeMethod === 'csv' && (
              <div className="input-group">
                <label className="input-label">CSV Row</label>
                <textarea
                  className="input-field"
                  rows={5}
                  value={csvRow}
                  onChange={(e) => setCsvRow(e.target.value)}
                  placeholder="e.g. 0.01,1,443,0,12.4"
                />
              </div>
            )}

            {activeMethod === 'file' && (
              <div className="input-group">
                <label className="input-label">Upload CSV File</label>
                <div style={{
                  border: '1px dashed var(--border-hover)',
                  borderRadius: 10,
                  padding: 16,
                  background: 'var(--bg2)',
                }}>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => setFile(e.target.files ? e.target.files[0] : null)}
                  />
                  <div style={{ marginTop: 8, color: 'var(--text-dim)', fontSize: 12 }}>
                    {file ? `Selected: ${file.name}` : 'Choose a .csv file containing one or more samples'}
                  </div>
                </div>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginTop: 12, flexWrap: 'wrap' }}>
              <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>
                Mode: <strong style={{ color: 'var(--text-bright)' }}>{activeMethod.toUpperCase()}</strong>
              </div>
              <button className="btn btn-primary" type="submit" disabled={submitting}>
                <Sparkles size={14} />
                {submitting ? 'Predicting...' : 'Run Prediction'}
              </button>
            </div>
          </form>
        </div>

        <div className="card" style={{ alignSelf: 'start' }}>
          <div className="section-title" style={{ marginBottom: 12 }}>
            <Sparkles size={16} />
            Prediction Results
          </div>

          {resultMeta && (
            <div className="card-sm" style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, letterSpacing: 1, textTransform: 'uppercase', color: 'var(--text-dim)', marginBottom: 6 }}>
                File Summary
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-bright)', marginBottom: 4 }}>
                File: <strong>{resultMeta.filename || 'N/A'}</strong>
              </div>
              <div style={{ fontSize: 12, color: 'var(--text)', display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                <span>Predicted: <strong>{resultMeta.totalPredicted ?? (result?.length || 0)}</strong></span>
                <span>Skipped: <strong>{resultMeta.totalSkipped ?? 0}</strong></span>
              </div>
              {resultMeta.skippedRows && resultMeta.skippedRows.length > 0 && (
                <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-dim)' }}>
                  Skipped Rows: {resultMeta.skippedRows.join(', ')}
                </div>
              )}
            </div>
          )}

          <div className="stat-grid" style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', marginBottom: 14 }}>
            <div className="stat-card">
              <div className="stat-value">{summary.total}</div>
              <div className="stat-label">Samples</div>
            </div>
            <div className="stat-card danger">
              <div className="stat-value" style={{ color: 'var(--red)' }}>{summary.malicious}</div>
              <div className="stat-label">Malicious</div>
            </div>
            <div className="stat-card">
              <div className="stat-value" style={{ color: 'var(--green)' }}>{summary.benign}</div>
              <div className="stat-label">Benign</div>
            </div>
            <div className="stat-card warn">
              <div className="stat-value" style={{ color: 'var(--yellow)' }}>{summary.avgConfidence}%</div>
              <div className="stat-label">Avg Confidence</div>
            </div>
          </div>

          {!result || result.length === 0 ? (
            <div style={{ textAlign: 'center', color: 'var(--text-dim)', fontSize: 13, padding: '20px 0' }}>
              Prediction output will appear here after submission.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, maxHeight: 440, overflowY: 'auto', paddingRight: 2 }}>
              {result.map((r) => {
                const label = r.prediction.toUpperCase();
                const isMalicious = label === 'MALICIOUS';

                return (
                  <div
                    key={r.index}
                    style={{
                      borderRadius: 10,
                      padding: 12,
                      border: isMalicious ? '1px solid rgba(239,68,68,.35)' : '1px solid rgba(16,185,129,.35)',
                      background: isMalicious ? 'var(--red-dim)' : 'var(--green-dim)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10 }}>
                      <div style={{ fontSize: 13, color: 'var(--text-dim)' }}>Sample #{r.index}</div>
                      <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 700, color: isMalicious ? 'var(--red)' : 'var(--green)' }}>
                        {isMalicious ? <ShieldAlert size={14} /> : <CheckCircle2 size={14} />}
                        {label}
                      </div>
                    </div>

                    <div style={{ marginTop: 8, fontSize: 13, color: 'var(--text-bright)' }}>
                      Confidence: <strong>{r.confidence !== null ? `${r.confidence}%` : 'N/A'}</strong>
                    </div>

                    {(r.receivedFeatures !== undefined || r.usedFeatures !== undefined) && (
                      <div style={{ marginTop: 6, display: 'flex', gap: 10, flexWrap: 'wrap', fontSize: 11, color: 'var(--text-dim)' }}>
                        {r.receivedFeatures !== undefined && <span>Received: {r.receivedFeatures}</span>}
                        {r.usedFeatures !== undefined && <span>Used: {r.usedFeatures}</span>}
                        {r.droppedExtraCount !== undefined && <span>Dropped Extra: {r.droppedExtraCount}</span>}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Prediction;
