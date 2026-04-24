import { useEffect, useState } from 'react';
import { LayoutDashboard, RefreshCw, TrendingUp, Shield, AlertTriangle } from 'lucide-react';
import { apiGetJson } from '../api';
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts';


const COLORS = {
  malicious: '#ff4560',
  suspicious: '#ffc107',
  benign: '#00c896',
};

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ name: string; value: number; fill?: string }>;
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'var(--bg2)', border: '1px solid var(--border)',
      borderRadius: 8, padding: '10px 14px', fontSize: 12,
    }}>
      <div style={{ color: 'var(--text)', fontWeight: 700 }}>{payload[0].name}</div>
      <div style={{ color: payload[0].fill || 'var(--green)', marginTop: 4 }}>
        {payload[0].value} scans
      </div>
    </div>
  );
}

interface StatsData {
  total_scans: number;
  malicious_found: number;
  suspicious_found: number;
  benign: number;
  threat_rate: number;
}

interface ScanData {
  scan_id: string;
  package_name?: string;
  timestamp: string;
  risk?: { score?: number; label: string; shap_factors?: Array<{ impact: number; type: string; factor: string }> };
}

export default function Dashboard() {
  const [stats, setStats] = useState<StatsData | null>(null);
  const [history, setHistory] = useState<ScanData[]>([]);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    try {
      const [s, h] = await Promise.all([
        apiGetJson<StatsData>('/stats'),
        apiGetJson<{ scans?: ScanData[] }>('/history'),
      ]);
      setStats(s);
      setHistory((h.scans || []).slice(0, 50));
    } catch {
      // API offline – show demo data
      setStats({ total_scans: 0, malicious_found: 0, suspicious_found: 0, benign: 0, threat_rate: 0 } as StatsData);
      setHistory([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh', gap: 12 }}>
      <span className="spinner" style={{ width: 24, height: 24 }} />
      <span style={{ color: 'var(--text-dim)' }}>Loading dashboard...</span>
    </div>
  );

  const pieData = [
    { name: 'Malicious', value: stats?.malicious_found || 0, color: COLORS.malicious },
    { name: 'Suspicious', value: stats?.suspicious_found || 0, color: COLORS.suspicious },
    { name: 'Benign', value: stats?.benign || 0, color: COLORS.benign },
  ];

  // Build bar chart from history — group by day
  const dayMap: Record<string, { day: string; malicious: number; suspicious: number; benign: number }> = {};
  history.forEach(scan => {
    const day = new Date(scan.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    if (!dayMap[day]) dayMap[day] = { day, malicious: 0, suspicious: 0, benign: 0 };
    const label = (scan.risk?.label || '').toLowerCase();
    if (label === 'malicious') dayMap[day].malicious++;
    else if (label === 'suspicious') dayMap[day].suspicious++;
    else dayMap[day].benign++;
  });
  const barData = Object.values(dayMap).slice(-7);

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="page-title">Risk <span>Dashboard</span></h1>
          <p className="page-subtitle">Aggregated threat intelligence from all scans</p>
        </div>
        <button className="btn btn-outline" onClick={load} style={{ gap: 6 }}>
          <RefreshCw size={14} />
          Refresh
        </button>
      </div>

      {/* Stat Cards */}
      <div className="stat-grid">
        <div className="stat-card">
          <div className="stat-value">{stats?.total_scans || 0}</div>
          <div className="stat-label">Total Scans</div>
        </div>
        <div className="stat-card danger">
          <div className="stat-value" style={{ color: 'var(--red)' }}>{stats?.malicious_found || 0}</div>
          <div className="stat-label">Malicious</div>
        </div>
        <div className="stat-card warn">
          <div className="stat-value" style={{ color: 'var(--yellow)' }}>{stats?.suspicious_found || 0}</div>
          <div className="stat-label">Suspicious</div>
        </div>
        <div className="stat-card">
          <div className="stat-value" style={{ color: 'var(--green)' }}>{stats?.benign || 0}</div>
          <div className="stat-label">Benign</div>
        </div>
        <div className="stat-card danger">
          <div className="stat-value" style={{ color: 'var(--red)' }}>{stats?.threat_rate || 0}%</div>
          <div className="stat-label">Threat Rate</div>
        </div>
      </div>

      {stats?.total_scans === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '64px 32px' }}>
          <LayoutDashboard size={48} color="var(--text-dim)" style={{ margin: '0 auto 16px' }} />
          <div style={{ fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 700, marginBottom: 8 }}>
            No scans yet
          </div>
          <div style={{ color: 'var(--text-dim)', fontSize: 13 }}>
            Go to Scanner and analyze a package to see data here.
          </div>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 20 }}>
          {/* Pie Chart */}
          <div className="card">
            <div className="section-title">
              <Shield size={16} />
              Threat Distribution
            </div>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={pieData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  innerRadius={40}
                  paddingAngle={3}
                >
                  {pieData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} stroke="transparent" />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', justifyContent: 'center', gap: 20, flexWrap: 'wrap' }}>
              {pieData.map(({ name, color, value }) => (
                <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                  <div style={{ width: 10, height: 10, borderRadius: 2, background: color }} />
                  <span style={{ color: 'var(--text-dim)' }}>{name}:</span>
                  <span style={{ color: 'var(--text)', fontWeight: 700 }}>{value}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Bar Chart */}
          <div className="card">
            <div className="section-title">
              <TrendingUp size={16} />
              Scan Activity (Last 7 Days)
            </div>
            {barData.length > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={barData} barSize={12}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,200,150,0.08)" />
                  <XAxis dataKey="day" tick={{ fontSize: 11, fill: 'var(--text-dim)', fontFamily: 'Space Mono' }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 11, fill: 'var(--text-dim)', fontFamily: 'Space Mono' }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="malicious" name="Malicious" fill={COLORS.malicious} radius={[4,4,0,0]} />
                  <Bar dataKey="suspicious" name="Suspicious" fill={COLORS.suspicious} radius={[4,4,0,0]} />
                  <Bar dataKey="benign" name="Benign" fill={COLORS.benign} radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ height: 240, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)', fontSize: 13 }}>
                No activity data
              </div>
            )}
          </div>
        </div>
      )}

      {/* Recent Threats */}
      {history.filter(s => s.risk?.label !== 'BENIGN').length > 0 && (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="section-title">
            <AlertTriangle size={16} />
            Recent Threats
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Package</th>
                <th>Risk Score</th>
                <th>Label</th>
                <th>Top Factor</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {history.filter(s => s.risk?.label !== 'BENIGN').slice(0, 10).map(s => (
                <tr key={s.scan_id}>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{s.package_name}</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div className="progress-track" style={{ width: 60 }}>
                        <div
                          className={`progress-fill ${s.risk?.label?.toLowerCase()}`}
                          style={{ width: `${s.risk?.score || 0}%` }}
                        />
                      </div>
                      <span style={{ fontSize: 12 }}>{s.risk?.score}%</span>
                    </div>
                  </td>
                  <td>
                    <span className={`risk-badge ${s.risk?.label?.toLowerCase()}`}>
                      {s.risk?.label}
                    </span>
                  </td>
                  <td style={{ fontSize: 12, color: 'var(--text-dim)', maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {s.risk?.shap_factors?.[0]?.factor || '—'}
                  </td>
                  <td style={{ fontSize: 12, color: 'var(--text-dim)' }}>
                    {new Date(s.timestamp).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
