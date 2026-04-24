export default function Footer() {
  return (
    <footer style={{
      borderTop: '1px solid var(--border)',
      background: 'var(--bg2)',
      padding: '48px 24px',
      marginTop: 64,
    }}>
      <div style={{
        maxWidth: '1400px',
        margin: '0 auto',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
        gap: 32,
        fontSize: 13,
        color: 'var(--text-dim)',
      }}>
        <div>
          <div style={{ fontWeight: 700, color: 'var(--text)', marginBottom: 12 }}>SCALA-Guard</div>
          <p style={{ lineHeight: 1.6 }}>
            Open-source supply chain security for Python and Node.js ecosystems. Combining behavioral sandboxing, ML, and AI.
          </p>
        </div>
        <div>
          <div style={{ fontWeight: 700, color: 'var(--text)', marginBottom: 12 }}>Features</div>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li><a href="#" style={{ color: 'inherit', textDecoration: 'none' }} onMouseEnter={(e) => e.currentTarget.style.color = 'var(--green)'} onMouseLeave={(e) => e.currentTarget.style.color = 'inherit'}>Scanner</a></li>
            <li><a href="#" style={{ color: 'inherit', textDecoration: 'none' }} onMouseEnter={(e) => e.currentTarget.style.color = 'var(--green)'} onMouseLeave={(e) => e.currentTarget.style.color = 'inherit'}>Batch Audit</a></li>
            <li><a href="#" style={{ color: 'inherit', textDecoration: 'none' }} onMouseEnter={(e) => e.currentTarget.style.color = 'var(--green)'} onMouseLeave={(e) => e.currentTarget.style.color = 'inherit'}>Dashboard</a></li>
          </ul>
        </div>
        <div>
          <div style={{ fontWeight: 700, color: 'var(--text)', marginBottom: 12 }}>Resources</div>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li><a href="#" style={{ color: 'inherit', textDecoration: 'none' }} onMouseEnter={(e) => e.currentTarget.style.color = 'var(--green)'} onMouseLeave={(e) => e.currentTarget.style.color = 'inherit'}>GitHub</a></li>
            <li><a href="#" style={{ color: 'inherit', textDecoration: 'none' }} onMouseEnter={(e) => e.currentTarget.style.color = 'var(--green)'} onMouseLeave={(e) => e.currentTarget.style.color = 'inherit'}>Documentation</a></li>
            <li><a href="#" style={{ color: 'inherit', textDecoration: 'none' }} onMouseEnter={(e) => e.currentTarget.style.color = 'var(--green)'} onMouseLeave={(e) => e.currentTarget.style.color = 'inherit'}>API Docs</a></li>
          </ul>
        </div>
      </div>
      <div style={{
        borderTop: '1px solid var(--border)',
        marginTop: 32,
        paddingTop: 24,
        textAlign: 'center',
        fontSize: 12,
        color: 'var(--text-dim)',
      }}>
        <p>© 2026 SCALA-Guard. All rights reserved.</p>
      </div>
    </footer>
  );
}
