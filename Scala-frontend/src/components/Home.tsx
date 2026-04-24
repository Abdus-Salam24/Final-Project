
import { useNavigate } from 'react-router-dom';
import { Shield, ScanLine, Layers, Zap, Lock, Globe, ArrowRight } from 'lucide-react';

export default function Home() {
  const navigate = useNavigate();

  const features = [
    {
      icon: <ScanLine size={20} />,
      title: 'Behavioral Sandbox',
      desc: 'Docker-isolated execution with strace syscall tracing and tcpdump network capture.',
    },
    {
      icon: <Zap size={20} />,
      title: 'ML Risk Scoring',
      desc: 'Random Forest classifier assigns malicious probability with SHAP explainability.',
    },
    {
      icon: <Shield size={20} />,
      title: 'AI Remediation',
      desc: 'DeepSeek LLM generates fix steps, safe alternatives, and CVE mappings.',
    },
    {
      icon: <Layers size={20} />,
      title: 'Batch Audit',
      desc: 'Scan entire requirements.txt or package.json in one submission.',
    },
    {
      icon: <Globe size={20} />,
      title: 'Registry Support',
      desc: 'Analyze packages directly from PyPI and NPM registries by name.',
    },
    {
      icon: <Lock size={20} />,
      title: 'CVE Mapping',
      desc: 'Cross-reference scan results against known vulnerability databases.',
    },
  ];

  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="pt-6 pb-16 sm:pt-10 sm:pb-24 md:pt-14 md:pb-32 px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <div className="inline-flex items-center justify-center gap-2 px-3 sm:px-4 py-2 bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 rounded-full mb-6 sm:mb-8 text-xs sm:text-sm font-semibold">
            🔒 Open-Source Security Intelligence
          </div>

          <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl xl:text-7xl font-black text-gray-900 dark:text-white mb-4 sm:mb-6 leading-tight">
            Detect <span className="text-emerald-600 dark:text-emerald-400">Malicious</span><br className="hidden sm:block" />
            Packages <span className="text-gray-400 dark:text-gray-600">Before</span><br className="hidden sm:block" />
            They Strike
          </h1>


          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center mb-12 sm:mb-16">
            <button 
              className="btn btn-primary px-4 sm:px-6 py-2 sm:py-3 text-sm sm:text-base font-semibold w-full sm:w-auto"
              onClick={() => navigate('/scanner')}
            >
              <ScanLine size={18} />
              Scan a Package
            </button>
            <button 
              className="btn btn-outline px-4 sm:px-6 py-2 sm:py-3 text-sm sm:text-base font-semibold w-full sm:w-auto"
              onClick={() => navigate('/batch')}
            >
              <Layers size={18} />
              Batch Audit
            </button>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 sm:gap-4 md:gap-6 max-w-3xl mx-auto px-2">
            {[
              { num: '3-Layer', label: 'Analysis Pipeline' },
              { num: 'ML+AI', label: 'Hybrid Detection' },
              { num: 'PyPI+NPM', label: 'Registry Support' },
              { num: 'Real-time', label: 'Remediation' },
            ].map(({ num, label }) => (
              <div key={label}>
                <div className="text-3xl font-black text-emerald-600 dark:text-emerald-400">{num}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400 font-medium">{label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-12 sm:py-16 md:py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-2xl sm:text-3xl md:text-4xl font-black text-gray-900 dark:text-white mb-8 sm:mb-12 text-center">
            Platform Capabilities
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
            {features.map(({ icon, title, desc }) => (
              <div 
                key={title} 
                className="card hover:shadow-lg hover:-translate-y-1 transition-all duration-300 group"
              >
                <div className="w-12 h-12 bg-emerald-100 dark:bg-emerald-950 border border-emerald-200 dark:border-emerald-800 rounded-lg flex items-center justify-center text-emerald-600 dark:text-emerald-400 mb-4 group-hover:bg-emerald-200 dark:group-hover:bg-emerald-900 transition-colors">
                  {icon}
                </div>
                <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-3">
                  {title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed">
                  {desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pipeline */}
      <section className="py-12 sm:py-16 md:py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-emerald-50 to-white dark:from-slate-900 dark:to-slate-800">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-2xl sm:text-3xl md:text-4xl font-black text-gray-900 dark:text-white mb-8 sm:mb-12 text-center">
            Analysis Pipeline
          </h2>
          <div className="card overflow-x-auto">
            <div className="flex items-center justify-between gap-1 sm:gap-2 py-4 sm:py-6 min-w-max md:min-w-fit">
              {[
                { label: 'Package Input', sub: 'NPM / PyPI / Upload', color: 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-100' },
                { label: 'Docker Sandbox', sub: 'strace + tcpdump', color: 'bg-blue-200 dark:bg-blue-900 text-blue-800 dark:text-blue-100' },
                { label: 'ML Classifier', sub: 'Random Forest + SHAP', color: 'bg-yellow-200 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-100' },
                { label: 'LLM Remediation', sub: 'DeepSeek AI', color: 'bg-emerald-200 dark:bg-emerald-900 text-emerald-800 dark:text-emerald-100' },
                { label: 'Dashboard', sub: 'Visual Results', color: 'bg-emerald-200 dark:bg-emerald-900 text-emerald-800 dark:text-emerald-100' },
              ].map(({ label, sub, color }, i, arr) => (
                <div key={label} className="flex items-center flex-shrink-0">
                  <div className="text-center px-4">
                    <div className={`w-12 h-12 rounded-full ${color} flex items-center justify-center font-bold text-base mb-3 mx-auto border-2 border-current border-opacity-30`}>
                      {i + 1}
                    </div>
                    <div className="text-sm font-bold text-gray-900 dark:text-white text-center whitespace-nowrap">{label}</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400 text-center whitespace-nowrap">{sub}</div>
                  </div>
                  {i < arr.length - 1 && (
                    <div className="hidden md:block w-8 h-0.5 bg-gradient-to-r from-gray-300 to-emerald-300 dark:from-gray-600 dark:to-emerald-600 flex-shrink-0" />
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-12 sm:py-16 md:py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl sm:rounded-2xl p-6 sm:p-12 md:p-16 text-center text-white shadow-xl">
          <h2 className="text-2xl sm:text-3xl md:text-4xl font-black mb-4 sm:mb-6">
            Ready to Secure Your Supply Chain?
          </h2>
          <p className="text-base sm:text-lg text-emerald-100 mb-6 sm:mb-8">
            Start scanning packages today and protect your dependencies from malicious code.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center">
            <button 
              className="inline-flex items-center justify-center gap-2 px-6 sm:px-8 py-2 sm:py-3 bg-white text-emerald-600 font-bold rounded-lg hover:bg-gray-100 transition-colors w-full sm:w-auto"
              onClick={() => navigate('/scanner')}
            >
              Get Started
              <ArrowRight size={18} />
            </button>
            <button 
              className="inline-flex items-center justify-center gap-2 px-6 sm:px-8 py-2 sm:py-3 bg-emerald-600 border-2 border-white text-white font-bold rounded-lg hover:bg-emerald-700 transition-colors w-full sm:w-auto"
              onClick={() => navigate('/dashboard')}
            >
              View Dashboard
              <ArrowRight size={18} />
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
