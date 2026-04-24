export default function Footer() {
  return (
    <footer className="border-t border-slate-200/80 dark:border-slate-700/70 bg-white/80 dark:bg-slate-900/80 backdrop-blur py-10 sm:py-12 md:py-14">
      <div className="max-w-330 mx-auto px-3 sm:px-5 md:px-8 lg:px-10">
        <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6 md:gap-8 mb-6 sm:mb-8">
          {/* Brand */}
          <div className="col-span-2 sm:col-span-1">
            <h3 className="font-black text-slate-900 dark:text-white mb-3 text-base sm:text-lg tracking-tight">SCALA-Guard</h3>
            <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
              Open-source supply chain security for Python and Node.js ecosystems. Combining behavioral sandboxing, ML, and AI.
            </p>
          </div>

          {/* Links */}
          <div>
            <h4 className="font-semibold text-slate-900 dark:text-white mb-3 sm:mb-4 text-xs sm:text-sm uppercase tracking-wide">Features</h4>
            <ul className="space-y-1.5 sm:space-y-2">
              {['Scanner', 'Batch Audit', 'Dashboard', 'History'].map(link => (
                <li key={link}>
                  <a
                    href="#"
                    className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 hover:text-emerald-600 dark:hover:text-emerald-300 transition-colors"
                  >
                    {link}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="font-semibold text-slate-900 dark:text-white mb-3 sm:mb-4 text-xs sm:text-sm uppercase tracking-wide">Resources</h4>
            <ul className="space-y-1.5 sm:space-y-2">
              {['Documentation', 'GitHub', 'API Reference', 'Support'].map(link => (
                <li key={link}>
                  <a
                    href="#"
                    className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 hover:text-emerald-600 dark:hover:text-emerald-300 transition-colors"
                  >
                    {link}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h4 className="font-semibold text-slate-900 dark:text-white mb-3 sm:mb-4 text-xs sm:text-sm uppercase tracking-wide">Legal</h4>
            <ul className="space-y-1.5 sm:space-y-2">
              {['Privacy', 'Terms', 'License', 'Security'].map(link => (
                <li key={link}>
                  <a
                    href="#"
                    className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 hover:text-emerald-600 dark:hover:text-emerald-300 transition-colors"
                  >
                    {link}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Divider */}
        <div className="border-t border-slate-200/80 dark:border-slate-700/70 pt-6 sm:pt-8">
          <p className="text-center text-xs sm:text-sm text-slate-600 dark:text-slate-400">
            © 2026 SCALA-Guard. All rights reserved. Building secure supply chains, one package at a time.
          </p>
        </div>
      </div>
    </footer>
  );
}
