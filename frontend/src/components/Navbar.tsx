import { useState } from "react";
import { Menu, ScanSearch, ShieldCheck, X } from "lucide-react";

const links = [
  { label: "Detector", href: "#detector" },
  { label: "About", href: "#about" },
  { label: "Model", href: "#model" },
];

export function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 border-b border-white/10 bg-[#0b1020]/90 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6">
        <a href="#top" className="flex items-center gap-3">
          <span className="rounded-xl bg-violet-600 p-2.5">
            <ShieldCheck size={22} />
          </span>

          <span>
            <span className="block text-lg font-bold leading-tight">
              TrueLens <span className="text-violet-400">AI</span>
            </span>
            <span className="block text-xs text-slate-400">
              AI Fake Photo Detector
            </span>
          </span>
        </a>

        <div className="hidden items-center gap-7 md:flex">
          {links.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="text-sm text-slate-300 transition hover:text-white"
            >
              {link.label}
            </a>
          ))}

          <a
            href="#detector"
            className="flex items-center gap-2 rounded-lg bg-violet-600 px-4 py-2.5 text-sm font-semibold transition hover:bg-violet-500"
          >
            <ScanSearch size={16} />
            Try Detector
          </a>
        </div>

        <button
          type="button"
          aria-label={open ? "Close menu" : "Open menu"}
          aria-expanded={open}
          onClick={() => setOpen((value) => !value)}
          className="rounded-lg border border-white/10 p-2 md:hidden"
        >
          {open ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {open && (
        <div className="border-t border-white/10 px-4 py-4 md:hidden">
          <div className="flex flex-col gap-3">
            {links.map((link) => (
              <a
                key={link.href}
                href={link.href}
                onClick={() => setOpen(false)}
                className="rounded-lg px-3 py-2 text-sm text-slate-300 hover:bg-white/5 hover:text-white"
              >
                {link.label}
              </a>
            ))}

            <a
              href="#detector"
              onClick={() => setOpen(false)}
              className="mt-1 flex items-center justify-center gap-2 rounded-lg bg-violet-600 px-4 py-2.5 text-sm font-semibold"
            >
              <ScanSearch size={16} />
              Try Detector
            </a>
          </div>
        </div>
      )}
    </nav>
  );
}
