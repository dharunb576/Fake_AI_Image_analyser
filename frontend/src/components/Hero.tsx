import { Image as ImageIcon, ShieldCheck, Sparkles, Zap } from "lucide-react";

export function Hero() {
  return (
    <section className="relative overflow-hidden px-4 pb-16 pt-16 text-center sm:px-6 sm:pt-24">
      <div className="pointer-events-none absolute left-1/2 top-0 h-72 w-72 -translate-x-1/2 rounded-full bg-violet-600/20 blur-[110px]" />

      <div className="relative mx-auto max-w-4xl">
        <div className="mb-7 inline-flex items-center gap-2 rounded-full border border-violet-400/30 bg-violet-500/10 px-4 py-2 text-sm text-violet-300">
          <Sparkles size={16} />
          AI-powered image detection
        </div>

        <h1 className="text-4xl font-extrabold leading-tight sm:text-6xl">
          Is Your Image{" "}
          <span className="bg-gradient-to-r from-violet-400 to-cyan-400 bg-clip-text text-transparent">
            Real or AI?
          </span>
        </h1>

        <p className="mx-auto mt-6 max-w-2xl leading-8 text-slate-400">
          Upload any supported image and let our deep learning model analyze
          whether it is a real photograph or AI-generated.
        </p>

        <div className="mt-8 flex flex-wrap justify-center gap-x-6 gap-y-3 text-sm text-slate-400">
          <span className="flex items-center gap-2">
            <ShieldCheck className="text-emerald-400" size={17} />
            EfficientNetB0
          </span>

          <span className="flex items-center gap-2">
            <Zap className="text-yellow-400" size={17} />
            Deep Learning
          </span>

          <span className="flex items-center gap-2">
            <ImageIcon className="text-violet-400" size={17} />
            RGB Analysis
          </span>
        </div>

        <div className="mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row">
          <a
            href="#detector"
            className="w-full rounded-xl bg-violet-600 px-7 py-3.5 font-semibold transition hover:bg-violet-500 sm:w-auto"
          >
            Analyze an Image
          </a>
          <a
            href="#about"
            className="w-full rounded-xl border border-white/10 px-7 py-3.5 font-semibold text-slate-300 transition hover:border-violet-400 hover:text-white sm:w-auto"
          >
            How It Works
          </a>
        </div>
      </div>
    </section>
  );
}
