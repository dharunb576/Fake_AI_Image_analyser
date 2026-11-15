import { Binary, Layers, ScanLine, ShieldCheck } from "lucide-react";

import { ModelInfoCard } from "./ModelInfoCard";

const capabilities = [
  {
    icon: Layers,
    title: "EfficientNetB0",
    description:
      "A compound-scaled convolutional neural network that balances depth, width and resolution for accurate image classification.",
  },
  {
    icon: ScanLine,
    title: "224 × 224 RGB",
    description:
      "Uploaded images are converted to RGB and resized to the exact input shape used during training.",
  },
  {
    icon: Binary,
    title: "Binary classification",
    description:
      "A sigmoid output produces a FAKE probability. Values at or above 0.5 are classified as FAKE, otherwise REAL.",
  },
];

export function AboutSection() {
  return (
    <section
      id="about"
      className="border-t border-white/10 bg-[#0b1020] px-4 py-16 sm:px-6"
    >
      <div className="mx-auto max-w-5xl">
        <div className="text-center">
          <ShieldCheck className="mx-auto text-violet-400" size={35} />

          <h2 className="mt-5 text-3xl font-bold">Powered by Deep Learning</h2>

          <p className="mx-auto mt-4 max-w-2xl text-sm leading-7 text-slate-400">
            TrueLens AI uses a trained EfficientNetB0 model to classify images
            as REAL or AI-generated. Each upload is validated, converted to RGB,
            resized to 224 × 224 pixels, and passed through the model for
            inference.
          </p>
        </div>

        <div className="mt-10 grid gap-4 md:grid-cols-3">
          {capabilities.map(({ icon: Icon, title, description }) => (
            <div
              key={title}
              className="rounded-2xl border border-white/10 bg-[#111827] p-5"
            >
              <div className="mb-4 inline-flex rounded-xl bg-violet-500/10 p-3 text-violet-400">
                <Icon size={22} />
              </div>

              <h3 className="font-semibold">{title}</h3>

              <p className="mt-2 text-sm leading-6 text-slate-400">
                {description}
              </p>
            </div>
          ))}
        </div>

        <div className="mt-6">
          <ModelInfoCard />
        </div>

        <p className="mt-6 text-center text-xs leading-6 text-slate-500">
          This tool is intended for educational and experimental purposes.
          Predictions are not guaranteed to be accurate for every image.
        </p>
      </div>
    </section>
  );
}
