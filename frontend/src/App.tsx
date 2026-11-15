import { useEffect, useRef, useState } from "react";
import { ShieldCheck } from "lucide-react";

import { AboutSection } from "./components/AboutSection";
import { DetectionResult } from "./components/DetectionResult";
import { Hero } from "./components/Hero";
import { ImageUploader } from "./components/ImageUploader";
import { Navbar } from "./components/Navbar";
import { detectImage } from "./services/api";
import type { DetectionResponse } from "./types/detection";

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
const ALLOWED_EXTENSIONS = /\.(jpe?g|png|webp)$/i;

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const detectorRef = useRef<HTMLElement>(null);
  const previewUrlRef = useRef("");

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  const clearPreview = () => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = "";
    }

    setPreview("");
  };

  const updatePreview = (selected: File) => {
    clearPreview();

    const url = URL.createObjectURL(selected);
    previewUrlRef.current = url;
    setPreview(url);
  };

  const selectImage = (selected?: File) => {
    if (!selected) return;

    setError("");
    setResult(null);

    if (!ALLOWED_EXTENSIONS.test(selected.name)) {
      setError("Please select a JPG, JPEG, PNG or WEBP image.");
      return;
    }

    if (selected.type && !ALLOWED_TYPES.includes(selected.type)) {
      setError("Please select a JPG, JPEG, PNG or WEBP image.");
      return;
    }

    if (selected.size === 0) {
      setError("The selected file is empty.");
      return;
    }

    if (selected.size > MAX_FILE_SIZE) {
      setError("Maximum image size is 10 MB.");
      return;
    }

    setFile(selected);
    updatePreview(selected);
  };

  const analyzeImage = async () => {
    if (!file || loading) return;

    setLoading(true);
    setError("");

    try {
      const response = await detectImage(file);
      setResult(response);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(`Image detection failed: ${message}`);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    clearPreview();
    setFile(null);
    setResult(null);
    setError("");

    detectorRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div id="top" className="min-h-screen bg-[#080b14] text-white">
      <Navbar />

      <main>
        <Hero />

        <section
          ref={detectorRef}
          id="detector"
          className="scroll-mt-24 px-4 pb-24 sm:px-6"
        >
          <div className="mx-auto max-w-5xl">
            <h2 className="mb-3 text-center text-3xl font-bold">
              Image Detection
            </h2>

            <p className="mb-10 text-center text-sm text-slate-400">
              Upload an image to begin your analysis.
            </p>

            <div className="grid gap-7 lg:grid-cols-2">
              <ImageUploader
                file={file}
                preview={preview}
                error={error}
                loading={loading}
                onSelect={selectImage}
                onAnalyze={analyzeImage}
                onReset={reset}
              />

              <DetectionResult
                result={result}
                loading={loading}
                onReset={reset}
              />
            </div>
          </div>
        </section>

        <AboutSection />
      </main>

      <footer className="border-t border-white/10 px-4 py-7 text-center text-xs text-slate-500">
        <p>© 2026 TrueLens AI · AI Fake Photo Detector</p>
        <p className="mt-2 flex items-center justify-center gap-1.5">
          <ShieldCheck size={13} />
          EfficientNetB0 · TensorFlow · FastAPI
        </p>
      </footer>
    </div>
  );
}

export default App;
