import { useEffect, useState } from "react";
import { Cpu, LoaderCircle } from "lucide-react";

import { getModelInfo } from "../services/api";
import type { ModelInfo } from "../types/detection";

export function ModelInfoCard() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    getModelInfo()
      .then((data) => {
        if (active) setInfo(data);
      })
      .catch((err: unknown) => {
        if (active) {
          setError(
            err instanceof Error ? err.message : "Unable to load model info."
          );
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  return (
    <div
      id="model"
      className="rounded-2xl border border-white/10 bg-[#111827] p-5 text-left sm:p-6"
    >
      <h3 className="flex items-center gap-3 text-lg font-semibold">
        <Cpu className="text-violet-400" size={20} />
        Model Information
      </h3>

      {loading && (
        <div className="flex items-center gap-3 py-8 text-sm text-slate-400">
          <LoaderCircle size={18} className="animate-spin" />
          Loading model details...
        </div>
      )}

      {!loading && error && (
        <p className="mt-5 rounded-lg bg-red-500/10 p-3 text-sm text-red-300">
          {error}
        </p>
      )}

      {!loading && info && (
        <dl className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <dt className="text-xs uppercase tracking-wider text-slate-500">
              Architecture
            </dt>
            <dd className="mt-1 font-semibold">{info.model_name}</dd>
          </div>

          <div>
            <dt className="text-xs uppercase tracking-wider text-slate-500">
              Input size
            </dt>
            <dd className="mt-1 font-semibold">
              {info.input_size} · {info.input_format}
            </dd>
          </div>

          <div>
            <dt className="text-xs uppercase tracking-wider text-slate-500">
              Classes
            </dt>
            <dd className="mt-1 font-semibold">{info.classes.join(" / ")}</dd>
          </div>

          <div>
            <dt className="text-xs uppercase tracking-wider text-slate-500">
              Threshold
            </dt>
            <dd className="mt-1 font-semibold">{info.prediction_threshold}</dd>
          </div>

          <div>
            <dt className="text-xs uppercase tracking-wider text-slate-500">
              Supported formats
            </dt>
            <dd className="mt-1 font-semibold uppercase">
              {info.supported_formats
                .map((format) => format.replace(".", ""))
                .join(" · ")}
            </dd>
          </div>

          <div>
            <dt className="text-xs uppercase tracking-wider text-slate-500">
              Maximum upload
            </dt>
            <dd className="mt-1 font-semibold">
              {info.maximum_upload_size_mb} MB
            </dd>
          </div>
        </dl>
      )}
    </div>
  );
}
