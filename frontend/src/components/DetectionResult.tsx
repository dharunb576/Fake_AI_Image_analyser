import {
  AlertCircle,
  CheckCircle2,
  Clock3,
  Image as ImageIcon,
  LoaderCircle,
  RefreshCw,
  ScanSearch,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { DetectionResponse } from "../types/detection";

interface DetectionResultProps {
  result: DetectionResponse | null;
  loading: boolean;
  onReset: () => void;
}

export function DetectionResult({
  result,
  loading,
  onReset,
}: DetectionResultProps) {
  if (loading) {
    return (
      <div className="rounded-2xl border border-white/10 bg-[#111827] p-5 sm:p-6">
        <h2 className="mb-5 flex items-center gap-3 text-lg font-semibold">
          <ScanSearch className="text-cyan-400" />
          Detection Results
        </h2>

        <div className="flex min-h-[370px] flex-col items-center justify-center rounded-xl bg-[#0b1220]">
          <LoaderCircle size={48} className="animate-spin text-violet-400" />
          <p className="mt-5 font-semibold">Analyzing Your Image...</p>
          <p className="mt-2 px-6 text-center text-sm text-slate-400">
            EfficientNetB0 is processing your image.
          </p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="rounded-2xl border border-white/10 bg-[#111827] p-5 sm:p-6">
        <h2 className="mb-5 flex items-center gap-3 text-lg font-semibold">
          <ScanSearch className="text-cyan-400" />
          Detection Results
        </h2>

        <div className="flex min-h-[370px] flex-col items-center justify-center rounded-xl border border-dashed border-slate-700 bg-[#0b1220] px-6 text-center">
          <ImageIcon size={45} className="text-slate-500" />
          <h3 className="mt-5 font-semibold">Waiting for Analysis</h3>
          <p className="mt-2 max-w-xs text-sm text-slate-400">
            Your image prediction will appear here.
          </p>
        </div>
      </div>
    );
  }

  const isFake = result.prediction === "FAKE";

  const chartData = [
    { name: "REAL", probability: result.real_probability },
    { name: "FAKE", probability: result.fake_probability },
  ];

  return (
    <div className="rounded-2xl border border-white/10 bg-[#111827] p-5 sm:p-6">
      <h2 className="mb-5 flex items-center gap-3 text-lg font-semibold">
        <ScanSearch className="text-cyan-400" />
        Detection Results
      </h2>

      <div className="rounded-xl bg-[#0b1220] p-5">
        <div className="text-center">
          <div
            className={`mx-auto flex h-16 w-16 items-center justify-center rounded-2xl ${
              isFake
                ? "bg-red-500/15 text-red-400"
                : "bg-emerald-500/15 text-emerald-400"
            }`}
          >
            {isFake ? (
              <AlertCircle size={35} />
            ) : (
              <CheckCircle2 size={35} />
            )}
          </div>

          <p className="mt-4 text-sm text-slate-400">Prediction</p>

          <p
            className={`mt-2 text-4xl font-extrabold ${
              isFake ? "text-red-400" : "text-emerald-400"
            }`}
          >
            {result.prediction}
          </p>

          <p className="mt-2 text-sm text-slate-400">
            Model confidence: {result.confidence.toFixed(2)}%
          </p>
        </div>

        <div className="mt-7">
          <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
            Probability distribution
          </p>

          <div className="h-[210px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                layout="vertical"
                margin={{ top: 4, right: 24, bottom: 4, left: 0 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(148, 163, 184, 0.12)"
                  horizontal={false}
                />
                <XAxis
                  type="number"
                  domain={[0, 100]}
                  tick={{ fill: "#94a3b8", fontSize: 12 }}
                  tickLine={false}
                  axisLine={false}
                  unit="%"
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={52}
                  tick={{ fill: "#cbd5e1", fontSize: 12, fontWeight: 600 }}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  cursor={{ fill: "rgba(148, 163, 184, 0.06)" }}
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    border: "1px solid rgba(148, 163, 184, 0.2)",
                    borderRadius: "10px",
                    color: "#e2e8f0",
                  }}
                  labelStyle={{ color: "#94a3b8" }}
                  formatter={(value) => [`${Number(value).toFixed(2)}%`, "Probability"]}
                />
                <Bar dataKey="probability" radius={[0, 6, 6, 0]} barSize={30}>
                  {chartData.map((entry) => (
                    <Cell
                      key={entry.name}
                      fill={entry.name === "FAKE" ? "#ef4444" : "#10b981"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="mt-5 space-y-5">
          <div>
            <div className="mb-2 flex justify-between text-sm">
              <span className="text-slate-400">REAL probability</span>
              <span className="text-emerald-400">
                {result.real_probability.toFixed(2)}%
              </span>
            </div>

            <div className="h-2 rounded-full bg-slate-700">
              <div
                className="h-2 rounded-full bg-emerald-500"
                style={{ width: `${result.real_probability}%` }}
              />
            </div>
          </div>

          <div>
            <div className="mb-2 flex justify-between text-sm">
              <span className="text-slate-400">FAKE probability</span>
              <span className="text-red-400">
                {result.fake_probability.toFixed(2)}%
              </span>
            </div>

            <div className="h-2 rounded-full bg-slate-700">
              <div
                className="h-2 rounded-full bg-red-500"
                style={{ width: `${result.fake_probability}%` }}
              />
            </div>
          </div>
        </div>

        <div className="mt-7 grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="rounded-lg bg-white/5 p-3">
            <p className="text-xs text-slate-400">Model</p>
            <p className="mt-2 truncate font-semibold">{result.model}</p>
          </div>

          <div className="rounded-lg bg-white/5 p-3">
            <p className="flex items-center gap-1.5 text-xs text-slate-400">
              <Clock3 size={13} />
              Processing time
            </p>
            <p className="mt-2 font-semibold">{result.processing_time}s</p>
          </div>
        </div>

        <button
          type="button"
          onClick={onReset}
          className="mt-6 flex w-full items-center justify-center gap-2 rounded-lg border border-white/10 py-3 text-sm transition hover:border-violet-400"
        >
          <RefreshCw size={16} />
          Analyze Another Image
        </button>
      </div>
    </div>
  );
}
