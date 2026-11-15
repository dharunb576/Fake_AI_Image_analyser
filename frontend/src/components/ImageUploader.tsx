import { useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";
import {
  AlertCircle,
  Image as ImageIcon,
  LoaderCircle,
  ScanSearch,
  UploadCloud,
  X,
} from "lucide-react";

interface ImageUploaderProps {
  file: File | null;
  preview: string;
  error: string;
  loading: boolean;
  onSelect: (file?: File) => void;
  onAnalyze: () => void;
  onReset: () => void;
}

export function ImageUploader({
  file,
  preview,
  error,
  loading,
  onSelect,
  onAnalyze,
  onReset,
}: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    if (!file && inputRef.current) {
      inputRef.current.value = "";
    }
  }, [file]);

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    onSelect(event.dataTransfer.files?.[0]);
  };

  return (
    <div className="rounded-2xl border border-white/10 bg-[#111827] p-5 sm:p-6">
      <h2 className="mb-5 flex items-center gap-3 text-lg font-semibold">
        <UploadCloud className="text-violet-400" />
        Upload Image
      </h2>

      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={(event) => {
          onSelect(event.target.files?.[0]);
          event.target.value = "";
        }}
      />

      {!file ? (
        <div
          role="button"
          tabIndex={0}
          aria-label="Choose an image to upload"
          onClick={() => inputRef.current?.click()}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              inputRef.current?.click();
            }
          }}
          onDragEnter={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          className={`flex min-h-[300px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition ${
            dragActive
              ? "border-violet-400 bg-violet-500/5"
              : "border-slate-700 bg-[#0b1220] hover:border-violet-500"
          }`}
        >
          <UploadCloud size={45} className="mb-5 text-violet-400" />

          <h3 className="text-lg font-semibold">
            {dragActive ? "Drop your image here" : "Drag and drop your image"}
          </h3>

          <p className="mt-2 text-sm text-slate-400">Or click to browse files</p>

          <span className="mt-6 rounded-lg bg-violet-600 px-6 py-3 text-sm font-semibold">
            Choose Image
          </span>

          <p className="mt-5 text-xs text-slate-500">
            JPG · JPEG · PNG · WEBP · Maximum 10 MB
          </p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-xl border border-white/10 bg-[#0b1220]">
          <div className="flex items-center justify-between p-3">
            <span className="text-sm font-medium">Image Preview</span>

            <button
              type="button"
              onClick={onReset}
              title="Remove image"
              aria-label="Remove image"
              className="rounded p-1 text-slate-400 transition hover:text-red-400"
            >
              <X size={18} />
            </button>
          </div>

          <div className="flex h-[240px] items-center justify-center p-3 sm:h-[260px]">
            {preview ? (
              <img
                src={preview}
                alt="Selected image preview"
                className="max-h-full max-w-full rounded-lg object-contain"
              />
            ) : (
              <ImageIcon size={40} className="text-slate-600" />
            )}
          </div>

          <div className="border-t border-white/10 p-4">
            <p className="truncate text-sm">{file.name}</p>
            <p className="mt-1 text-xs text-slate-400">
              {(file.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        </div>
      )}

      {error && (
        <div
          role="alert"
          className="mt-4 flex gap-2 rounded-lg bg-red-500/10 p-3 text-sm text-red-300"
        >
          <AlertCircle size={18} className="shrink-0" />
          {error}
        </div>
      )}

      <button
        type="button"
        onClick={onAnalyze}
        disabled={!file || loading}
        className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-violet-600 py-4 font-semibold transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {loading ? (
          <>
            <LoaderCircle className="animate-spin" size={20} />
            Analyzing...
          </>
        ) : (
          <>
            <ScanSearch size={20} />
            Detect Image
          </>
        )}
      </button>
    </div>
  );
}
