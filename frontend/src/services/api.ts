import axios from "axios";

import type { DetectionResponse, ModelInfo } from "../types/detection";

const API_BASE_URL =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60_000,
});

function extractErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    if (error.code === "ERR_NETWORK") {
      return `Cannot reach the backend at ${API_BASE_URL}. Make sure FastAPI is running.`;
    }

    const detail = error.response?.data?.detail;

    if (typeof detail === "string") {
      return detail;
    }

    if (Array.isArray(detail)) {
      return detail
        .map((item) =>
          typeof item?.msg === "string" ? item.msg : "Invalid request."
        )
        .join(" ");
    }

    if (error.response?.status) {
      return `The server responded with status ${error.response.status}.`;
    }

    return error.message;
  }

  if (error instanceof Error) {
    return error.message;
  }

  return "An unexpected error occurred.";
}

export async function detectImage(file: File): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const { data } = await api.post<DetectionResponse>(
      "/api/predict",
      formData
    );

    return data;
  } catch (error) {
    throw new Error(extractErrorMessage(error), { cause: error });
  }
}

export async function getModelInfo(): Promise<ModelInfo> {
  try {
    const { data } = await api.get<ModelInfo>("/model-info");
    return data;
  } catch (error) {
    throw new Error(extractErrorMessage(error), { cause: error });
  }
}
