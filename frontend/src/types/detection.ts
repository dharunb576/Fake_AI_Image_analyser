export type PredictionLabel = "REAL" | "FAKE";

export interface DetectionResponse {
  success: boolean;
  filename: string;
  prediction: PredictionLabel;
  confidence: number;
  real_probability: number;
  fake_probability: number;
  model: string;
  processing_time: number;
}

export interface ModelInfo {
  model_name: string;
  input_size: string;
  input_format: string;
  classes: PredictionLabel[];
  prediction_threshold: number;
  supported_formats: string[];
  maximum_upload_size_mb: number;
}
