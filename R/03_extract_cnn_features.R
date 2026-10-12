library(reticulate)
library(readr)
library(dplyr)

# Force Python environment where TensorFlow/Keras is installed
use_python(
  "~/.virtualenvs/r-tensorflow/bin/python",
  required = TRUE
)

tf <- import("tensorflow")

# ============================================================
# Configuration
# ============================================================

real_dir <- "data/real"
fake_dir <- "data/fake"

output_file <- "data/cnn_dataset.csv"
output_file_alt <- "data/cnn_features.csv"

image_width <- 224
image_height <- 224
batch_size <- 64

# ============================================================
# Get image files
# ============================================================

real_images <- list.files(
  real_dir,
  pattern = "\\.(jpg|jpeg|png|webp)$",
  full.names = TRUE,
  ignore.case = TRUE
)

fake_images <- list.files(
  fake_dir,
  pattern = "\\.(jpg|jpeg|png|webp)$",
  full.names = TRUE,
  ignore.case = TRUE
)

cat("REAL images found:", length(real_images), "\n")
cat("FAKE images found:", length(fake_images), "\n")

# ============================================================
# Load pre-trained MobileNetV2 CNN
# ============================================================

cat("\nLoading pre-trained MobileNetV2 model...\n")

base_model <- tf$keras$applications$MobileNetV2(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = list(as.integer(image_height), as.integer(image_width), 3L),
  pooling = "avg"
)

cat("CNN loaded successfully!\n")

# ============================================================
# Helper function to extract features in batches
# ============================================================

extract_batch_features <- function(image_paths, batch_size = 64) {
  n <- length(image_paths)
  all_features <- list()
  
  if (n == 0) return(data.frame())
  
  num_batches <- ceiling(n / batch_size)
  
  for (b in seq_len(num_batches)) {
    start_idx <- (b - 1) * batch_size + 1
    end_idx <- min(b * batch_size, n)
    batch_paths <- image_paths[start_idx:end_idx]
    
    cat(sprintf("\rProcessing batch %d/%d (images %d-%d)...", b, num_batches, start_idx, end_idx))
    flush.console()
    
    img_tensors <- lapply(batch_paths, function(p) {
      tryCatch({
        img <- tf$keras$utils$load_img(p, target_size = list(as.integer(image_height), as.integer(image_width)))
        tf$keras$utils$img_to_array(img)
      }, error = function(e) {
        warning(paste("Failed to read image:", p))
        NULL
      })
    })
    
    # Filter out failed images if any
    valid_mask <- !sapply(img_tensors, is.null)
    valid_tensors <- img_tensors[valid_mask]
    
    if (length(valid_tensors) > 0) {
      batch_stack <- tf$stack(valid_tensors)
      batch_preprocessed <- tf$keras$applications$mobilenet_v2$preprocess_input(batch_stack)
      batch_pred <- base_model$predict(batch_preprocessed, verbose = 0L)
      
      # Convert predictions to R matrix
      feat_mat <- as.matrix(batch_pred)
      all_features[[b]] <- feat_mat
    }
  }
  cat("\n")
  
  feat_matrix <- do.call(rbind, all_features)
  return(as.data.frame(feat_matrix))
}

# ============================================================
# Extract REAL image features
# ============================================================

cat("\nExtracting CNN features from REAL images...\n")
real_data <- extract_batch_features(real_images, batch_size = batch_size)
if (nrow(real_data) > 0) {
  real_data$image <- basename(real_images[1:nrow(real_data)])
  real_data$label <- "real"
}

# ============================================================
# Extract FAKE image features
# ============================================================

cat("\nExtracting CNN features from FAKE images...\n")
fake_data <- extract_batch_features(fake_images, batch_size = batch_size)
if (nrow(fake_data) > 0) {
  fake_data$image <- basename(fake_images[1:nrow(fake_data)])
  fake_data$label <- "fake"
}

# ============================================================
# Combine & reorder columns
# ============================================================

features <- bind_rows(real_data, fake_data)

features <- features %>%
  select(image, label, everything())

# Standardize feature column names: image, label, feature_1, feature_2...
feature_cols <- names(features)[-(1:2)]
new_cols <- c("image", "label", paste0("feature_", seq_along(feature_cols)))
colnames(features) <- new_cols

# ============================================================
# Save CNN features dataset
# ============================================================

cat("\nSaving CNN dataset to CSV...\n")
write_csv(features, output_file)
write_csv(features, output_file_alt)

cat("CNN feature extraction completed successfully!\n")
cat("Primary dataset saved to:", output_file, "\n")
cat("Secondary features saved to:", output_file_alt, "\n")
cat("Total images processed:", nrow(features), "\n")
cat("Features per image:", ncol(features) - 2, "\n")