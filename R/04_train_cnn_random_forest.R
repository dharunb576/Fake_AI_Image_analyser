library(readr)
library(dplyr)
library(randomForest)

# ============================================================
# Configuration
# ============================================================

features_file <- "data/cnn_dataset.csv"
model_file <- "models/cnn_random_forest.rds"

# ============================================================
# Read extracted CNN features
# ============================================================

if (!file.exists(features_file)) {
  stop("Dataset file ", features_file, " not found. Run 03_extract_cnn_features.R first.")
}

features <- read_csv(
  features_file,
  show_col_types = FALSE
)

cat("Total images in CNN dataset:", nrow(features), "\n")

# ============================================================
# Identify TRAIN and TEST images
# ============================================================

features <- features %>%
  mutate(
    dataset_type = case_when(
      grepl("_train_", image) ~ "train",
      grepl("_test_", image) ~ "test",
      TRUE ~ NA_character_
    )
  )

# ============================================================
# Keep ONLY training images
# ============================================================

train_data <- features %>%
  filter(dataset_type == "train") %>%
  select(-dataset_type, -image)

cat("Training images:", nrow(train_data), "\n")

train_data$label <- as.factor(train_data$label)

# ============================================================
# Train Random Forest model on CNN features
# ============================================================

cat("\nTraining Random Forest model on CNN features...\n")

set.seed(123)

model <- randomForest(
  label ~ .,
  data = train_data,
  ntree = 200
)

# ============================================================
# Save trained model
# ============================================================

dir.create("models", showWarnings = FALSE)

saveRDS(model, model_file)

cat("\nCNN Random Forest training completed!\n")
cat("Model saved to:", model_file, "\n")
