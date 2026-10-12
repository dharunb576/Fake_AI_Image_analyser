library(readr)
library(dplyr, warn.conflicts = FALSE)
library(randomForest)

# ============================================================
# Configuration
# ============================================================

features_file <- "data/cnn_dataset.csv"
model_file <- "models/cnn_random_forest.rds"

# ============================================================
# Read features
# ============================================================

if (!file.exists(features_file)) {
  stop("Dataset file ", features_file, " not found. Run 03_extract_cnn_features.R first.")
}

features <- read_csv(
  features_file,
  show_col_types = FALSE
)

# ============================================================
# Identify TRAIN / TEST images
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
# Select ONLY test images
# ============================================================

test_data <- features %>%
  filter(dataset_type == "test")

cat("Test images:", nrow(test_data), "\n")

# ============================================================
# Load trained model
# ============================================================

if (!file.exists(model_file)) {
  stop("Model file ", model_file, " not found. Run 04_train_cnn_random_forest.R first.")
}

model <- readRDS(model_file)
cat("Trained CNN Random Forest model loaded successfully!\n")

# ============================================================
# Prepare test data & Predict
# ============================================================

actual_labels <- as.factor(test_data$label)

test_features <- test_data %>%
  select(-image, -label, -dataset_type)

cat("\nTesting CNN model...\n")

predictions <- predict(model, newdata = test_features)

results <- data.frame(
  image = test_data$image,
  actual = actual_labels,
  predicted = predictions
)

accuracy <- mean(results$actual == results$predicted)

cat("\nTesting completed!\n")
cat("Correct predictions:", sum(results$actual == results$predicted), "\n")
cat("Incorrect predictions:", sum(results$actual != results$predicted), "\n")
cat("Accuracy:", round(accuracy * 100, 2), "%\n")

cat("\nConfusion Matrix:\n")
print(table(Actual = results$actual, Predicted = results$predicted))

dir.create("results", showWarnings = FALSE)
write_csv(results, "results/cnn_test_predictions.csv")
cat("\nTest results saved to: results/cnn_test_predictions.csv\n")
