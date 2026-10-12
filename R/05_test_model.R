library(readr)
library(dplyr, warn.conflicts = FALSE)

# ============================================================
# Configuration
# ============================================================

features_file <- "data/features.csv"
model_file <- "models/fake_photo_model.rds"

# ============================================================
# Read features
# ============================================================

features <- read_csv(
  features_file,
  show_col_types = FALSE
)

cat("Total images:", nrow(features), "\n")

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

model <- readRDS(model_file)

cat("\nTrained model loaded successfully!\n")

# ============================================================
# Prepare test data
# ============================================================

actual_labels <- as.factor(test_data$label)

test_features <- test_data %>%
  select(-image, -label, -dataset_type)

# ============================================================
# Make predictions
# ============================================================

cat("\nTesting model...\n")

predictions <- predict(
  model,
  newdata = test_features,
  type = "class"
)

# ============================================================
# Compare predictions with actual labels
# ============================================================

results <- data.frame(
  image = test_data$image,
  actual = actual_labels,
  predicted = predictions
)

# ============================================================
# Calculate accuracy
# ============================================================

accuracy <- mean(
  results$actual == results$predicted
)

cat("\nTesting completed!\n")
cat("Correct predictions:",
    sum(results$actual == results$predicted),
    "\n")

cat("Incorrect predictions:",
    sum(results$actual != results$predicted),
    "\n")

cat("Accuracy:",
    round(accuracy * 100, 2),
    "%\n")

# ============================================================
# Show confusion matrix
# ============================================================

cat("\nConfusion Matrix:\n")

print(
  table(
    Actual = results$actual,
    Predicted = results$predicted
  )
)

# ============================================================
# Save test results
# ============================================================

write_csv(
  results,
  "results/test_predictions.csv"
)

cat("\nTest results saved to: results/test_predictions.csv\n")