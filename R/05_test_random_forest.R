library(readr)
library(dplyr, warn.conflicts = FALSE)
library(randomForest)

# ============================================================
# Configuration
# ============================================================

features_file <- "data/features.csv"
model_file <- "models/fake_photo_random_forest.rds"
results_file <- "results/random_forest_predictions.csv"

# ============================================================
# Read extracted features
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
# Load trained Random Forest model
# ============================================================

model <- readRDS(model_file)

cat("\nRandom Forest model loaded successfully!\n")

# ============================================================
# Prepare test data
# ============================================================

actual_labels <- as.factor(test_data$label)

test_features <- test_data %>%
  select(-image, -label, -dataset_type)

# ============================================================
# Make predictions
# ============================================================

cat("\nTesting Random Forest model...\n")

predictions <- predict(
  model,
  newdata = test_features,
  type = "class"
)

# ============================================================
# Calculate accuracy
# ============================================================

correct <- sum(actual_labels == predictions)
incorrect <- sum(actual_labels != predictions)

accuracy <- correct / length(actual_labels)

cat("\nTesting completed!\n")

cat("Correct predictions:", correct, "\n")
cat("Incorrect predictions:", incorrect, "\n")
cat("Accuracy:", round(accuracy * 100, 2), "%\n")

# ============================================================
# Confusion Matrix
# ============================================================

cat("\nConfusion Matrix:\n")

confusion_matrix <- table(
  Actual = actual_labels,
  Predicted = predictions
)

print(confusion_matrix)

# ============================================================
# Save prediction results
# ============================================================

dir.create(
  "results",
  showWarnings = FALSE
)

results <- data.frame(
  image = test_data$image,
  actual = actual_labels,
  predicted = predictions
)

write_csv(
  results,
  results_file
)

cat("\nResults saved to:", results_file, "\n")