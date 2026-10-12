library(readr)
library(dplyr)
library(randomForest)

# ============================================================
# Configuration
# ============================================================

features_file <- "data/advanced_features.csv"
model_file <- "models/advanced_random_forest.rds"

# ============================================================
# Read advanced features
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
# Select ONLY training images
# ============================================================

train_data <- features %>%
  filter(dataset_type == "train") %>%
  select(-image, -dataset_type)

cat("Training images:", nrow(train_data), "\n")

# ============================================================
# Convert label to factor
# ============================================================

train_data$label <- as.factor(train_data$label)

# ============================================================
# Train Random Forest
# ============================================================

cat("\nTraining Advanced Random Forest model...\n")

set.seed(123)

model <- randomForest(
  label ~ .,
  data = train_data,
  ntree = 300
)

# ============================================================
# Save model
# ============================================================

dir.create(
  "models",
  showWarnings = FALSE
)

saveRDS(
  model,
  model_file
)

cat("\nAdvanced Random Forest training completed!\n")
cat("Model saved to:", model_file, "\n")