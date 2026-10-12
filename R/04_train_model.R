library(readr)
library(dplyr)
library(rpart)
library(rpart.plot)

# ============================================================
# Configuration
# ============================================================

features_file <- "data/features.csv"
model_file <- "models/fake_photo_model.rds"

# ============================================================
# Read extracted features
# ============================================================

features <- read_csv(
  features_file,
  show_col_types = FALSE
)

cat("Total images:", nrow(features), "\n")

# ============================================================
# Identify TRAIN and TEST images
# ============================================================

# The feature file currently contains:
# real_train_1.jpg, real_train_2.jpg, ...
# fake_train_1.jpg, fake_train_2.jpg, ...
# real_test_1.jpg, real_test_2.jpg, ...
# fake_test_1.jpg, fake_test_2.jpg, ...

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

# ============================================================
# Convert label to factor
# ============================================================

train_data$label <- as.factor(train_data$label)

# ============================================================
# Train Decision Tree model
# ============================================================

cat("\nTraining model...\n")

model <- rpart(
  label ~ .,
  data = train_data,
  method = "class"
)

# ============================================================
# Save trained model
# ============================================================

dir.create(
  "models",
  showWarnings = FALSE
)

saveRDS(
  model,
  model_file
)

cat("\nTraining completed successfully!\n")
cat("Model saved to:", model_file, "\n")