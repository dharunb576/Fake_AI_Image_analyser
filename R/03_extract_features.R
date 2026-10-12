library(imager)
library(readr)
library(dplyr)

# ============================================================
# Configuration
# ============================================================

real_dir <- "data/real"
fake_dir <- "data/fake"

output_file <- "data/features.csv"

image_width <- 32
image_height <- 32

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

cat("REAL images:", length(real_images), "\n")
cat("FAKE images:", length(fake_images), "\n")

# ============================================================
# Function to extract features from one image
# ============================================================

extract_features <- function(image_path) {

  img <- load.image(image_path)

  # Convert to grayscale
  img <- grayscale(img)

  # Resize image
  img <- resize(
    img,
    size_x = image_width,
    size_y = image_height
  )

  # Convert image pixels into numeric vector
  pixels <- as.numeric(img)

  return(pixels)
}

# ============================================================
# Extract REAL features
# ============================================================

cat("\nExtracting REAL image features...\n")

real_features <- lapply(real_images, extract_features)

real_data <- as.data.frame(
  do.call(rbind, real_features)
)

# Add image filename
real_data$image <- basename(real_images)

# Add label
real_data$label <- "real"

# ============================================================
# Extract FAKE features
# ============================================================

cat("Extracting FAKE image features...\n")

fake_features <- lapply(fake_images, extract_features)

fake_data <- as.data.frame(
  do.call(rbind, fake_features)
)

# Add image filename
fake_data$image <- basename(fake_images)

# Add label
fake_data$label <- "fake"

# ============================================================
# Combine REAL + FAKE
# ============================================================

features <- bind_rows(
  real_data,
  fake_data
)

# ============================================================
# Move image and label columns to the beginning
# ============================================================

features <- features %>%
  select(image, label, everything())

# ============================================================
# Save extracted features
# ============================================================

write_csv(
  features,
  output_file
)

cat("\nFeature extraction completed!\n")
cat("Features saved to:", output_file, "\n")
cat("Total images:", nrow(features), "\n")