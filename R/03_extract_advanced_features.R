library(imager)
library(readr)
library(dplyr)

# ============================================================
# Configuration
# ============================================================

real_dir <- "data/real"
fake_dir <- "data/fake"

output_file <- "data/advanced_features.csv"

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
# Feature extraction function
# ============================================================

extract_advanced_features <- function(image_path) {

  img <- load.image(image_path)

  # ----------------------------------------------------------
  # Resize
  # ----------------------------------------------------------

  img <- resize(
    img,
    size_x = image_width,
    size_y = image_height
  )

  # ----------------------------------------------------------
  # Convert to grayscale
  # ----------------------------------------------------------

  gray <- grayscale(img)

  gray_pixels <- as.numeric(gray)

  # ----------------------------------------------------------
  # Basic statistical features
  # ----------------------------------------------------------

  brightness <- mean(gray_pixels)

  contrast <- sd(gray_pixels)

  min_pixel <- min(gray_pixels)

  max_pixel <- max(gray_pixels)

  median_pixel <- median(gray_pixels)

  # ----------------------------------------------------------
  # Edge features
  # ----------------------------------------------------------

  dx <- imgradient(gray, "x")
  dy <- imgradient(gray, "y")

  edge_strength <- sqrt(dx^2 + dy^2)

  edge_pixels <- as.numeric(edge_strength)

  mean_edge <- mean(edge_pixels)

  sd_edge <- sd(edge_pixels)

  max_edge <- max(edge_pixels)

  # ----------------------------------------------------------
  # Texture-related features
  # ----------------------------------------------------------

  pixel_variance <- var(gray_pixels)

  pixel_range <- max_pixel - min_pixel

  # ----------------------------------------------------------
  # Color features
  # ----------------------------------------------------------

  red_channel <- R(img)
  green_channel <- G(img)
  blue_channel <- B(img)

  red_pixels <- as.numeric(red_channel)
  green_pixels <- as.numeric(green_channel)
  blue_pixels <- as.numeric(blue_channel)

  mean_red <- mean(red_pixels)
  mean_green <- mean(green_pixels)
  mean_blue <- mean(blue_pixels)

  sd_red <- sd(red_pixels)
  sd_green <- sd(green_pixels)
  sd_blue <- sd(blue_pixels)

  # ----------------------------------------------------------
  # Combine all features
  # ----------------------------------------------------------

  features <- c(

    # Grayscale pixel values
    gray_pixels,

    # Brightness / contrast
    brightness,
    contrast,
    min_pixel,
    max_pixel,
    median_pixel,
    pixel_variance,
    pixel_range,

    # Edge features
    mean_edge,
    sd_edge,
    max_edge,

    # Color features
    mean_red,
    mean_green,
    mean_blue,
    sd_red,
    sd_green,
    sd_blue
  )

  return(features)
}

# ============================================================
# Extract REAL features
# ============================================================

cat("\nExtracting REAL advanced features...\n")

real_features <- lapply(
  real_images,
  extract_advanced_features
)

real_data <- as.data.frame(
  do.call(rbind, real_features)
)

real_data$image <- basename(real_images)

real_data$label <- "real"

# ============================================================
# Extract FAKE features
# ============================================================

cat("Extracting FAKE advanced features...\n")

fake_features <- lapply(
  fake_images,
  extract_advanced_features
)

fake_data <- as.data.frame(
  do.call(rbind, fake_features)
)

fake_data$image <- basename(fake_images)

fake_data$label <- "fake"

# ============================================================
# Combine REAL + FAKE
# ============================================================

features <- bind_rows(
  real_data,
  fake_data
)

# ============================================================
# Move image and label to beginning
# ============================================================

features <- features %>%
  select(
    image,
    label,
    everything()
  )

# ============================================================
# Save features
# ============================================================

write_csv(
  features,
  output_file
)

cat("\nAdvanced feature extraction completed!\n")
cat("Features saved to:", output_file, "\n")
cat("Total images:", nrow(features), "\n")
cat("Total features per image:", ncol(features) - 2, "\n")