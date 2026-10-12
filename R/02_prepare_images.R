
library(imager)
library(readr)
library(dplyr)

# Project directories
data_dir <- "data"
real_dir <- file.path(data_dir, "real")
fake_dir <- file.path(data_dir, "fake")

# Read dataset
dataset <- read_csv(
  file.path(data_dir, "dataset.csv"),
  show_col_types = FALSE
)

print(dataset)

# Check whether each image exists
dataset <- dataset %>%
  mutate(
    path = case_when(
      label == "real" ~ file.path(real_dir, image),
      label == "fake" ~ file.path(fake_dir, image),
      TRUE ~ NA_character_
    ),
    exists = file.exists(path)
  )

//print(dataset)

# Show missing images
missing_images <- dataset %>%
  filter(!exists)

if (nrow(missing_images) > 0) {
  cat("\nMissing images:\n")
  print(missing_images)
} else {
  cat("\nAll images found successfully!\n")
}