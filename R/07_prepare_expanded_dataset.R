
# ============================================================
# 07_prepare_expanded_dataset.R
# AI Fake Photo Detector
#
# Creates a new balanced 20,000-image dataset.
# Original CIFAKE data and existing models remain untouched.
#
# Train      : 8,000 REAL + 8,000 FAKE
# Validation : 1,000 REAL + 1,000 FAKE
# Test       : 1,000 REAL + 1,000 FAKE
# ============================================================

# 1. Configuration -------------------------------------------

set.seed(42)

project_dir <- getwd()

source_dir <- file.path(
  project_dir, "data", "raw", "cifake"
)

output_dir <- file.path(
  project_dir, "data", "expanded_v1"
)

train_per_class <- 8000
val_per_class   <- 1000
test_per_class  <- 1000

# 2. Check source folders -------------------------------------

source_folders <- c(
  file.path(source_dir, "train", "REAL"),
  file.path(source_dir, "train", "FAKE"),
  file.path(source_dir, "test", "REAL"),
  file.path(source_dir, "test", "FAKE")
)

if (!all(dir.exists(source_folders))) {
  stop("CIFAKE source folders are missing. Check data/raw/cifake/")
}

if (dir.exists(output_dir)) {
  stop(
    "Output folder already exists: ", output_dir,
    "\nRename or back up that folder before running this script."
  )
}

# 3. Get image files -----------------------------------------

get_images <- function(folder) {

  files <- list.files(
    folder,
    pattern = "\\.(jpg|jpeg|png|bmp|webp)$",
    full.names = TRUE,
    recursive = TRUE,
    ignore.case = TRUE
  )

  files[file.info(files)$isdir %in% FALSE]
}

train_real <- get_images(source_folders[1])
train_fake <- get_images(source_folders[2])
test_real  <- get_images(source_folders[3])
test_fake  <- get_images(source_folders[4])

cat("\n========== ORIGINAL DATASET COUNTS ==========\n")

cat("Training REAL :", length(train_real), "\n")
cat("Training FAKE :", length(train_fake), "\n")
cat("Testing REAL  :", length(test_real), "\n")
cat("Testing FAKE  :", length(test_fake), "\n")

# 4. Check minimum image counts ------------------------------

if (length(train_real) < train_per_class + val_per_class) {
  stop("Not enough training REAL images.")
}

if (length(train_fake) < train_per_class + val_per_class) {
  stop("Not enough training FAKE images.")
}

if (length(test_real) < test_per_class) {
  stop("Not enough testing REAL images.")
}

if (length(test_fake) < test_per_class) {
  stop("Not enough testing FAKE images.")
}

# 5. Calculate hashes and remove exact duplicates ------------

cat("\nCalculating image hashes...\n")

get_hashes <- function(files) {

  hashes <- as.character(tools::md5sum(files))

  data.frame(
    path = files,
    hash = hashes,
    stringsAsFactors = FALSE
  )
}

train_real_df <- get_hashes(train_real)
train_fake_df <- get_hashes(train_fake)
test_real_df  <- get_hashes(test_real)
test_fake_df  <- get_hashes(test_fake)

# Remove duplicates within each source group.
train_real_df <- train_real_df[!duplicated(train_real_df$hash), ]
train_fake_df <- train_fake_df[!duplicated(train_fake_df$hash), ]
test_real_df  <- test_real_df[!duplicated(test_real_df$hash), ]
test_fake_df  <- test_fake_df[!duplicated(test_fake_df$hash), ]

# Remove cross-label duplicates.
real_hashes <- unique(c(train_real_df$hash, test_real_df$hash))
fake_hashes <- unique(c(train_fake_df$hash, test_fake_df$hash))

conflicting_hashes <- intersect(real_hashes, fake_hashes)

train_real_df <- train_real_df[
  !train_real_df$hash %in% conflicting_hashes, ]
test_real_df <- test_real_df[
  !test_real_df$hash %in% conflicting_hashes, ]

train_fake_df <- train_fake_df[
  !train_fake_df$hash %in% conflicting_hashes, ]
test_fake_df <- test_fake_df[
  !test_fake_df$hash %in% conflicting_hashes, ]

# Keep test images separate from training/validation.
test_hashes <- unique(c(test_real_df$hash, test_fake_df$hash))

train_real_df <- train_real_df[
  !train_real_df$hash %in% test_hashes, ]
train_fake_df <- train_fake_df[
  !train_fake_df$hash %in% test_hashes, ]

cat("Cross-label duplicate hashes removed:", 
    length(conflicting_hashes), "\n")

# 6. Verify sufficient unique images -------------------------

if (nrow(train_real_df) < train_per_class + val_per_class) {
  stop("Not enough unique training REAL images after deduplication.")
}

if (nrow(train_fake_df) < train_per_class + val_per_class) {
  stop("Not enough unique training FAKE images after deduplication.")
}

if (nrow(test_real_df) < test_per_class) {
  stop("Not enough unique testing REAL images after deduplication.")
}

if (nrow(test_fake_df) < test_per_class) {
  stop("Not enough unique testing FAKE images after deduplication.")
}

# 7. Select images --------------------------------------------

select_images <- function(df, count) {

  selected <- sample(
    seq_len(nrow(df)),
    size = count,
    replace = FALSE
  )

  df[selected, , drop = FALSE]
}

# REAL selections
real_train <- select_images(train_real_df, train_per_class)
real_val   <- select_images(
  train_real_df[
    !train_real_df$hash %in% real_train$hash, ],
  val_per_class
)
real_test  <- select_images(test_real_df, test_per_class)

# FAKE selections
fake_train <- select_images(train_fake_df, train_per_class)
fake_val   <- select_images(
  train_fake_df[
    !train_fake_df$hash %in% fake_train$hash, ],
  val_per_class
)
fake_test  <- select_images(test_fake_df, test_per_class)

# 8. Create output directories -------------------------------

splits <- c("train", "val", "test")
labels <- c("REAL", "FAKE")

for (split in splits) {
  for (label in labels) {

    dir.create(
      file.path(output_dir, split, label),
      recursive = TRUE,
      showWarnings = FALSE
    )

  }
}

# 9. Copy images and generate metadata ------------------------

metadata <- data.frame(
  filename = character(),
  source_path = character(),
  label = character(),
  split = character(),
  md5 = character(),
  stringsAsFactors = FALSE
)

copy_group <- function(df, label, split) {

  destination <- file.path(
    output_dir, split, label
  )

  for (i in seq_len(nrow(df))) {

    source_file <- df$path[i]
    hash_value <- df$hash[i]

    extension <- tools::file_ext(source_file)

    filename <- paste0(
      substr(hash_value, 1, 16),
      ".", extension
    )

    destination_file <- file.path(
      destination, filename
    )

    copied <- file.copy(
      source_file,
      destination_file,
      overwrite = FALSE
    )

    if (!copied) {
      stop("Failed to copy: ", source_file)
    }

    metadata[nrow(metadata) + 1, ] <<- list(
      filename = filename,
      source_path = source_file,
      label = label,
      split = split,
      md5 = hash_value
    )
  }
}

cat("\nCopying images...\n")

copy_group(real_train, "REAL", "train")
copy_group(fake_train, "FAKE", "train")

copy_group(real_val, "REAL", "val")
copy_group(fake_val, "FAKE", "val")

copy_group(real_test, "REAL", "test")
copy_group(fake_test, "FAKE", "test")

# 10. Save metadata -------------------------------------------

write.csv(
  metadata,
  file.path(output_dir, "metadata.csv"),
  row.names = FALSE
)

# 11. Final dataset report ------------------------------------

cat("\n========== EXPANDED DATASET SUMMARY ==========\n")

for (split in splits) {

  real_count <- sum(
    metadata$split == split &
    metadata$label == "REAL"
  )

  fake_count <- sum(
    metadata$split == split &
    metadata$label == "FAKE"
  )

  cat(
    toupper(split),
    ": REAL =", real_count,
    "| FAKE =", fake_count,
    "| TOTAL =", real_count + fake_count,
    "\n"
  )
}

cat("\nTotal images:", nrow(metadata), "\n")

cat("\nDataset created successfully!\n")
cat("Location:", output_dir, "\n")
cat("Metadata: data/expanded_v1/metadata.csv\n")

cat("\nOriginal CIFAKE dataset was not modified.\n")
cat("Existing trained models were not modified.\n")
