dataset <- data.frame
  (
  image = c
  (
    "photo1.jpg",
    "photo2.jpg",
    "photo3.jpg",
    "photo4.jpg"
  ),
  label = c
  (
    "real",
    "fake",
    "real",
    "fake"
  )
)

write.csv(
  dataset,
  "data/dataset.csv",
  row.names = FALSE
)