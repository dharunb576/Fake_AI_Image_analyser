# AI Fake Photo Detector

An R-based machine learning project that classifies images as real photographs or AI-generated/fake images using image feature extraction and Random Forest classification.

## Features

- Image dataset preparation and preprocessing
- Image feature extraction for machine learning
- Random Forest model training and testing
- Model evaluation using accuracy and a confusion matrix
- Comparison of classification results

## Technology Stack

- **Language:** R
- **Machine Learning:** Random Forest, Decision Tree
- **Data Processing:** readr, dplyr
- **Image Processing:** Image feature extraction
- **Dataset:** CIFAKE

## Project Structure

- `data/` – Dataset and image files
- `R/` – R scripts for preprocessing, feature extraction, training, and testing
- `models/` – Saved machine learning models
- `results/` – Model evaluation results
- `plots/` – Generated visualizations
- `app/` – Application files

## Model Performance

The advanced Random Forest model achieved **79.9% accuracy** in the recorded test run. Results may vary depending on the dataset split and model configuration.

## Purpose

This project explores how machine learning and image analysis can help distinguish real photographs from AI-generated images.

## Disclaimer

This project is intended for educational and experimental purposes. Predictions are not guaranteed to be accurate for every image.