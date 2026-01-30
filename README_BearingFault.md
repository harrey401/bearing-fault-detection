# Bearing Fault Detection for Predictive Maintenance

A machine learning approach to early bearing fault detection using three-class classification. Instead of the traditional binary healthy/faulty approach, this system identifies an intermediate "degrading" state to provide earlier warning of impending failures.

## Overview

Traditional bearing monitoring systems detect faults at 80-90% of bearing life, leaving only days to schedule repairs. This project explores whether detecting degradation earlier (around 60% of bearing life) is feasible, even at the cost of some overall accuracy.

## Dataset

NASA IMS (Intelligent Maintenance Systems) bearing dataset:
- Run-to-failure vibration data from bearing test rigs
- 4 bearings per shaft, 2000 RPM, 6000 lbs radial load
- 20 kHz sampling rate, 20480 points per snapshot
- Total of 7588 vibration snapshots across three test sets

## Approach

### Three-Class Labeling

Based on percentage of bearing life:
- Healthy: 0-60%
- Degrading: 60-90%
- Critical: 90-100%

### Feature Extraction

Nine features computed from each vibration snapshot:

Time-domain:
- RMS (root mean square)
- Kurtosis
- Skewness
- Crest factor
- Peak-to-peak amplitude
- Spectral flatness

Frequency-domain:
- Band energy 500-1500 Hz
- Band energy 1500-3000 Hz
- Band energy 3000-6000 Hz

### Classification

Multi-class SVM with RBF kernel using one-vs-all coding. Class weights applied to handle imbalance and emphasize detection of critical failures.

## Results

- Overall accuracy: 74.9%
- Degrading recall: 61.0%
- Critical failure detection: 95.7%
- False negative rate for critical failures: 4.3%

The system successfully detects most degrading bearings at 60% of life, providing 2-3 weeks of advance warning compared to 2-3 days for binary systems.

## Files

- `feature_extraction.m`: Extracts time and frequency domain features from raw vibration data
- `train_classifier.m`: Trains the SVM model with class weighting
- `evaluate_model.m`: Generates confusion matrix and performance metrics

## Requirements

MATLAB with Statistics and Machine Learning Toolbox

## Usage

1. Download the NASA IMS dataset
2. Run feature extraction on the raw vibration files
3. Train the classifier on the extracted features
4. Evaluate on held-out test set

## Author

Harreynish Gowtham Sarav
San Jose State University
November 2025
