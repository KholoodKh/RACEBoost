# RACEBoost

### Exploring how training-instance importance and presentation order affect rule-based classification.

This project investigates how **training-instance weighting and ordering strategies influence the performance of RACER (Rule Aggregation Classifier)**. It explores whether presenting training samples in different sequences, guided by instance difficulty, usefulness, and other data characteristics, can improve classification accuracy.

## 🔍 Key Features

- **Instance Weighting:** Implements multiple weighting and scoring methods, including harmfulness, usefulness, LSR, LSC, N2, F1, Mahalanobis distance, Gaussian-based measures, kNN-based methods, and rule-based measures.
- **Sample Ordering:** Experiments with ascending and descending hardness, partition-based ordering, three-level difficulty grouping, and probabilistic sampling strategies.
- **Curriculum Learning:** Investigates how the order in which training instances are presented affects the learning process and the resulting classification rules.
- **Systematic Evaluation:** Tests combinations of weighting and ordering strategies using cross-validation, keeping test data untouched by training-time ordering.
- **Performance Analysis:** Compares classification accuracy across experimental configurations and summarizes results across folds.

## 🎯 Research Objective

To understand how training-instance characteristics and presentation order affect RACER's rule generation and classification performance, and to identify combinations of weighting and ordering strategies that can improve its effectiveness.

## 🛠️ Technologies

- Python
- NumPy
- Pandas
- scikit-learn
- Joblib

## 📊 Experiments

The project evaluates different weighting–ordering combinations across multiple datasets and cross-validation folds. Results are saved for further analysis, enabling systematic comparison of strategies and their impact on classification performance.
