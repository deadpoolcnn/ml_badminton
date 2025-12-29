# Badminton Shot Type Recognition System

A machine learning-based badminton shot recognition system that uses sensor data (accelerometer and gyroscope) to classify four types of shots: Smash, Clear, Drive, and Lift.

## 📋 Project Overview

This project uses machine learning methods to recognize different shot types in badminton. By processing accelerometer and gyroscope sensor data, extracting features, and training classification models, it achieves automatic recognition.

**Course**: CPSC 5616EL Assignment 3 - Group 6

## ✨ Key Features

- **Data Preprocessing**: Timestamp normalization, sensor bias correction, gravity compensation, noise reduction
- **Feature Extraction**: 
  - Time-domain features (mean, standard deviation, peak values, etc.)
  - Frequency-domain features (power spectral density, dominant frequency, etc.)
  - Statistical features (skewness, kurtosis, quartiles, etc.)
- **Data Augmentation**: Physically reasonable augmentation techniques (axis perturbation, amplitude scaling, Gaussian jittering, etc.)
- **Multi-Model Support**: Random Forest, Gradient Boosting, Logistic Regression
- **Complete Evaluation Pipeline**: Confusion matrix, classification report, F1 scores

## 🗂️ Project Structure

```
ml_badminton/
├── assignment3.py                 # Main training script (Chinese comments)
├── CPSC_5616EL_A3_G6.py          # Main training script (English comments)
├── assignment3_preprocess.py      # Data preprocessing module (Chinese)
├── data_preprogress.py           # Data preprocessing module (English)
├── assignment3.ipynb             # Jupyter Notebook for analysis
├── dataset/                       # Dataset directory
│   ├── clear_x30.csv             # Clear shot raw data
│   ├── smash_x30.csv             # Smash shot raw data
│   ├── drive_x30.csv             # Drive shot raw data
│   ├── lift_x30.csv              # Lift shot raw data
│   ├── *_preprocessed.csv        # Preprocessed data
│   ├── *_features.csv            # Extracted feature data
│   └── augmented_*.csv/npz       # Augmented datasets
└── new_dataset/                   # New dataset directory
    └── *_10_kyle.csv             # Kyle's data samples
```

## 🔧 Requirements

### Python Version
- Python 3.8+

### Main Dependencies
```bash
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd ml_badminton

# Install dependencies
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

## 🚀 Usage

### 1. Data Preprocessing

```python
from assignment3_preprocess import (
    resample_clean_timestamp,
    remove_sensor_bias,
    gravity_compensation,
    noise_reduction
)

# Load raw data
df = pd.read_csv('dataset/clear_x30.csv')

# Preprocessing steps
df = resample_clean_timestamp(df, target_hz=10)
df, sensor_cols = remove_sensor_bias(df)
df = gravity_compensation(df, accel_cols, sampling_rate=10)
df = noise_reduction(df, sensor_cols)
```

### 2. Feature Extraction

Feature extraction includes:
- **Time-domain features**: Mean, variance, peak values, energy
- **Frequency-domain features**: FFT, power spectral density, dominant frequency
- **Statistical features**: Skewness, kurtosis, quartiles
- **Cross features**: Correlation between accelerometer and gyroscope

### 3. Model Training

```python
from assignment3 import load_data, train_random_forest

# Load data
X, y, feature_columns = load_data('dataset/A3_features.csv')

# Split train and test sets
X_train, X_test, y_train, y_test = split_train_test_data(X, y)

# Train Random Forest model
rf_model, rf_scores = train_random_forest(X_train, y_train, X_test, y_test)
```

### 4. Data Augmentation

```python
# Apply data augmentation
X_train_aug, y_train_aug = data_augumentation(
    X_train, 
    y_train, 
    feature_columns,
    augumentation_factor=3
)
```

## 📊 Shot Type Descriptions

This project recognizes the following four badminton shot types:

1. **Smash**: High-speed downward attacking shot
2. **Clear**: High-arc defensive or transitional shot
3. **Drive**: Fast, flat attacking shot
4. **Lift**: Defensive shot lifting the shuttlecock from the net

## 🎯 Model Performance

The project supports multiple machine learning models:

- **Random Forest**: Ensemble learning method with voting from multiple decision trees
- **Histogram Gradient Boosting**: Efficient gradient boosting algorithm
- **Logistic Regression**: Baseline linear model

Training process includes:
- Grid search for hyperparameter optimization
- Cross-validation evaluation
- Detailed performance metrics output

## 📈 Evaluation Metrics

- **Accuracy**
- **F1 Score** - Macro-average and weighted-average
- **Confusion Matrix**
- **Classification Report** - Including precision, recall, and F1 score

## 🔬 Data Augmentation Techniques

To improve model generalization, the following physically reasonable augmentation methods are implemented:

1. **Axis Perturbation**: Simulates slight sensor misalignment
2. **Amplitude Scaling**: Simulates different shot intensities
3. **Gaussian Jittering**: Adds sensor noise
4. **Time Warping**: Simulates different shot speeds
5. **Feature Smoothing**: Reduces feature noise

## 📝 File Descriptions

- **assignment3.py / CPSC_5616EL_A3_G6.py**: Complete training pipeline including data loading, augmentation, model training, and evaluation
- **assignment3_preprocess.py / data_preprogress.py**: Data preprocessing function library
- **assignment3.ipynb**: Interactive analysis and visualization notebook
- **A3_features*.csv**: Extracted feature datasets

## 🤝 Contributors

CPSC 5616EL - Group 6

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 🔗 Related Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [SciPy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html)

## 💡 Notes

1. Ensure input data format is correct (containing timestamp, accelerometer, and gyroscope data)
2. Preprocessing parameters (such as sampling frequency, filter parameters) may need adjustment based on specific data
3. Augmentation factor is recommended to be 2-5x; too much may lead to overfitting
4. GPU acceleration is recommended for training large-scale datasets

## 📧 Contact

For questions or suggestions, please contact us through Issues or Pull Requests.

---

**Language**: [中文](README.md) | **English**
