# Deep-Learning-Based-Intrusion-Detection-System
CyberGuard: AI-Powered Intrusion Detection

Overview

CyberGuard is a deep learning-based Intrusion Detection System (IDS) designed to detect and classify cyber threats using a Multi-Layer Perceptron (MLP) neural network. The project leverages machine learning techniques to preprocess network traffic data, train an intelligent model, and classify network activity as benign or malicious.

Features

Uses deep learning (MLP) for intrusion detection.

Preprocesses and normalizes network traffic data.

Automatically encodes categorical features.

Provides high detection accuracy for cybersecurity threats.

Compares results with traditional machine learning models.

Dataset

The dataset used for this project is a cybersecurity attack dataset sourced from Kaggle. It includes various network traffic attributes and labeled attack types. The preprocessing steps include:

Removing irrelevant features (e.g., IP addresses, timestamps).

Handling missing values.

Encoding categorical variables.

Normalizing numerical features.

Installation

To set up this project, install the required dependencies:

pip install pandas numpy scikit-learn tensorflow keras matplotlib

Usage

Preprocess the dataset:

Load and clean the data.

Encode categorical variables.

Normalize numerical values.

Train the model:

Define the MLP architecture.

Train the model using an 80-20 train-test split.

Optimize the model using the Adam optimizer.

Evaluate the model:

Measure accuracy, precision, recall, and F1-score.

Compare with traditional ML models (e.g., Decision Trees, SVM).

To run the project in Google Colab:

# Load dataset
import pandas as pd
df = pd.read_csv("cybersecurity_attacks.csv")

# Preprocess, train, and evaluate the model
# (Refer to project notebook for detailed implementation)

Results

Achieved over 90% accuracy in detecting cyber threats.

Demonstrated the effectiveness of deep learning over traditional methods.

Future improvements include exploring LSTMs for sequential attack detection.

Future Work

Enhance feature selection for better classification accuracy.

Implement additional deep learning architectures (e.g., LSTMs, CNNs).

Deploy the model in a real-time cybersecurity environment.

Contributors

Sehman Kabir – Research, Implementation, and Documentation.

License

This project is licensed under the MIT License.

CyberGuard: Protecting Networks with AI-Powered Intrusion Detection. 🚀

