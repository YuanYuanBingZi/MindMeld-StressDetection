# MindMeld: Stress Detection Using NLP and Machine Learning

### Overview
**MindMeld** is a machine learning project designed to detect stress-related content from online posts using natural language processing (NLP). This project compares different feature extraction methods, such as **Bag of Words (BoW)** and **Word2Vec**, and evaluates several machine learning models, including **Logistic Regression**, **Support Vector Machines (SVM)**, and **Convolutional Neural Networks (CNN)**. The goal is to understand how different approaches affect the performance of stress classification and explore their real-world applications in mental health monitoring.

### Table of Contents
- [Motivation](#motivation)
- [Features](#features)
- [Technologies](#technologies)
- [Data](#data)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Next Steps](#next-steps)
- [How to Run](#how-to-run)

### Motivation
With the increasing reliance on online platforms for expressing emotions and thoughts, identifying stress in digital communications has become crucial. Accurately classifying stress-related content can help in mental health monitoring, providing timely interventions, and contributing to personalized support. This project demonstrates how machine learning and NLP can address these challenges.

### Features
- **NLP-Based Stress Prediction:** Classifies online posts into stress-related and non-stress-related categories.
- **Multiple Text Representation Techniques:** Compares the effectiveness of BoW and Word2Vec for text feature extraction.
- **Model Comparison:** Evaluates three models (Logistic Regression, SVM, CNN) to determine which performs best on stress classification tasks.
- **Critical Insights:** Provides explanations for model performance, including the impact of dataset size and feature extraction methods on CNNs and traditional models like Logistic Regression and SVM.

### Technologies
- **Python 3.8+**
- **Scikit-learn**
- **Keras with TensorFlow backend**
- **NLTK for Text Preprocessing**
- **Pandas and Numpy for Data Handling**

### Data
The project utilizes a **stress prediction dataset** from Kaggle, which contains posts from various subreddits related to mental health, such as PTSD, anxiety, relationships, and homelessness. It consists of 2820 rows of text data, preprocessed to remove stop words, punctuation, and special symbols.

**Source:**  
[Kaggle Stress Prediction Dataset](https://www.kaggle.com/datasets/kreeshrajani/human-stress-prediction)

### Modeling Approach
1. **Text Preprocessing:**
   - Tokenization, removing stopwords, converting text to lowercase.
2. **Feature Extraction:**
   - **Bag of Words (BoW):** Simple frequency-based representation of the text.
   - **Word2Vec:** Dense vector embeddings to capture semantic context.
3. **Model Selection:**
   - **Logistic Regression:** A simple, yet effective baseline model for binary classification.
   - **Support Vector Machines (SVM):** Creates decision boundaries to separate stress and non-stress content.
   - **Convolutional Neural Networks (CNN):** Applied to Word2Vec embeddings to capture local and hierarchical patterns in text.

### Results
- **Best Performing Models:**  
  - **Bag of Words + SVM** achieved the highest accuracy (72.71%) in predicting stress-related posts.
  - **Word2Vec + CNN** had a lower performance due to the small dataset size and complexity of the model (54.57% accuracy).
  
- **Key Insights:**  
  - **BoW outperformed Word2Vec** for this specific dataset due to the strong correlation between word frequency and stress.
  - **CNN struggled** due to limited data, but has potential in future work with larger datasets or more advanced embeddings (e.g., BERT).

### Next Steps
To improve upon the current results:
1. **Expand the dataset:** Collect more data from diverse sources to provide richer information for training models like CNN.
2. **Explore advanced embeddings:** Investigate techniques like GloVe and BERT, which may capture more nuanced semantic information.
3. **Optimize CNN architecture:** Experiment with different architectures and hyperparameters to enhance the CNN's ability to generalize.

### How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YuanYuanBingZi/MindMeld-StressDetection.git
   ```

2. **Install required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   Download the stress dataset from [Kaggle](https://www.kaggle.com/datasets/kreeshrajani/human-stress-prediction) and place it in the `/data` directory.

4. **Preprocess the data:**
   Run the preprocessing script to clean and tokenize the text:
   ```bash
   python preprocess.py
   ```

5. **Train the models:**
   To train the models, run the following command:
   ```bash
   python train.py
   ```

6. **Evaluate the models:**
   After training, evaluate the models using:
   ```bash
   python evaluate.py
   ```

7. **View the results:**
   The evaluation script will output the accuracy of each model and generate visualizations comparing their performance.

### Conclusion
MindMeld is a practical demonstration of how machine learning and NLP can address mental health challenges through stress detection. The project not only highlights different approaches to text classification but also provides thoughtful analysis and suggestions for future improvements.
