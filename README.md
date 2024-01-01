# Driving Insights: "Stress Detection via Heartbeat Big Data Analytics through RNN"
Stress Detection via Heartbeat Big Data Analytics: Utilizes GRU-based neural network on MIT-BIH &amp; PTB datasets. Achieves 98.19% accuracy. Streamlit app for interactive visualization. Explore stress-heartbeat dynamics in this impactful project.

## Abstract

This project delves into the correlation between stress and heartbeat signals by leveraging a curated dataset from the MIT-BIH Arrhythmia Dataset and The PTB Diagnostic ECG Database. Through preprocessing, a deep neural network, specifically a Gated Recurrent Unit (GRU) architecture, is trained to differentiate normal and stress-indicative abnormal heartbeats. Abnormal heartbeats are considered potential indicators of stress, offering a non-invasive means for stress detection. The project contributes to stress detection methodologies, providing insights into the dynamic interplay between stress and cardiovascular responses, with potential applications in mental and cardiovascular well-being.

## Introduction

In the modern era, stress has become a pervasive challenge affecting mental and physical well-being. This project explores the intricate relationship between stress and the cardiovascular system, particularly focusing on the dynamic patterns of the human heartbeat. The research utilizes advanced machine learning techniques, specifically recurrent neural networks (RNNs), to discern between normal and stress-indicative abnormal heartbeats. The project aligns with the broader understanding of stress in contemporary life, emphasizing the need for innovative approaches to stress management.

## Project Tools:

- **Programming Language:** Python
- **Libraries & Frameworks:**
  - TensorFlow for building and training the deep neural network
  - Scikit-learn for data preprocessing and evaluation metrics
  - Streamlit for creating an interactive web app
  - Seaborn and Matplotlib for data visualization
- **Dataset:** ECG Heartbeat Categorization Dataset
- **IDE:** Google Collab for code development
- **Version Control:** Git and GitHub for collaborative development
- **Documentation:** Markdown for clear and organized project documentation
- **Data Visualization:** Seaborn and Matplotlib for creating visualizations in the project report
- **Others:** Pandas for data manipulation, NumPy for numerical operations

## Methodology

The methodology involves meticulous loading and preprocessing of heartbeat data, utilizing a curated dataset from MIT-BIH Arrhythmia and PTB Diagnostic ECG databases. The Gated Recurrent Unit (GRU) architecture is chosen for its ability to capture temporal dependencies in sequential data. The project includes data collection and preprocessing, model architecture design, training, evaluation, results interpretation, and deployment. A Streamlit app is developed for visualization, providing a comprehensive understanding of the stress-heartbeat relationship.
![Methodology](https://github.com/SadiaAdrees/Stress-Detection-via-Heartbeat/assets/110346827/9293c0ae-1efe-4dcf-b46a-3c54b61de4fd)


## Dataset Source 
The dataset used in this project is a curated subset from the MIT-BIH Arrhythmia Dataset and The PTB Diagnostic ECG Database. It encompasses diverse heartbeat signals, preprocessed and segmented electrocardiogram (ECG) data. With each segment corresponding to an individual heartbeat, the dataset provides a rich source for training a Gated Recurrent Unit (GRU) neural network. The curated dataset ensures uniformity, facilitating nuanced exploration of stress-related patterns in cardiovascular responses.
[Dataset Link](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) 

## Model Working

The model employs a GRU layer to process sequential heartbeat data, followed by a Flatten layer and a Dense layer for binary classification. The model is trained using the Adam optimizer and binary cross-entropy loss function. The training process involves 100 epochs, with early stopping to prevent overfitting. The model's working is detailed, emphasizing key components such as the input layer, data expansion, GRU layer, Flatten layer, and the output layer.

## Accuracy

The model achieves an impressive accuracy of 98.19% on the test set, demonstrating its effectiveness in classifying normal and abnormal heartbeats. The Area Under the Curve (AUC) is reported as 0.9941, indicating excellent performance. The classification report shows high precision, recall, and F1-score for both normal and abnormal heartbeats. Overall accuracy is reported as 97%, showcasing the model's reliability in making accurate predictions across both classes.

## Code

The provided code implements data loading, preprocessing, model building, training, evaluation, and a Streamlit app for visualization. The code is well-organized and demonstrates the steps involved in developing and deploying the stress detection model.

## Front-end

The front end is designed using Streamlit, providing an interactive and user-friendly interface for exploring the model's information, classification report, confusion matrix, and other relevant visualizations.

## Result

The model exhibits robust performance in differentiating between normal and abnormal heartbeats, with high accuracy and strong performance metrics. The results confirm the model's potential for real-world applications in stress detection based on heartbeat patterns.

## Conclusion

In conclusion, the project successfully develops a deep neural network model for stress detection via heartbeat analysis. The model's exceptional accuracy and performance metrics validate its efficacy in capturing both normal and abnormal heartbeat patterns. The findings contribute to the fields of physiological signal analysis and stress management, suggesting potential applications for real-time stress detection in diverse scenarios. Future work could focus on refining adaptability, considering long-term trends, and addressing ethical considerations in stress detection.

## Acknowledgement

The project acknowledges the support and guidance of Instructor Maam Sehrish Aqeel from the Department of Computer Science at the University of South Asia.

## References

The project draws insights from relevant literature, including studies on deep learning for heartbeat classification and related advancements in neural networks.

