# 💳 Financial Fraud Detection using Machine Learning
An end-to-end Machine Learning project that detects fraudulent financial transactions using classification algorithms. The project includes data preprocessing, exploratory data analysis (EDA), model training, performance evaluation, and interactive visualizations to identify suspicious transaction patterns.

**📘 Project Overview**

Online fraud in digital transactions is a growing concern in today’s cashless economy.  
This project aims to **predict fraudulent transactions** using machine learning algorithms that analyze transaction patterns and user behavior.

The system uses a labeled dataset of transaction records and applies classification models such as **Random Forest, Decision Tree, Logistic Regression, KNN,** and **SVM**.  
Among these, the **Random Forest Classifier** demonstrated the highest **accuracy and F1-score**, making it the most robust and effective model for fraud detection.



## 🎯 Objectives

- Detect fraudulent transactions automatically based on input features.  
- Compare multiple ML models to identify the most effective one.  
- Visualize fraud patterns and model performance for easy interpretation.  
- Build a foundation for future fraud analytics and prevention systems.



## 🧩 Features

- 🧮 Data preprocessing and encoding of transaction types  
- 📈 Interactive data visualization using **Plotly** and **Seaborn**  
- 🧠 Model training and accuracy comparison among 5 algorithms  
- 📊 Graphical representation of model performance  
- 💾 Export of results to a CSV file for record and analysis  
- 🌐 Browser-based visualization (Plotly charts open in your default browser)



## ⚙️ Machine Learning Models Implemented

| Algorithm | Description | Key Strength |
|------------|--------------|---------------|
| Logistic Regression | Simple and interpretable model | Good baseline for binary classification |
| Decision Tree | Tree-based classifier | Handles non-linear relationships well |
| Random Forest | Ensemble of Decision Trees | Highest accuracy and robustness |
| K-Nearest Neighbors | Distance-based classifier | Works well with balanced datasets |
| Support Vector Machine | Margin-based classifier | Effective for small, well-separated data |



## 🧠 Technologies Used

  - **Language:** Python  
  - **Libraries:**  
  - pandas  
  - numpy  
  - scikit-learn  
  - seaborn  
  - matplotlib  
  - plotly  



## 🧾 Dataset Information

The dataset contains synthetic transaction records with the following key features:
- `type`: Transaction type (Cash-In, Cash-Out, Payment, Transfer, Debit)  
- `amount`: Transaction amount  
- `oldbalanceOrg`: Sender’s balance before transaction  
- `newbalanceOrig`: Sender’s balance after transaction  
- `isFraud`: Target label indicating whether the transaction is fraudulent (1) or genuine (0)



## 📊 Visualizations

1. **Transaction Type Distribution**  
   - Pie chart displaying proportion of each transaction category.

2. **Correlation Heatmap**  
   - Highlights relationships between transaction features.

3. **Model Accuracy Comparison**  
   - Bar chart showcasing the accuracy of each ML model.

4. **Performance Table**  
   - Displays accuracy and F1-scores for all models in a tabular format.




## 🧰 Installation & Setup

  1. Clone the repository:
  git clone https://github.com/Viiddzzz/financial-fraud-detection.git
cd financial-fraud-detection

  2. Install dependencies:
   pip install pandas numpy matplotlib seaborn plotly scikit-learn

   3.Run the project:
     python fraud_detection.py

   4.View results:
     Real-time charts open in your browser.
     Model comparison results are saved in fraud_detection_comparison.csv.



**🧮 Sample Output**

Console Output Example:

🏆 Best Model: Random Forest (Accuracy: 0.97)

💾 Results saved to fraud_detection_comparison.csv

🎯 Online Fraud Detection System execution complete!




**Visualization Output Example:**

Pie chart of transaction types

Bar graph comparing model accuracies

Confusion matrix (optional)




**🧩 File Structure**

📁 financial-fraud-detection/
│
├── fraud_detection.py          # Main Python project code

├── fraud_detection_comparison.csv  # Saved model results

├── README.md                   # Project documentation

└── requirements.txt             # Dependencies list




**🏁 Results & Insights**

Random Forest Classifier outperformed other models with the highest accuracy and stability.

Transaction type and amount were key indicators in predicting fraudulent activity.

Visualization improved interpretability for non-technical audiences.

The system can easily integrate into a larger fraud prevention pipeline.


## 💼 Business Impact

- Helps financial institutions detect suspicious transactions more efficiently.
- Reduces financial losses through early fraud detection.
- Demonstrates the practical application of machine learning in financial security.
- Supports data-driven decision-making using interactive visualizations.

**🚀 Future Enhancements**

Integrate with a real-time transaction API for live fraud detection.

Implement deep learning (LSTM/ANN) for time-series transaction data.

Add streamlit or Flask dashboard for web-based visualization.

Perform feature engineering using advanced statistical methods.


**⭐ Acknowledgements**

Inspired by online payment fraud detection research papers.

Developed as part of a college mini-project on Machine Learning Applications in Cybersecurity.

Libraries used: scikit-learn, Plotly, Seaborn, and Matplotlib.

## 🛠 Skills Demonstrated

- Python Programming
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Data Visualization
- Machine Learning
- Classification Algorithms
- Model Evaluation

🧑‍💻 Author Vidyashree s
    Aspiring Data Analyst | Machine Learning Enthusiast
📧 vidyaa1103@gmail.com

🌐 https://github.com/Viiddzzz/financial-fraud-detection



💡 This project demonstrates the practical application of machine learning for real-world financial fraud detection, focusing on accuracy, interpretability, and data visualization.
