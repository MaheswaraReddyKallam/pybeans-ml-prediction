# PyBeans Customer Prediction Model ☕🤖

### Overview
This project is an end-to-end Machine Learning pipeline built for a fictional coffee shop (PyBeans). It uses traditional classification algorithms to predict whether a customer will purchase a pastry based on their age, coffee consumption habits, and distance from the store.

### Technologies Used
* **Python** * **Pandas** (Data manipulation and cleaning)
* **Scikit-Learn** (Machine Learning, Train/Test Split, Model Evaluation)

### The Machine Learning Pipeline
1. **Data Prep:** Cleaned raw customer data and split it into Features (`X`) and Targets (`y`).
2. **Preprocessing:** Utilized `StandardScaler` to put varying metrics (Age vs. Distance) on the same mathematical scale.
3. **Training:** Split the data (80/20) and trained a `LogisticRegression` model on the training set.
4. **Evaluation:** Tested the model against unseen data, achieving a high accuracy score using `accuracy_score` metrics.

### How to Run
Ensure you have the required libraries installed:
`pip install pandas scikit-learn`
Run the script to see the model's predictions vs. actual answers.
