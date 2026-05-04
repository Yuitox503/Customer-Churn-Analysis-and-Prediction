# Customer Churn Prediction & Analytics


## Project Overview
This project focuses on analyzing and predicting customer churn using machine learning and business intelligence tools. The objective is to identify high-risk customers, understand key churn drivers, and provide actionable insights to improve customer retention.

The project combines data analysis, machine learning, and interactive dashboarding to deliver a complete end-to-end solution.

## Objectives
- Analyze customer behavior and identify factors contributing to churn  
- Build predictive models to identify high-risk customers  
- Develop an interactive dashboard for business decision-making  
- Provide actionable insights to reduce customer attrition  

## Tech Stack
- Python (Pandas, NumPy, Scikit-learn)  
- Power BI (Dashboard and Visualization)  
- Jupyter Notebook (EDA and Modeling)

## Exploratory Data Analysis (EDA)
Key insights from the dataset include:
- Customers on month-to-month contracts exhibit the highest churn rates  
- Customers with shorter tenure are significantly more likely to churn  
- Fiber optic users show elevated churn, suggesting potential dissatisfaction  
- Higher monthly charges are associated with increased churn  

## Feature Engineering
- Converted categorical variables into numerical format using binary encoding and one-hot encoding  
- Removed high-cardinality and low-value features (e.g., City, Zip Code)  
- Eliminated data leakage variables (e.g., Churn Score, Churn Reason)  
- Prepared dataset for machine learning modeling  

## Machine Learning Models
Models implemented:
- Logistic Regression  
- Random Forest  

### Final Model: Optimized Random Forest

Performance:
- Accuracy: ~77.5%  
- Recall (Churn): ~73%  
- Precision (Churn): ~56%  

The final model was selected based on its balanced performance in identifying churners while minimizing false positives.

## Power BI Dashboard
The project includes an interactive dashboard with the following sections:

### 1. Overview
- Key performance indicators such as churn rate, total customers, and revenue  
- Overall churn distribution  

### 2. Customer Behaviour
- Churn analysis by contract type, tenure, and payment method  
- Demographic segmentation  

### 3. Services and Product Usage
- Impact of service features (e.g., security, streaming, support) on churn  

### 4. Revenue and Risk Analysis
- Financial impact of churn  
- Identification of high-risk customer segments  
- Analysis of churn reasons  

### 5. Customer Risk and Prediction
- Churn probability distribution  
- Identification of high-risk customers  
- Model-driven segmentation  

## Key Business Insights
- Long-term contracts significantly reduce churn  
- New customers require targeted onboarding and retention strategies  
- High-value customers can still churn, posing revenue risk  
- Service quality and pricing are major contributors to churn  

## Business Impact
- Enables targeted retention strategies for high-risk customers  
- Improves efficiency in customer engagement efforts  
- Supports data-driven decision-making  
- Helps reduce potential revenue loss  

## Future Improvements
- Deploy the model as a web application (e.g., Streamlit)  
- Enable real-time prediction capabilities  
- Perform advanced hyperparameter tuning  
- Integrate SQL for scalable data pipelines  

---

## Author
Ferdinand Taslim
