# Predicting Customer Churn Using Machine Learning

# Introduction
Dived into developing and depolying a predictive machine learning model focusing on predicting customer churn from historical data. This involved Defining the Problem & objective of the project, Data Collection, Data cleaning and Preprocessing, Exploratory Data Analysis, Feature Engineering, Model Creation, Model Evaluation, Model Optimization, and Model Deployment

Python code? Check them out here on GitHub: https://github.com/anormanangel/Predicting-Customer-Churn-Using-Machine-Learning

# Background
Driven by the quest to optimize business processes using machine learning, this project was born bt a desire to to develop a model that can accurately Predict customer churn based on customer data and relevant indicators. 

The data for this project was obtained from Kaggle for a Tours and Travel company in the United States. This is a historical customer dataset where each row represents one customer and offers straightforward insights. As it is more cost-effective to retain existing customers than acquire new ones, our primary goal in this analysis is to predict which customers will remain loyal to our services.

This data set provided key indicators of customer churn allowing us to anticipate the behaviors that contribute to customer retention and predict what behavior will help us to retain customers.

The dataset includes information about:

* Age of users
* FrequesntFlyer: Whether the customer takes frequent flights
* AnnualIncomeClass: Class of annual income of the user
* ServicesOpted: Number of times services opted during recent years
* AccountSyncedToSocialMedia: Whether Company Account of the User is Synchronized to their Social Media
* BookedHotelOrNot: Whether the customer booked lodging/Hotels using company services
* Target: 1 = Customers Churns, 0 = Customer Dosent Churn

Link to dataset on Kaggle: https://bit.ly/3rzy2Sw


### The three Questions I was asking were;

1. What are the key indicators of customer churn?
2. How can predictive models be built to predict customer churn for a tour and travel company based on customer data?
3. How can insights gained from the predictive model be utilized to tailor retention efforts and allocate resources more effectively, including targeted marketing campaigns, personalized offers, enhanced customer retention, or improved customer satisfaction?

# Tools I Used
For my machine learning project to predict customer churn, I used the powers of several key tools:

1. **Data manipulation and analysis:** pandas & numpy
2. **Handling missing data:** missingno
3. **Data visualization:** matplotlib & seaborn
4. **Machine learning:** scikit-learn (for various models and preprocessing), imblearn (for handling imbalanced data)
5. **Model development:**
    * Logistic Regression
    * K-Nearest Neighbors
    * Random Forest
    * Gradient Boosting
    * Decision Tree
    * Support Vector Machine (SVM)
    * Naive Bayes
6. **Model evaluation and optimization:**
    * GridSearchCV (for hyperparameter tuning)
    * Various metrics like accuracy_score, classification_report, confusion_matrix
7. **Data preprocessing:** StandardScaler, LabelEncoder
8. **Model deployment:** Gradio for creating a web interface for the model

# The Analysis 
### 1. What are the key indicators of customer churn?
The key indicators of customer churn were age, frequent flyer status, and income class. Specifically,
* The younger customers (27-28y) tend to churn more often.
* Customers who opted for more services (5+) churned more than those who opted for a few services.
* Frequent flyers churn more than non-frequent flyers.
* High-income individuals churn more than low- and middle-income classes.
* Customers who synced their accounts to social media churned more than those who didn’t.
* Customers who used the company’s service to book a hotel churned more than those who didn’t. 

### 2. How can predictive models be built to predict customer churn for a tour and travel company based on customer data?
#### Data Cleaning & Preprocessing
* I started by checking for missing data in the dataset. The main reason for handling missing data in datasets for machine learning is to improve model performance and accuracy. Missing data can skew the results and lead to inaccurate predictions. Properly managing missing data ensures that the model can make reliable and valid predictions, thereby enhancing its overall performance and effectiveness.

![Handling Missing Data](https://github.com/anormanangel/Predicting-Customer-Churn-Using-Machine-Learning/blob/main/Assets/Missing%20Numbers.png)

*Bar chat showing all columns in the dataset with no missing data*

#### Exploring the relationship between variables**
* Univariate Analysis: I explored the distribution of churn variables individually to understand their distribution and characteristics.
The data is unbalanced, with 76.5% are customers in the "no-churn" class (modest class imbalance 3:1), which needs to be taken into account when training and evaluating the models.

![Churn Distribution](https://github.com/anormanangel/Predicting-Customer-Churn-Using-Machine-Learning/blob/main/Assets/Churn%20Distribution.png)

*Bar graph visualizing churn distribution*

* Bivariate: For bivariate analysis, I explored the relationships between pairs of variables. The  chart below shows churn distribution by age.  Younger customers (27-28y) tend to churn proportionally more often than older customers.

![Churn by Age](https://github.com/anormanangel/Predicting-Customer-Churn-Using-Machine-Learning/blob/main/Assets/Churn%20by%20Age.png)

*Bar graph visualizing churn rate by Age*

* Multivariate Analysis: I examined the relationships between three or more variables. The correlation matrix indicates that annual income class and frequent flyer status are correlated. This also indicates that frequent flyers are associated with churning.

![Correlation Matrix](https://github.com/anormanangel/Predicting-Customer-Churn-Using-Machine-Learning/blob/main/Assets/Correlation%20Matrix.png)

*Spearman's correlation showing the relationship between variables*

#### Machine Learning Modeling

Train test split: The process involves dividing the dataset into two parts: one for training the machine learning model and the other for testing its performance. The features (independent variables) are separated from the target variable (the outcome we want to predict). About two-thirds of the data is used to train the model, while the remaining one-third is reserved to test how well the model predicts outcomes on new, unseen data. This ensures the model is both accurate and reliable

**Model Perfomance & Evaluation**
Here we test some classic algorithms but also explore some classifiers that specifically account for unbalanced data (using imlearn).
* Among the models we compared, the Decision tree classifier performed the best with an overall accuracy of 90% with F1 Score of 93 and 77 for class 0 and class 1 respectivelt. The Gradient Boosting Classifier followed in order to predict customer churn. It performed with an overall accuracy of 89%, as well as an F1 score of 76 for the minority class.
* The random forest classifier (88% Accuracy) performs overall better than logistic regression. The balanced random forest classifier classifies the minority class often correctly, however, at the cost of many false negatives.
* With an accuracy is 80%, Logistic Regression performs well but needs improvement. Introducing the balanced weight improved the recall for class 0 (from 0.94 to 0.79) but did not improve the overall performance of the model (76% accuracy)
* The balanced bagging classifier also favors the underrepresented class 1 (churned), providing the best f1-value for the minority class and an overall good accuracy of 87%.
* Based on the analysis, the K Nearest Neighbors (KNN) model performs best when considering only the nearest neighbor (the single closest data point) when making predictions making it sensitve to outliers and a risk of overfitting

![Decision Tree](https://github.com/anormanangel/Predicting-Customer-Churn-Using-Machine-Learning/blob/main/Assets/Decison%20Tree.png)

*Confusion matrix for Decision Tree*

**Model Optimization**
* Random Forest, Gradient Boosting, and Balanced Bagging are ensemble methods that generally perform well in various scenarios.
* Introducing class_weight='balanced' in a classification model as a way of addressing imbalanced data. It assigns higher weights to the minority class, making the model pay more attention to it during training.

**Hyperparameter Tuning**
* Tune the hyperparameters of the Decision Tree Classifier and Gradient Boosting Classifier using techniques such as Grid Search. Experiment with parameters like n_estimators, learning_rate, max_depth, and subsample.
After Hyperparameter Tuning, The Model accuracy for Gradient Boosting improved from 89% to 90%. Retraining the Decision tree improved its accuracy by 0.9514866979655712 (95%)

**Criteria for Choice of Model to Deploy**
* Both the Decision Tree and Random Forest models have the same accuracy of approximately 0.9515, meaning they both correctly classify around 95.15% of instances.
* But the Decision Tree model has a higher precision of approximately 0.9578 compared to the Random Forest model's precision of approximately 0.9450. This means that the Decision Tree model tends to make fewer false positive predictions compared to the Random Forest model.

* Precision is important in the context of churn prediction because it represents the proportion of correctly identified churn cases out of all customers predicted to churn. High precision means that the model correctly identifies a high percentage of customers who are likely to churn, minimizing false positives. In the context of churn prediction, high precision ensures that resources and efforts to retain customers are efficiently allocated to those who are truly at risk of leaving.

* For our case, the cost of false positives (incorrectly identifying loyal customers as churners) is relatively high (e.g., offering unnecessary retention incentives to loyal customers), hence we are higher precision prioritized over recall.

### Deploy decision tree Model Using Gradio
Gradio is a Python library that simplifies the process of creating and deploying machine learning models with user interfaces. It provides an easy-to-use Python library for building web-based interfaces for machine learning models, allowing users to interact with the models through a browser.

### Insights & Interpretation
1. The key indicators of customer churn were age, frequent flyer status, and income class. Specifically,

2. The younger customers (27-28y) tend to churn  more often.

3. Customers who opted for more services (5+) churned more than those who opted for a few services.

4. Frequent flyers churn more than non-frequent flyers.

5. High-income individuals churn more than low- and middle-income classes.

6. Customers who synced their accounts to social media churned more than those who didn’t.

# What I Learned
* 🧩 **Advanced Python Concept for Data Analytics:** Mastered the art of using Python for data manipulation, processing, visualization and machine learning.

* 📊 **Machine Learning:** I implemented various classification models to predict customer churn including Logistic Regression, K-Nearest Neighbors, Random Forest, Gradient Boosting, Decision Tree, Support Vector Machine (SVM) & Naive Bayes.

* 📌 **Deploying Models using Gradio** I build a web application using Gradio, integrating a machine learning model into a production environment where it can take in an input and return an output

* 💡 **Analytical Wizardry:** Leveled up my real-world problem-solving skills turning questions into actionable insights using Python


# Conclusions

This project enhanced my data science & analytics skills using Python to build machine learning models and and provided valuable insights to predict customer churn and optimize business processes. Businesses can make data-driven decision to identify customers at risk of churning and prevent them from churning thereby improving customer retention, increasing customer satisfaction, and preventing loss of revenue, 


