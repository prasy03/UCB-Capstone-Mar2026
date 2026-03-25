# **Project Title**

**Machine Learning for Supply Chain Optimization: Predictive Demand, Risk Classification, and Inventory Strategy**

## Author: Sylvester Prasanna 

Date: Mar 23, 2026 (revised, including Time Series model in a separate Jupyter notebook)

---

## Introduction: 

*This project aims to address fundamental issues in global supply and demand pipelines, which significantly impact macroeconomics and daily life.*  
*It utilizes Machine Learning methodologies to analyze key influencing factors like stockouts, shipping times, and evolving customer needs to improve operational flow and profitability.*

## Executive Summary 

Global supply chain management is the heart of the modern world. It impacts every household every day in multiple ways. Retail, commercial, medical, manufacturing, industry and all other business organizations directly contribute to macro economics which has both financial, economic and lifestyle impact to every corner of the world. Supply, demand, shipping times, predictable delivery, on time commitment, evolving customer needs, stockouts, over stocking, perishable products, seasonal products etc are some key factors which strongly influence the global supply chain. A majority of these factors have a  strong correlation and determine the flow and profitability of the operations. Using Machine learning methodologies, this project is a small effort to solve basic issues that impact today's supply and demand pipelines. The modern world fundamentally relies on global supply chain management, which affects every household daily across various sectors, including retail, commerce, medicine, manufacturing, and industry. These sectors collectively shape macroeconomics, influencing the financial, economic, and lifestyle aspects of the world. The flow and profitability of these operations are heavily influenced and correlated by key factors such as supply, demand, shipping times, predictable delivery, on-time commitment, evolving customer needs, stockouts, overstocking, and the handling of perishable and seasonal products. This project seeks to address fundamental issues within today's supply and demand pipelines through the application of Machine Learning methodologies.

## Rationale 

*(Would anyone care about this question)* 

Buyers make informed choices when aware of a product's source, distribution, and pipeline. These factors significantly influence product pricing, regardless of size. Alternatively large organizations require in-depth analysis to optimize their operations. Some of the key factors include the following: 

1. **Enterprises (Retail, Manufacturing, Logistics, and E-commerce):**  
   * **Increased Profitability:** By using Machine Learning to improve supply-demand forecasting, they can reduce costly issues like *stockouts* (lost sales) and *overstocking* (wasted capital, storage costs, and potential product obsolescence).  
   * **Operational Efficiency:** Optimizing shipping times and ensuring *predictable delivery* leads to smoother operations, competitive advantage and lower logistics costs.  
   * **Better Inventory Management:** ML is vital for managing complex factors like *perishable* and *seasonal products*, minimizing waste and maximizing sales opportunities.  
2. **Buyers / Customers / Consumers:**  
   * **Product Availability:** Fewer stockouts mean they can buy what they want, when they want it.  
   * **Lower Costs:** Efficiencies gained by businesses often translate into more competitive pricing for the consumer.  
   * **Reliable Delivery:** *On-time commitment* and predictable delivery schedules improve the overall shopping experience, especially for e-commerce.  
3. **Economists and Government Agencies:**  
   * **Macroeconomic Stability:** A well-functioning global supply chain is a cornerstone of economic health. Issues addressed by the project (like unpredictable flow) can lead to inflation or economic slowdowns. Improving efficiency contributes to *macroeconomics*.  
   * **Resilience:** The project aims to solve "basic issues," which can help supply chains better withstand major disruptions (like pandemics or geopolitical events) by improving fundamental visibility and predictability.

## 

## Research Question

*What are you trying to answer ?* 

*The “Why” question?* 

- *How can we predict and mitigate the impact of market and seasonal variability on product delivery lead times?*

The key objectives and resulting deliverables for this project are:

1. **Demand Forecasting:** Predict near-term product demand across high-demand regions using historical data.  
2. **Risk Analysis and Classification:** Employ machine learning models to classify products and locations, identifying the risk of future stockouts or overstock scenarios to facilitate data-driven decision-making.  
3. **Differentiated Planning Strategy:** Develop customized planning strategies by segmenting products based on location and demand variability.

The Supply Chain domain  offers a rich and challenging environment for applying and validating ML methodologies, making it a valuable case study.

## Data Source 

The chosen dataset was selected after an extensive search across various platforms, including Kaggle, UC Irvine, datacatalog, and opendatabay.

1. Original source: DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS (year of publication: 2019\) source [link](https://data.mendeley.com/datasets/8gx2fvg2k6/5)  
   1. This dataset consists of 52 features (columns) and over 182K samples (rows). 

To maintain statistical accuracy and facilitate quicker model training with reduced complexity, the dataset was streamlined. This lean version focuses solely on 12 months (the dataset comprises **50,000 rows and 53 features**.) of transaction data and is designed to meet the following core criteria:

1. **Operational supply chain data**: purchase orders, promised vs actual delivery dates, supplier lead times, shipment tracking data, and inventory levels.  
2. **Demand signals**: historical demand, forecast inputs, seasonality patterns,    
3. **External factors**: Shipping, source and destination region    
4. **Engineered features**.

## 

## Methodology

*What methods are you trying to use to answer your question ?* 

The project stages have been categorized and explained according to the CRISP-DM methodology.

1. Step 1: Business Understanding    
2. Step 2: Data understanding and analysis   
3. Step 3: Data Preparation, Exploratory data analysis   
4. Step 4: Modeling   
5. Step 5: Evaluation  
6. Step 6: Deployment / Inference

### **Step 1: Business Understanding**

The objective of this project is to leverage data-driven insights to optimize supply chain efficiency. Specifically, the business goals are to:

* **Sales prediction:** Use historical data to predict sales and forecast revenue values accurately.   
* **Mitigate Logistics Risk:** Predict the likelihood of shipment delays for improved operations and increased customer satisfaction.  
* **Optimize Inventory:** Identify frequently bought items and group market regions into clusters to recommend strategic stock placement.  
* **Analyze Trends:** Understand seasonal sales patterns and regional performance.

### **Step 2: Data Understanding and Analysis**

The dataset consisted of $50,107$ records across $53$ variables.

* **Key Variables:** Included shipping days (real vs. scheduled), profit margins, sales totals, delivery status, market regions, and product categories.  
* **Initial Findings:** A significant portion of shipments ($54.6\\%$) were flagged with a Late\_delivery\_risk.  
* **Data Quality:** Identified that columns like Order Zipcode and Product Description were entirely empty and required removal. Numerical data showed wide ranges in sales (from $9.99 to over $ 1,900), indicating a diverse product portfolio.

### **Feature Identification and Initial Cleaning**

The dataset has been categorized into three groups based on their data types and relevance to supply chain modeling (Sales Prediction, Delay Prediction, and Clustering).

#### **1\. Numerical Columns**

These represent measurable quantities and ratios used for mathematical modeling:

* **Shipping & Logistics:** Days for shipping (real), Days for shipment (scheduled), Late\_delivery\_risk.  
* **Financials:** Benefit per order, Sales per customer, Order Item Discount, Order Item Discount Rate, Order Item Product Price, Order Item Profit Ratio, Sales, Order Item Total, Order Profit Per Order, Product Price.  
* **Geography:** Latitude, Longitude, Customer Zipcode.  
* **Operations:** Order Item Quantity.

#### **2\. Categorical Columns**

These represent discrete labels used for classification, trend analysis, and one-hot encoding:

* **Customer & Market:** Type, Customer Segment, Market, Customer City, Customer Country, Customer State.  
* **Order & Shipping:** Delivery Status, Order City, Order Country, Order Region, Order State, Order Status, Shipping Mode.  
* **Product & Category:** Category Name, Department Name, Product Name.  
* **Temporal (Datetime):** order date (DateOrders), shipping date (DateOrders).

#### **3\. Dropped Columns**

The following **17 columns** were removed because they are either empty, contain unique identifiers that do not aid in general pattern recognition, or contain sensitive personal information:

* **Sensitive/Irrelevant:** Customer Email, Customer Password, Customer Fname, Customer Lname, Customer Street.  
* **Empty/Technical:** Product Description, Order Zipcode, Product Image.  
* **Internal IDs:** Customer Id, Order Id, Order Item Id, Product Card Id, Order Customer Id, Category Id, Department Id, Product Category Id.  
* **Constant Values:** Product Status (contains only zero values).

**Result:**

The processed data df\_cleaned now contains **36 columns** and **50,107 rows**.

I have saved the result to df\_cleaned.csv. Let me know when you are ready for the next step (Exploratory Data Analysis and Feature Engineering).

### **Step 3: Data Preparation & Exploratory Data Analysis**

Data was refined to ensure model readiness:

* **Cleaning:** Dropped irrelevant or empty features (Customer Email, Product Image, etc.).  
* **Feature Engineering:** Converted date strings into temporal features such as order\_month and order\_day\_of\_week to capture seasonality.  
* **One-Hot Encoding:** Categorical variables like Market, Shipping Mode, and Segment were encoded into numerical formats for algorithmic processing.  
* **Visual Analysis:** \* **Market Analysis:** Identified **LATAM** and **Europe** as the dominant markets by total sales.  
  * **Correlation:** Found a near-perfect correlation between Product Price and Sales, while Days for shipment (scheduled) showed a relationship with delivery outcomes.  
  * **Seasonality:** Observed fluctuations in sales volume across different months of the year.

### **Step 4: Modeling**

Multiple algorithms were deployed to address different business needs:

* **Sales Prediction:** A **Linear Regression** model was used to predict total sales.  
* **Delay Classification:** Both **Logistic Regression** and **K-Nearest Neighbors (KNN)** were trained to predict if a shipment would be late (Binary Classification: $1$ for risk, $0$ for no risk).  
* **Market Segmentation:** **K-Means Clustering** was used to group market regions based on their average sales, profit, and shipping efficiency.  
* **Inventory Analysis:** Frequency-based filtering was applied to identify the most popular products for stock recommendations.

### **Step 5: Evaluation**

The models were evaluated using industry-standard metrics:

* **Regression Performance:** The Sales Prediction model achieved an $R^{2}$ score of $0.9635$, indicating that $96.35\\%$ of the variance in sales is explained by the model features.  
* **Classification Performance:** \* **Logistic Regression:** Accuracy of $69.49\\%$.  
  * **KNN:** Accuracy of $66.48\\%$.  
  * Logistic Regression showed a better balance of precision and recall for predicting delays.  
* **Clustering Insights:** The K-Means algorithm successfully segmented the markets into three distinct tiers (High Volume, High Profit, and Moderate Performance).

### **Step 6: Deployment / Inference**

The project provides actionable insights for supply chain managers:

* **Strategic Stocking:** Top items should be prioritized for inventory (eg: Perfect Fitness Perfect Rip Deck,  Nike Men's Cleats, etc.,)    
* **Regional Strategy:** \* **Europe & LATAM:** Focus on expanding local fulfillment centers to handle high volume and reduce the 54% late delivery risk.  
  * **Pacific Asia:** High profit margins suggest this region is ideal for premium shipping services to maintain customer loyalty.  
* **Operational Planning:** Sales forecasting allows the finance department to predict monthly revenue with high confidence based on incoming order quantities and product mixes.

1. Image1: Logistic Regression : Confussion Matrix
![Logistic Regression - Confussion Matrix](https://github.com/prasy03/UCB-Capstone-Mar2026/blob/main/Final%20ver%20-%20Module%2024/images/confusion_matrix_logreg.png)

2. Image2: Optiman No of Clusters using Elbow method.
![Optimal Number of clusters](https://github.com/prasy03/UCB-Capstone-Mar2026/blob/main/Final%20ver%20-%20Module%2024/images/Optimal%20Number%20of%20Clusters%20-%20Elbow%20method.png)

3. Image3: Time Series : Baseline Model BEFORE hyperparameter tuning.
![TimeSeries-Baseline-beforeTuning](https://github.com/prasy03/UCB-Capstone-Mar2026/blob/main/Final%20ver%20-%20Module%2024/images/Baseline%20ModelComparison%20-%20before%20tuning%20.png)


5. Image4: Precision Recall Curve
![Precision-Recall-Curve](https://github.com/prasy03/UCB-Capstone-Mar2026/blob/main/Final%20ver%20-%20Module%2024/images/Precision-Recall_curve.png)

5. Image5: Grid Search CV
![Grid Search CV](https://github.com/prasy03/UCB-Capstone-Mar2026/blob/main/Final%20ver%20-%20Module%2024/images/grid_search_penalth.png)

6. Image6: ROC Curve
   
![ROC Curve](https://github.com/prasy03/UCB-Capstone-Mar2026/blob/main/Final%20ver%20-%20Module%2024/images/roc_curve_final%20copy.png)




## Results

The research successfully developed and evaluated Machine Learning models to optimize supply chain operations, yielding four core findings:

1. **Demand Forecasting:** A Linear Regression model achieved a high $R^{2}$ score of $0.9635$, accurately explaining $96.35%$ of total sales variance, enabling reliable revenue prediction.  
2. **Risk Analysis (Delay Prediction):** Logistic Regression proved the best classifier ($69.49%$ accuracy) for predicting the $54.6%$ of shipments flagged with a `Late_delivery_risk`.  
3. **Market Segmentation:** K-Means Clustering grouped markets into three strategic tiers: High Volume (e.g., Europe & LATAM), High Profit (e.g., Pacific Asia), and Moderate Performance.  
4. **Inventory Strategy:** Frequency analysis identified specific top-selling products (e.g., Perfect Fitness Perfect Rip Deck) for strategic stocking to prevent stockouts.  
5. **Time Series Model:** The analysis successfully transitioned from static supply chain records to a dynamic forecasting model. The Stationarity Test (ADF) confirmed that while sales are volatile, they can be modeled after differencing. The Auto-ARIMA model outperformed the baseline by identifying a specific weekly seasonality that was not immediately apparent in the raw data.


### Next steps  

* The model is highly sensitive to the 12-month window. While it captures monthly fluctuations well, its ability to predict long-term annual cycles is limited by the dataset's duration.  
* he data being synthetic behaves ideal for model conditions. Real world data may be different in its samples (like zero sales, promotion events, external influences, supply chain, data leakage etc).  
* The actionable recommendation is to focus operational expansion in High-Volume regions and invest in premium logistics for High-Profit regions.  
* **Feature Engineering:** Include external factors such as promotion periods or global shipping disruptions as external factors /  variables (SARIMAX).  
* **Advanced Models:**  Models like LSTM to better handle non-linear trends and complex seasonality.

### Outline for project   

The chosen dataset was selected after an extensive search across various platforms, including Kaggle, UC Irvine, datacatalog, and opendatabay.

1. Original source: DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS (year of publication: 2019\) source [link](https://data.mendeley.com/datasets/8gx2fvg2k6/5)  
   1. This dataset consists of 52 features (columns) and over 182K samples (rows). 

To maintain statistical accuracy and facilitate quicker model training with reduced complexity, the dataset was streamlined. This lean version focuses solely on 12 months of transaction data and is designed to meet the following core criteria:

1. **Operational supply chain data**: purchase orders, promised vs actual delivery dates, supplier lead times, shipment tracking data, and inventory levels.  
2. **Demand signals**: historical demand, forecast inputs, seasonality patterns,    
3. **External factors**: Shipping, source and destination region    
4. **Engineered features**.

------

## Acknowledgments / Credits 

**Data Provenance & Ethical AI Disclosure**

1. **Source Dataset:** The core empirical analysis was conducted using the DataCo Smart Supply Chain Dataset, a filtered 12-month longitudinal transaction log (Source: Constante et al., Mendeley Data).  
   2. **Computational Assistance:** LLMs like Gemini and ChatGPT were utilized as a High-Level Architectural Advisor. Specifically, the LLM assisted in:  
   * Defining the **dual-track model architecture** (Classification/Regression).  
   * Synthesizing **Feature Engineering strategies** for high-cardinality categorical variables.  
   * Structuring the **Data Pipeline** logic to ensure prevention of temporal data leakage.  
   3. **Human-in-the-Loop:** All final model selections, hyperparameter tuning, and business interpretations were performed and validated by the primary researcher to ensure domain-specific accuracy.




