# Databricks notebook source
# DBTITLE 1,Load and Explore Data(Data Collection)

# Importing necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, log1p
from pyspark.ml import Pipeline


# Loading the dataset into a DataFrame
df = spark.read.table("default.electric_vehicle_population_dataset_3_csv")


# COMMAND ----------


# Display basic information about the DataFrame
print("Data Schema:")
df.printSchema()


# COMMAND ----------

# Dislaying the sample data 
print("Sample Data:")
df.show(5)

# COMMAND ----------

# Get the number of rows
num_rows = df.count()
print("Number of rows:", num_rows)


# COMMAND ----------

# Get the number of columns
print(df.columns)

# COMMAND ----------

# DBTITLE 1,Data Cleaning and Preprocessing

# Importing necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Create or get Spark session
spark = SparkSession.builder.appName("EV Data Cleaning").getOrCreate()

# Check for missing values in each column
missing_counts = df.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas()
print("Missing Value Counts in Each Column:")
print(missing_counts)

# Drop rows with missing values in specific columns (e.g., 'Electric Range')
df_cleaned = df.dropna(subset=['Electric Range'])

# Calculate the most frequent 'Make' for imputation
most_frequent_make = df_cleaned.groupBy('Make').count().orderBy('count', ascending=False).first()['Make']

# Impute missing values in 'Make' column with the most frequent value
df_cleaned = df_cleaned.fillna({'Make': most_frequent_make})




# COMMAND ----------


# Remove rows where any of these specific columns have missing values
columns_to_check = ['County', 'City', 'Postal Code', 'Electric Utility', '2020 Census Tract', 'Vehicle Location']
df_cleaned = df.na.drop(subset=columns_to_check)

# For Legislative District, considering dropping the column because it's not essential
df_cleaned = df_cleaned.drop('Legislative District')

# Display the DataFrame to check the result
df_cleaned.show(5)


# COMMAND ----------


# Importing necessary functions
from pyspark.sql.functions import when

# Replace placeholder zeros in 'Base MSRP' and 'Electric Range' with null values
df_cleaned = df.withColumn('Base MSRP', when(df['Base MSRP'] == 0, None).otherwise(df['Base MSRP']))
df_cleaned = df_cleaned.withColumn('Electric Range', when(df_cleaned['Electric Range'] == 0, None).otherwise(df_cleaned['Electric Range']))

# Display the schema of the DataFrame
df_cleaned.printSchema()


# COMMAND ----------

# Importing necessary functions
from pyspark.sql.functions import when

# Replace placeholder zeros in 'Base MSRP' and 'Electric Range' with null values
df_cleaned = df.withColumn('Base MSRP', when(df['Base MSRP'] == 0, None).otherwise(df['Base MSRP']))
df_cleaned = df_cleaned.withColumn('Electric Range', when(df_cleaned['Electric Range'] == 0, None).otherwise(df_cleaned['Electric Range']))


# COMMAND ----------

# Importing necessary functions
from pyspark.sql.functions import upper

# Standardize 'Make' and 'Model' columns by converting to uppercase
df_cleaned = df_cleaned.withColumn('Make', upper(df_cleaned['Make']))
df_cleaned = df_cleaned.withColumn('Model', upper(df_cleaned['Model']))

# Check unique values in 'Electric Vehicle Type'
df_cleaned.select('Electric Vehicle Type').distinct().show()
# Depending on the output, standardize the values if needed


# COMMAND ----------

# DBTITLE 1,EDA - Data Visualization
# MAGIC %md
# MAGIC

# COMMAND ----------

# DBTITLE 1,Registration trends
import matplotlib.pyplot as plt
import pandas as pd

# Sample data for illustration
dates = pd.date_range('2022-01-01', periods=12, freq='M')
registrations = [100, 120, 150, 200, 250, 300, 320, 350, 380, 400, 420, 450]

# Create a Pandas DataFrame
df = pd.DataFrame({'Date': dates, 'Registrations': registrations})

# Plotting the line plot
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Registrations'], marker='o')
plt.xlabel('Date')
plt.ylabel('Number of Registrations')
plt.title('EV Registrations Over Time')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# COMMAND ----------

# Descriptive statistics
df_cleaned.describe(['Electric Range', 'Base MSRP']).show()

# COMMAND ----------

# DBTITLE 1,Electric range and Base MSRP analysis

# Data visualization (requires Matplotlib and Pandas)
import matplotlib.pyplot as plt
import pandas as pd

# Convert Spark DataFrame to Pandas DataFrame for plotting
df_pd = df_cleaned.select(['Electric Range', 'Base MSRP']).toPandas()

# Histogram of Electric Range
plt.hist(df_pd['Electric Range'], bins=20)
plt.xlabel('Electric Range')
plt.ylabel('Frequency')
plt.title('Distribution of Electric Range')
plt.show()


# COMMAND ----------


import matplotlib.pyplot as plt
import pandas as pd

# Convert Spark DataFrame to Pandas DataFrame for plotting
df_pd = df_cleaned.select(['Electric Range', 'Base MSRP']).toPandas()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df_pd['Electric Range'], df_pd['Base MSRP'], alpha=0.5)
plt.title('Electric Range vs. Base MSRP')
plt.xlabel('Electric Range')
plt.ylabel('Base MSRP')
plt.grid(True)
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.sql.functions as F

# Remove rows with None values in 'Electric Range' column
df_cleaned = df_cleaned.filter(F.col('Electric Range').isNotNull())

# Box plot of Electric Range by Make
plt.figure(figsize=(12, 8))
sns.boxplot(x='Make', y='Electric Range', data=df_cleaned.toPandas())
plt.xlabel('Make')
plt.ylabel('Electric Range')
plt.title('Electric Range by Make')
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust plot layout for better spacing
plt.show()


# COMMAND ----------

# DBTITLE 1,Geographic variations and EV registrations

# Filter the DataFrame for BEVs and PHEVs
df_bev = df_cleaned.filter(df_cleaned['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)')
df_phev = df_cleaned.filter(df_cleaned['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)')

# Plotting separate scatter plots for BEVs and PHEVs
plt.figure(figsize=(10, 6))

# Scatter plot for BEVs
plt.scatter(df_bev.select('Electric Range').toPandas(), df_bev.select('Base MSRP').toPandas(), color='blue', alpha=0.5, label='BEV')

# Scatter plot for PHEVs
plt.scatter(df_phev.select('Electric Range').toPandas(), df_phev.select('Base MSRP').toPandas(), color='red', alpha=0.5, label='PHEV')

plt.title('Electric Range vs. Base MSRP (BEVs vs. PHEVs)')
plt.xlabel('Electric Range')
plt.ylabel('Base MSRP')
plt.legend()
plt.grid(True)
plt.show()


# COMMAND ----------


# Check the first few rows of df_cleaned to verify the column names and data
df_cleaned.show(5)

# Plotting using Seaborn
plt.figure(figsize=(12, 8))
sns.barplot(x='Make', y='Electric Range', data=df_cleaned.toPandas())  # Convert to Pandas DataFrame for plotting
plt.xlabel('Make')
plt.ylabel('Electric Range')
plt.title('Electric Range by Make')
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

# Split the "Vehicle Location" column into relevant components
from pyspark.sql.functions import split
split_col = split(df_cleaned['Vehicle Location'], ',')
df_cleaned = df_cleaned.withColumn('City', split_col.getItem(0))
df_cleaned = df_cleaned.withColumn('State', split_col.getItem(1))
df_cleaned = df_cleaned.withColumn('Country', split_col.getItem(2))

# Check the updated DataFrame schema
df_cleaned.printSchema()


# COMMAND ----------

# Split the "Vehicle Location" column into City, State, and Country
split_location = split(df_cleaned['Vehicle Location'], ',')
df_cleaned = df_cleaned.withColumn('City', split_location.getItem(0))
df_cleaned = df_cleaned.withColumn('State', split_location.getItem(1))
df_cleaned = df_cleaned.withColumn('Country', split_location.getItem(2))

# Check the updated DataFrame schema
df_cleaned.printSchema()


# COMMAND ----------

# DBTITLE 1,correlation analysis
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import corr

# Compute the correlation matrix
corr_matrix = df_cleaned.select([corr(c1, c2).alias(c1 + '_' + c2) for c1 in ['Electric Range', 'Base MSRP'] for c2 in ['Electric Range', 'Base MSRP']]).toPandas()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()


# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.sql.types import DoubleType

# Verify and convert data types if needed
df_cleaned = df_cleaned.withColumn('Model Year', col('Model Year').cast(DoubleType()))
df_cleaned = df_cleaned.withColumn('Legislative District', col('Legislative District').cast(DoubleType()))

# Check for missing values and handle them
df_cleaned = df_cleaned.dropna(subset=['Model Year', 'Legislative District'])

# Apply VectorAssembler after ensuring data compatibility
vector_assembler = VectorAssembler(inputCols=['Model Year', 'Legislative District'], outputCol='features')
df_vectorized = vector_assembler.transform(df_cleaned.select('Model Year', 'Legislative District'))

# Calculate correlation between Model Year and Legislative District
correlation_matrix = Correlation.corr(df_vectorized, 'features').head()
correlation_value = correlation_matrix[0].toArray()[0, 1]  # Assuming 2x2 matrix for correlation

print("Correlation between Model Year and Legislative District:")
print(correlation_value)


# COMMAND ----------

# Check Data Types
df_cleaned.printSchema()

# Handle Missing Values
from pyspark.sql.functions import col, isnan, when, count
df_cleaned.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_cleaned.columns]).show()

# Verify Column Names
df_cleaned.columns

# Inspect Data
df_cleaned.show(5)


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col

# Ensure no null values in the selected columns
df_cleaned = df_cleaned.dropna(subset=['Electric Range', 'Base MSRP', 'Model Year'])

# Check data types and convert if necessary
df_cleaned = df_cleaned.withColumn('Electric Range', col('Electric Range').cast('double'))
df_cleaned = df_cleaned.withColumn('Base MSRP', col('Base MSRP').cast('double'))
df_cleaned = df_cleaned.withColumn('Model Year', col('Model Year').cast('double'))

# VectorAssembler transformation
vector_assembler = VectorAssembler(inputCols=['Electric Range', 'Base MSRP', 'Model Year'], outputCol='features')
df_assembled = vector_assembler.transform(df_cleaned)

# Compute the correlation matrix
correlation_matrix = Correlation.corr(df_assembled, 'features').head()

# Extract the correlation matrix from the result
corr_matrix = correlation_matrix[0].toArray()

# Define the column names
column_names = ['Electric Range', 'Base MSRP', 'Model Year']

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=column_names, yticklabels=column_names)
plt.title('Correlation Heatmap')
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# Convert relevant columns to a vector for correlation analysis
vector_assembler = VectorAssembler(inputCols=['Model Year', 'Legislative District'], outputCol='features')
df_vectorized = vector_assembler.transform(df_cleaned)

# Compute the correlation matrix
correlation_matrix = Correlation.corr(df_vectorized, 'features').head()

# Extract the correlation matrix from the result
corr_matrix = correlation_matrix[0].toArray()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=['Model Year', 'Legislative District'], yticklabels=['Model Year', 'Legislative District'])
plt.title('Correlation Heatmap')
plt.show()


# COMMAND ----------

# DBTITLE 1,Trends and Insights
# MAGIC %md
# MAGIC

# COMMAND ----------

# Count unique values for Model Year, Model, and Legislative District
model_year_count = df_cleaned.select('Model Year').distinct().count()
model_count = df_cleaned.select('Model').distinct().count()
legislative_district_count = df_cleaned.select('Legislative District').distinct().count()

print(f"Unique Model Years: {model_year_count}")
print(f"Unique Models: {model_count}")
print(f"Unique Legislative Districts: {legislative_district_count}")


# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# Visualize distribution of Model Year
plt.figure(figsize=(10, 6))
df_cleaned.groupBy('Model Year').count().orderBy('Model Year').toPandas().plot(kind='bar', x='Model Year', y='count', title='Distribution of Model Year')
plt.xlabel('Model Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Visualize distribution of Model
plt.figure(figsize=(12, 6))
df_cleaned.groupBy('Model').count().orderBy('count', ascending=False).limit(10).toPandas().plot(kind='bar', x='Model', y='count', title='Top 10 Models by Count')
plt.xlabel('Model')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Visualize distribution of Legislative District
plt.figure(figsize=(12, 6))
df_cleaned.groupBy('Legislative District').count().orderBy('count', ascending=False).limit(10).toPandas().plot(kind='bar', x='Legislative District', y='count', title='Top 10 Legislative Districts by Count')
plt.xlabel('Legislative District')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

from pyspark.sql.functions import avg

# Fill null values in the 'STATE' column with 'WA'
df_cleaned_filled = df_cleaned.fillna({'STATE': 'WA'})

# Calculate average electric range by state
avg_range_by_state = df_cleaned_filled.groupBy('STATE').agg(avg('Electric Range').alias('Average Electric Range'))

# Show the results
avg_range_by_state.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have the data in a PySpark DataFrame named 'df_cleaned'
# Filter out rows with None values in the 'County' column
filtered_df = df_cleaned.filter(df_cleaned['County'].isNotNull())

# Calculate the average electric range by county
county_avg_range_df = (
    filtered_df.groupBy('County')
    .agg({'Electric Range': 'mean'})
    .withColumnRenamed('avg(Electric Range)', 'Average Electric Range')
    .toPandas()
)

# Sort the DataFrame by 'Average Electric Range' in descending order
sorted_df = county_avg_range_df.sort_values(by='Average Electric Range', ascending=False)

# Plotting the bar chart with increased width between bars
plt.figure(figsize=(18, 8))  # Increase the figure size to accommodate the wider bars
plt.bar(sorted_df['County'], sorted_df['Average Electric Range'], width=0.8)  # Adjust width as needed
plt.xlabel('County')
plt.ylabel('Average Electric Range')
plt.title('Average Electric Range by County')
plt.xticks(rotation=45, ha='right')  # Align x-axis labels to the right for better readability
plt.tight_layout()
plt.show()

# Print the top 5 counties with the highest and lowest average electric range
top5_highest = sorted_df.head(5)
top5_lowest = sorted_df.tail(5)

print("Top 5 counties with the highest average electric range:")
print(top5_highest)

print("\nTop 5 counties with the lowest average electric range:")
print(top5_lowest)


# COMMAND ----------

# Calculate average electric range by model year and make
avg_range_by_model_year_make = df_cleaned.groupBy('Model Year', 'Make').agg(F.avg('Electric Range').alias('Average Electric Range'))

# Show the results
avg_range_by_model_year_make.show()


# COMMAND ----------

# Calculate the count of electric vehicles registered in each city
ev_count_by_city = df_cleaned.groupBy('City').count().orderBy(F.desc('count'))

# Show the top cities with the highest EV adoption rates
top_cities_ev_adoption = ev_count_by_city.show(10)


# COMMAND ----------

# Calculate the count of electric vehicle types by country
ev_types_by_country = df_cleaned.groupBy('Country', 'Electric Vehicle Type').count().orderBy('Country')

# Show the results
ev_types_by_country.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import functions as F

# Assuming df_cleaned now has actual city names in the 'City' column
ev_count_by_city = df_cleaned.groupBy('City').count().orderBy(F.desc('count'))

# Get the top cities with the highest EV adoption rates
top_cities_ev_adoption = ev_count_by_city.limit(10).toPandas()

# Plotting the horizontal bar chart
plt.figure(figsize=(10, 8))
bars = plt.barh(top_cities_ev_adoption['City'], top_cities_ev_adoption['count'], color='skyblue')
plt.xlabel('Count of Electric Vehicles')
plt.ylabel('City')
plt.title('Top Cities with Highest EV Adoption Rates')
plt.tight_layout()

# Add data labels to the bars with horizontal alignment
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, 
             f'{bar.get_width():.0f}', ha='left', va='center')

plt.show()


# COMMAND ----------

# Display distinct values and their counts in the "City" column
city_counts = df_cleaned.groupBy('City').count().orderBy(F.desc('count'))
city_counts.show(50, truncate=False)


# COMMAND ----------

# DBTITLE 1,Model training


# Check for null values
null_counts = df_cleaned.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in df_cleaned.columns])
null_counts.show()

# Drop rows with null values in relevant columns
df_cleaned = df_cleaned.dropna(subset=['Electric Range', 'Model Year', 'Base MSRP'])

# Verify data types
df_cleaned.printSchema()

# Inspect data
df_cleaned.show(10)


# COMMAND ----------

# DBTITLE 1,Linear Regression Model for Base MSRP 
# Importing necessary libraries
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Assuming df_cleaned contains the preprocessed DataFrame

# Defining features and target variable
features = ['Electric Range', 'Model Year']
target = 'Base MSRP'

# Assembling features into a vector
assembler = VectorAssembler(inputCols=features, outputCol='features')
assembled_df = assembler.transform(df_cleaned)

# Splitting data into training and testing sets
train_data, test_data = assembled_df.randomSplit([0.8, 0.2], seed=42)

# Training the Linear Regression model
lr = LinearRegression(featuresCol='features', labelCol=target)
lr_model = lr.fit(train_data)

# Making predictions
predictions = lr_model.transform(test_data)

# Evaluating the model using RMSE
evaluator = RegressionEvaluator(labelCol=target, predictionCol='prediction', metricName='rmse')
lr_rmse = evaluator.evaluate(predictions)

# Displaying the RMSE
print("Root Mean Squared Error (RMSE):", lr_rmse)


# COMMAND ----------

# Displaying actual Base MSRP and predicted values from the model
predictions.select('Base MSRP', 'prediction').show(10)

# COMMAND ----------

# Checking for duplicate records
duplicate_count = assembled_df.count() - assembled_df.dropDuplicates().count()
print("Number of duplicate records:", duplicate_count)

# Removing duplicate records
assembled_df = assembled_df.dropDuplicates()


# COMMAND ----------

# DBTITLE 1,Random Forest Regression Model for Base MSRP
# Import necessary libraries
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, log1p

# Create a new feature by taking the logarithm of 'Electric Range'
assembled_df = assembled_df.withColumn('Log_Electric_Range', log1p(col('Electric Range')))

# Select features and target variable
features = ['Log_Electric_Range', 'Model Year']
target = 'Base MSRP'

# Assemble features into a vector with a different name
assembler = VectorAssembler(inputCols=features, outputCol='new_features')
assembled_df = assembler.transform(assembled_df)

# Split data into training and testing sets
train_data, test_data = assembled_df.randomSplit([0.8, 0.2], seed=42)

# Initialize Random Forest Regression model
rf = RandomForestRegressor(featuresCol='new_features', labelCol=target, numTrees=100, maxDepth=5)

# Train the Random Forest model
rf_model = rf.fit(train_data)

# Make predictions using the trained model
rf_predictions = rf_model.transform(test_data)

# Evaluate the model
rf_evaluator = RegressionEvaluator(labelCol=target, predictionCol='prediction', metricName='rmse')
rf_rmse = rf_evaluator.evaluate(rf_predictions)
print("Random Forest Regression RMSE:", rf_rmse)


# COMMAND ----------

# Import necessary libraries
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Assuming df_cleaned contains your preprocessed PySpark DataFrame
# Select features and target variable
features = ['Electric Range', 'Model Year']
target = 'Base MSRP'

# Assemble features into a vector
assembler = VectorAssembler(inputCols=features, outputCol='features')
assembled_df = assembler.transform(df_cleaned)

# Split data into training and testing sets
train_data, test_data = assembled_df.randomSplit([0.8, 0.2], seed=42)

# Initialize Gradient Boosting Regression model (GBTRegressor)
gbt = GBTRegressor(featuresCol='features', labelCol=target, maxIter=10)

# Train the Gradient Boosting model
gbt_model = gbt.fit(train_data)

# Make predictions using the trained model
gbt_predictions = gbt_model.transform(test_data)

# Evaluate the model using RMSE
evaluator = RegressionEvaluator(labelCol=target, predictionCol='prediction', metricName='rmse')
gbt_rmse = evaluator.evaluate(gbt_predictions)
print("Gradient Boosting Regression RMSE:", gbt_rmse)


# COMMAND ----------

# DBTITLE 1,Feature Engineering
# Import necessary libraries
from pyspark.ml.feature import PolynomialExpansion, VectorAssembler
from pyspark.sql.functions import col, month

# Interaction Terms
assembled_df = assembled_df.withColumn('Interaction_Term', col('Model Year') * col('Electric Range'))

# Polynomial Features
# Create a vector assembler to convert the column to a vector
vec_assembler = VectorAssembler(inputCols=['Model Year'], outputCol='Model_Year_Vector')
assembled_df = vec_assembler.transform(assembled_df)

poly_expander = PolynomialExpansion(degree=2, inputCol='Model_Year_Vector', outputCol='Model_Year_Polynomial')
assembled_df = poly_expander.transform(assembled_df)

# Temporal Features
assembled_df = assembled_df.withColumn('Month', month(col('Model Year').cast('timestamp')))

# Show the updated DataFrame schema
assembled_df.printSchema()


# COMMAND ----------

# DBTITLE 1,Advanced Model  Training
# Import necessary libraries
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Define features and target variable
features = ['Interaction_Term', 'Model_Year_Polynomial', 'Month']
target = 'Base MSRP'

# Split data into training and testing sets
train_data, test_data = assembled_df.randomSplit([0.8, 0.2], seed=42)

# Initialize Gradient Boosted Trees (GBT) regression model
gbt = GBTRegressor(featuresCol='features', labelCol=target)

# Train the GBT model
gbt_model = gbt.fit(train_data)

# Make predictions on the test data
gbt_predictions = gbt_model.transform(test_data)

# Evaluate the GBT model
evaluator = RegressionEvaluator(labelCol=target, predictionCol='prediction', metricName='rmse')
gbt_rmse_after = evaluator.evaluate(gbt_predictions)
print("Gradient Boosted Trees Regression RMSE:", gbt_rmse_after)


# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor

# Define the GBT model
gbt = GBTRegressor(featuresCol='features', labelCol='Base MSRP')

# Define the parameter grid for grid search
param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [5, 10, 15]) \
    .addGrid(gbt.maxIter, [10, 20, 30]) \
    .build()

# Define the evaluator
evaluator = RegressionEvaluator(labelCol='Base MSRP', predictionCol='prediction', metricName='rmse')

# Define the cross-validator
cross_validator = CrossValidator(estimator=gbt,
                                 estimatorParamMaps=param_grid,
                                 evaluator=evaluator,
                                 numFolds=3)  # You can adjust the number of folds

# Perform cross-validation and hyperparameter tuning
cv_model = cross_validator.fit(train_data)

# Make predictions on the test data
cv_predictions = cv_model.transform(test_data)

# Evaluate the tuned model
tuned_rmse = evaluator.evaluate(cv_predictions)

print("Tuned Gradient Boosted Trees Regression RMSE:", tuned_rmse)


# COMMAND ----------

# DBTITLE 1,Model Evaluation and  Validation
# Check RMSE values for each model
print("Linear Regression RMSE:", lr_rmse)
print("Random Forest Regression RMSE:", rf_rmse)
print("Gradient Boosted Trees RMSE:", gbt_rmse_after)
print("Tuned Gradient Boosted Trees RMSE:", tuned_rmse)

# Create a DataFrame to store RMSE values
data = {
    'Model': ['Linear Regression', 'Random Forest Regression', 'Gradient Boosted Trees', 'Tuned Gradient Boosted Trees'],
    'RMSE': [lr_rmse, rf_rmse, gbt_rmse_after, tuned_rmse]
}
rmse_summary = pd.DataFrame(data)

# Display the summary table
print(rmse_summary)


# COMMAND ----------

# DBTITLE 1,Insights 
# Feature Importance Analysis
# Random Forest Regression
rf_feature_importance = rf_model.featureImportances.toArray()
# Gradient Boosted Trees
gbt_feature_importance = gbt_model.featureImportances.toArray()

# Plotting
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.barh(range(len(rf_feature_importance)), rf_feature_importance, align='center')
plt.yticks(range(len(rf_feature_importance)), ['Model Year', 'Electric Range'])
plt.xlabel('Feature Importance')
plt.title('Random Forest')

plt.subplot(1, 2, 2)
plt.barh(range(len(gbt_feature_importance)), gbt_feature_importance, align='center')
plt.yticks(range(len(gbt_feature_importance)), ['Model Year', 'Electric Range'])
plt.xlabel('Feature Importance')
plt.title('Gradient Boosted Trees')

plt.tight_layout()
plt.show()

# Conclusion
"""
# Insights:
- Both Random Forest and Gradient Boosted Trees models indicate that 'Electric Range' is the most influential feature on Base MSRP, underscoring its critical role in electric vehicle pricing.
- Although Random Forest slightly outperforms Gradient Boosted Trees in terms of RMSE, both models exhibit comparable feature importance patterns.
- Further segmentation and detailed error analysis may reveal opportunities for model refinement and practical implementation in policy-making, infrastructure planning, and consumer awareness initiatives.
"""


# COMMAND ----------


