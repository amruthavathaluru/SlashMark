import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the Data
df = pd.read_csv('weather.csv')

# Step 2: Data Exploration
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# Step 3: Data Visualization
sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
plt.show()

# Step 4: Feature Engineering (if needed)
# Example: Adding a feature for the temperature range
df['TempRange'] = df['MaxTemp'] - df['MinTemp']

# Step 5: Data Analysis
# Check if 'Date' column exists
if 'Date' in df.columns:
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    monthly_avg_max_temp = df.groupby('Month')['MaxTemp'].mean()

    # Step 6: Data Visualization (Part 2)
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_avg_max_temp.index, monthly_avg_max_temp.values, marker='o')
    plt.xlabel('Month')
    plt.ylabel('Average Max Temperature')
    plt.title('Monthly Average Max Temperature')
    plt.grid(True)
    plt.savefig('monthly_avg_max_temp.png')
    plt.show()

    # Example: Identify the highest and lowest rainfall months
    monthly_rainfall = df.groupby('Month')['Rainfall'].sum()
    highest_rainfall_month = monthly_rainfall.idxmax()
    lowest_rainfall_month = monthly_rainfall.idxmin()
    print(f'\nHighest rainfall month: {highest_rainfall_month}, Lowest rainfall month: {lowest_rainfall_month}')
else:
    print("\nError: 'Date' column not found in the dataset.")
    # Handle the absence of 'Date' column appropriately
    monthly_avg_max_temp = None
    highest_rainfall_month = None
    lowest_rainfall_month = None

# Step 7: Advanced Analysis (e.g., predict Rainfall)
# Prepare the data for prediction
X = df[['MinTemp', 'MaxTemp', 'TempRange']]
y = df['Rainfall']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and calculate the Mean Squared Error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'\nMean Squared Error for Rainfall Prediction: {mse}')

# Step 8: Conclusions and Insights
summary = f"""
Data Analysis Summary:
1. Dataset Information:
{df.info()}

2. Statistical Summary:
{df.describe()}

3. Mean Squared Error for Rainfall Prediction: {mse}
"""

if monthly_avg_max_temp is not None:
    summary += f"\n4. Highest rainfall month: {highest_rainfall_month}\n5. Lowest rainfall month: {lowest_rainfall_month}\n"

future_work = """
Future Work:
1. Explore additional features to improve the rainfall prediction model.
2. Use more advanced algorithms like Random Forests or Neural Networks for prediction.
3. Analyze the impact of other weather parameters like humidity and wind speed on rainfall.
4. Visualize more complex relationships and patterns in the data.
5. Conduct time-series analysis for more accurate monthly predictions.
"""

# Combine summary and future work
complete_report = summary + "\n" + future_work
print(complete_report)

# Save the summary and future work to a text file
with open('data_analysis_summary.txt', 'w') as file:
    file.write(complete_report)

# Step 9: Communication
# Optionally save the visualizations to files
sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
plt.savefig('pairplot_weather_data.png')
plt.show()
