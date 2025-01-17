'''
ALTERNATIVE Implementation of Task 2 with Seaborn incorporated for better visualisation¶
Task 1: Tic-Tac-Toe AI Agent
Since Seaborn is not directly relevant for the Tic-Tac-Toe task, we'll focus only on Task 2.

Task 2: Seaborn for Data Visualization
Using Seaborn, we will:

Create a line plot for the total profit of all months.
Create subplots for bathing soap and facewash sales. Here’s the code to implement these tasks:

'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests  # Import requests for handling HTTP requests
from io import StringIO  # Required to handle raw CSV data


### URL to the dataset

# To be used for local acess to the repository 
file_path = "/mnt/c/Users/MR-BEST/flexisafe_internship_programme/beginner_phase/"
local_filename = "company_sales_data.csv"

# URL to the dataset
url = "https://pynative.com/wp-content/uploads/2019/01/company_sales_data.csv"

# Fetch the CSV data using requests
headers = {"User-Agent": "Mozilla/5.0"}
try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
    csv_data = StringIO(response.text)  # Convert response content into a file-like object
    data = pd.read_csv(csv_data)  # Load the dataset into a Pandas DataFrame
except requests.exceptions.RequestException as e:
    print(f"Error fetching the CSV: {e}")
    exit()

# Display the first few rows of the dataset
print(data.head())

# # Load the data
# url = "https://pynative.com/wp-content/uploads/2019/01/company_sales_data.csv"

# try:
#     # Send a GET request with a user-agent header
#     headers = {"User-Agent": "Mozilla/5.0"}
#     response = requests.get(url, headers=headers)
#     response.raise_for_status()  # Raise an error for HTTP errors (like 403)

#         # Write the content of the response to a local file
#     with open(local_filename, "wb") as file:
#         file.write(response.content)

#     # Read the content into Pandas
#     data = pd.read_csv(StringIO(response.text))
#     print(data.head(3)) # Print first 3 rows
    
# except requests.exceptions.RequestException as e:
#     print(f"Error fetching the CSV: {e}")

#     print(f"CSV file successfully downloaded and saved as '{local_filename}'")
# except requests.exceptions.RequestException as e:
#     print(f"Error downloading the file: {e}")

##

# # Alternate URL to the dataset
# url = "https://pynative.com/wp-content/uploads/2019/01/company_sales_data.csv"

# # Load the data
# data = pd.read_csv(url)

# # Display the first few rows of the dataset
# print(data.head())

# Set the Seaborn style
sns.set_theme(style="whitegrid")

# Task 2a: Line plot for total profit of all months
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='month_number', y='total_profit', marker='o', color='blue', label='Total Profit')
plt.title('Total Profit by Month', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Profit', fontsize=12)
plt.legend()
plt.show()

# Task 2b: Subplots for bathing soap and facewash sales
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Bathing soap sales
sns.lineplot(ax=axes[0], data=data, x='month_number', y='bathingsoap', marker='o', color='green', label='Bathing Soap')
axes[0].set_title('Bathing Soap Sales', fontsize=12)
axes[0].set_xlabel('Month', fontsize=10)
axes[0].set_ylabel('Sales', fontsize=10)

# Subplot 2: Facewash sales
sns.lineplot(ax=axes[1], data=data, x='month_number', y='facewash', marker='o', color='orange', label='Facewash')
axes[1].set_title('Facewash Sales', fontsize=12)
axes[1].set_xlabel('Month', fontsize=10)
axes[1].set_ylabel('Sales', fontsize=10)

# Display the subplots
plt.tight_layout()
plt.show()
