# Install all necesary libs for data visualisation

# Install all necesary libs for data visualisation
# pip install matplotlib pandas

import pandas as pd
import requests
import matplotlib.pyplot as plt
from io import StringIO


# To be used for local acess to the repository 
file_path = "/mnt/c/Users/MR-BEST/flexisafe_internship_programme/beginner_phase/"
local_filename = "company_sales_data.csv"

# Load the data
url = "https://pynative.com/wp-content/uploads/2019/01/company_sales_data.csv"

try:
    # Send a GET request with a user-agent header
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an error for HTTP errors (like 403)

        # Write the content of the response to a local file
    with open(local_filename, "wb") as file:
        file.write(response.content)

    # Read the content into Pandas
    data = pd.read_csv(StringIO(response.text))
    print(data.head(3)) # Print first 3 rows
    
except requests.exceptions.RequestException as e:
    print(f"Error fetching the CSV: {e}")

    print(f"CSV file successfully downloaded and saved as '{local_filename}'")
except requests.exceptions.RequestException as e:
    print(f"Error downloading the file: {e}")

# Extract the 'month_number' and 'total_profit' columns
months = data['month_number']
total_profit = data['total_profit']

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(months, total_profit, marker='o', color='b', label='Total Profit')
plt.title("Company Total Profit Over Months")
plt.xlabel("Month Number")
plt.ylabel("Total Profit")
plt.grid(True)
plt.legend()
plt.show()

# Extract columns for Bathing Soap and Facewash
bathing_soap_sales = data['bathingsoap']
facewash_sales = data['facewash']

# Create subplots
plt.figure(figsize=(10, 8))

# Subplot 1: Bathing Soap Sales
plt.subplot(2, 1, 1)
plt.plot(months, bathing_soap_sales, marker='o', color='g', label='Bathing Soap Sales')
plt.title("Bathing Soap Sales")
plt.xlabel("Month Number")
plt.ylabel("Sales Units")
plt.grid(True)
plt.legend()

# Subplot 2: Facewash Sales
plt.subplot(2, 1, 2)
plt.plot(months, facewash_sales, marker='o', color='r', label='Facewash Sales')
plt.title("Facewash Sales")
plt.xlabel("Month Number")
plt.ylabel("Sales Units")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
