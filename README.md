# Weather Data Processing Project

This project involves crawling, cleaning, and visualizing weather data for Da Nang from 2015 to 2025.

## Project Structure

```
├── Crawl_data.ipynb      # Script for crawling weather data
├── Clean_data.ipynb      # Script for cleaning the crawled data
├── Show_data.ipynb       # Script for visualizing the cleaned data
├── requirements.txt      # Required Python packages
├── weather_data_danang_2015_2025.csv  # Processed data file
├── Raw_data/
│   └── raw_data.csv      # Raw data from web crawling
└── Clean_data/
    └── clean_data.csv    # Cleaned and processed data
```

## Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Chrome browser (for Selenium web crawling)

## Setup

1. Clone this repository or download all files to your local machine.
2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Project

### Step 1: Crawl Data

Run the `Crawl_data.ipynb` notebook to collect weather data from the web:

1. Open `Crawl_data.ipynb` in Jupyter Notebook/Lab
2. Execute all cells in the notebook sequentially
3. The crawled data will be saved to `Raw_data/raw_data.csv`

Note: This process may take some time depending on the date range being crawled.

### Step 2: Clean Data

Run the `Clean_data.ipynb` notebook to process and clean the raw data:

1. Open `Clean_data.ipynb` in Jupyter Notebook/Lab
2. Execute all cells in the notebook sequentially
3. The cleaned data will be saved to `Clean_data/clean_data.csv`

This step handles:

- Date and time formatting
- Missing value handling
- Data type conversions
- Column organization

### Step 3: Visualize Data

Run the `Show_data.ipynb` notebook to visualize and analyze the cleaned data:

1. Open `Show_data.ipynb` in Jupyter Notebook/Lab
2. Execute all cells in the notebook sequentially
3. View the generated visualizations including:
   - Distribution of numeric variables
   - Frequency of different weather conditions
   - Various weather statistics

## Note

The crawling process uses Selenium with Chrome WebDriver, which requires Chrome browser to be installed on your system. The WebDriver is automatically managed by the webdriver-manager package.
