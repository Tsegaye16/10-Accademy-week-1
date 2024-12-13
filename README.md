## Project Overview

This project contains a Python script designed for analyzing and visualizing news article datasets. The script includes functions for data preprocessing, sentiment analysis, statistical computations, and graphical visualizations. It is modular, allowing users to load data, inspect it, preprocess it, and generate insights effectively.

---

## Features

### 1. **Data Inspection**

- **`LoadData(file_path)`**: Load the dataset from a specified file path.
- **`InspectData(dataFrame)`**: View the first few rows of the dataset.
- **`GetShape(df)`**: Get the shape (rows, columns) of the dataset.
- **`GetSummary(df)`**: Generate a statistical summary of numerical columns.
- **`GetColumns(df)`**: Display all column names.
- **`GetColumnNames(df)`**: Get the column names as a list.
- **`GetColumnTypes(df)`**: View the data types of columns.
- **`CheckMissingValue(df)`**: Check for missing values in the dataset.

### 2. **Data Preprocessing**

- **`RemoveColumn(df)`**: Remove unnecessary columns from the dataset.
- **`AddHeadlineLength(df)`**: Compute the length of article headlines.
- **`GtHeadlineLengthStats(df)`**: Get statistical insights about headline lengths.
- **`ConvertDdate(df, column_name='Date')`**: Convert date columns into a standard datetime format.
- **`ConvertDate(df)`**: Additional functionality to standardize date formats.
- **`ArticlePublishedYearly(df)`**: Extract yearly trends in article publication.

### 3. **Visualization**

- **`HeadlineLength(df)`**: Visualize headline length distribution.
- **`TopPublisher(df)`**: Identify and visualize top publishers.
- **`YearlyArticlePublished(df)`**: Plot the yearly trend of article publications.
- **`DailyArticlesPublishedEachMonth(df)`**: Visualize daily publication trends for each month.
- **`ThreeYearJanu(df)`**: Generate specific visualizations for January over three years.
- **`SentimentScore(df)`**: Compute sentiment scores for each article.
- **`SentimentClass(df)`**: Categorize articles by sentiment (e.g., Positive, Neutral, Negative).
- **`PlotDistributions(final_data)`**: Plot statistical distributions.
- **`PlotScatter(final_data)`**: Create scatter plots for data relationships.
- **`PlotHeatmap(final_data)`**: Generate heatmaps for correlation analysis.
- **`PlotScatterAndHeatmap(final_data)`**: Combine scatter and heatmap visualizations.

### 4. **Analysis and Technical Indicators**

- **`NumArticlePubTime(df)`**: Analyze article publication trends over time.
- **`CalculateTechnicalIndicators(df)`**: Compute advanced metrics or indicators (e.g., moving averages, volatility).
- **`PlotStockData(df)`**: Plot stock-related data trends based on the articles.

---

## Getting Started

### Prerequisites

- Required Libraries: pandas, matplotlib, seaborn, numpy

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Tsegaye16/Financial_News_and_Stock-Price_analaysis
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Import the necessary functions:
   ```python
   from script_name import LoadData, InspectData, AddHeadlineLength, ...
   ```
2. Load the dataset:
   ```python
   df = LoadData("path_to_your_dataset.csv")
   ```
3. Perform preprocessing and analysis:
   ```python
   InspectData(df)
   df = AddHeadlineLength(df)
   PlotScatter(df)
   ```

---

## Example Workflow

```python
# Step 1: Load the dataset
df = LoadData("articles.csv")

# Step 2: Inspect the dataset
InspectData(df)
print(GetShape(df))
print(CheckMissingValue(df))

# Step 3: Preprocess the data
df = AddHeadlineLength(df)
df = ConvertDdate(df, column_name='publication_date')

# Step 4: Analyze and visualize
TopPublisher(df)
YearlyArticlePublished(df)
PlotScatterAndHeatmap(df)
```

---

## Contributing

Feel free to fork the repository, submit issues, or make pull requests. Contributions are welcome!

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
