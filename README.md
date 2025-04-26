
# Climate Change Analysis | Code@Trix

This project aims to provide an in-depth analysis of climate change trends over time, leveraging various climate data indicators and advanced statistical techniques to explore how climate drivers (such as temperature fluctuations and greenhouse gas concentrations) impact climate indicators (like sea-level rise, precipitation patterns, and global temperature anomalies). The project visualizes these data relationships and calculates a **Climate Change Intensity (CCI)** to assess the overall intensity of climate change over different periods.

## Project Overview

The main objective of this project is to explore the interplay between independent and dependent climate drivers, evaluate their correlation, and compute a composite **Climate Change Intensity (CCI)** to summarize the combined effects of various climate indicators over time.

This analysis involves:
- **Independent Climate Drivers**: These are factors such as temperature variations, greenhouse gas concentrations, and other environmental metrics that influence climate change.
- **Dependent Climate Indicators**: These include metrics like sea-level rise, changes in precipitation patterns, and global temperature anomalies, which are affected by the independent drivers.

By using data analysis techniques such as correlation and regression analysis, the project computes the Climate Change Intensity (CCI) for each year, providing a single value representing the intensity of climate change based on the drivers and indicators. The visualization of this data allows for better understanding of long-term climate trends.

## Features

- **Data Visualization**: Interactive graphs using Plotly to visualize both independent and dependent climate indicators over time.
- **Climate Change Intensity (CCI)**: Calculation of CCI based on a formula that combines independent climate drivers and their correlation with dependent climate indicators.
- **Regression Analysis**: Scatter plots with regression lines to visualize the relationship between individual independent drivers and dependent indicators.
- **Interactive UI**: Built using Streamlit for dynamic interaction with the data, allowing users to select year ranges, indicators, and visualize trends.

## Data

The project uses two CSV files:

- **independentData.csv**: Contains data on independent climate drivers (e.g., temperature, CO2 levels).
- **dependentData.csv**: Contains data on dependent climate indicators (e.g., sea level rise, temperature anomalies).

Both datasets are normalized and cleaned to remove missing values, ensuring consistent comparisons across variables.

### Data Preprocessing
The data is cleaned and preprocessed in the following ways:
1. **Normalization**: All climate data is normalized to a scale from 0 to 1, allowing for easy comparison across variables with different units.
2. **Year Selection**: The data can be filtered by a range of years using a slider in the user interface.
3. **Correlation**: A correlation matrix is calculated to understand the strength and direction of relationships between independent and dependent indicators.

## Technologies Used

- **Python**: The main programming language for data analysis and visualization.
- **Streamlit**: Framework for building the web interface, allowing users to interact with the data.
- **Plotly**: A visualization library used for interactive graphs and plots.
- **Pandas**: Data manipulation and analysis, especially for reading CSV files and processing the data.
- **Statsmodels**: Used for performing regression analysis to fit models between independent and dependent variables.
- **Numpy**: For handling numerical operations, such as calculating correlations and applying normalization.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/climate-change-analysis.git
   cd climate-change-analysis
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scriptsctivate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

   This will launch the app in your browser, where you can interact with the data.

## Usage

Once the app is running, you can interact with the data in the following ways:

1. **Select Year Range**: Use the slider to choose a specific range of years to analyze.
2. **View Climate Drivers**: The first graph will show the normalized values of independent climate drivers (e.g., temperature, CO2 levels) over the selected years.
3. **View Climate Indicators**: The second graph displays the normalized values of dependent climate indicators (e.g., sea level rise, temperature anomalies).
4. **Scatter Plot with Regression**: Select one independent climate driver from the dropdown to overlay a scatter plot of that driver against all dependent climate indicators, along with regression lines to visualize the relationship.
5. **View CCI**: The final graph shows the **Climate Change Intensity (CCI)** over time, calculated from the correlation and normalized values of the independent drivers.

### Example: Understanding Climate Change Intensity (CCI)

The CCI value for each year is computed by considering the correlation between each independent driver and dependent indicator. This is done using the formula:

$$
CCI_y = \sum_{i=1}^{n} ar{r}_i \cdot x_{i,y}
$$

Where:
- $$CCI_y$$ is the **Climate Change Intensity** in year $$y$$,
- $$n$$ is the number of independent climate drivers,
- $$ ar{r}_i $$ is the **average correlation** of the **i-th independent climate driver** with all dependent climate indicators,
- $$ x_{i,y} $$ is the **normalized value** of the **i-th independent climate driver** for year $$y$$.

The CCI allows us to understand the overall trend in climate change intensity by combining the effects of multiple independent drivers. A higher CCI indicates stronger or more intense climate change, while a lower CCI suggests weaker impacts.

## File Structure

```plaintext
climate-change-analysis/
│
├── app.py                # Main Streamlit application file
├── independentData.csv   # CSV file containing independent climate driver data
├── dependentData.csv     # CSV file containing dependent climate indicator data
├── requirements.txt      # List of required Python libraries
└── README.md             # Project documentation
```

## Future Improvements

- **Advanced Machine Learning Models**: Implement machine learning models to predict future climate change trends based on current data.
- **User Customization**: Allow users to upload their own climate data and perform customized analysis.
- **Geospatial Analysis**: Incorporate geographic data to analyze climate change on a regional or global scale.
- **Integration with Real-Time Data**: Link the project to live climate data sources to provide real-time climate change monitoring.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request. Please ensure that your contributions adhere to the project’s coding standards and include relevant tests where possible.

### Steps to contribute:
1. Fork the repository.
2. Clone your fork to your local machine.
3. Create a new branch for your feature or bug fix.
4. Commit your changes and push them to your fork.
5. Submit a pull request for review.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The data used in this project is sourced from publicly available climate datasets.
- Special thanks to the contributors and libraries that made this project possible: Streamlit, Plotly, Pandas, Numpy, Statsmodels, and all the open-source contributors.

## Contact

For any inquiries or questions, feel free to reach out at:

- Email: aditya.65sinha@gmail.com
