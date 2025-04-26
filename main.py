import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import statsmodels.api as sm

st.set_page_config("Climate Change Analysis | Code@Trix", layout="wide")

def make_column_names_unique(columns):
    seen = {}
    unique_cols = []
    for col in columns:
        if col not in seen:
            seen[col] = 1
            unique_cols.append(col)
        else:
            seen[col] += 1
            unique_cols.append(f"{col}.{seen[col]}")
    return unique_cols

@st.cache_data
def load_normalized_data(file, top):
    df = pd.read_csv(file).replace('..', pd.NA)
    df.drop(columns=[col for col in ['Series Code', 'Country Name', 'Country Code'] if col in df.columns], inplace=True, errors='ignore')
    df.set_index('Series Name', inplace=True)
    df = df.head(top).transpose()
    df.index = df.index.str.extract(r'(\d{4})')[0].astype(int)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.columns = make_column_names_unique(df.columns)
    return (df - df.min()) / (df.max() - df.min())

ind_df = load_normalized_data("independentData.csv", 18)
dep_df = load_normalized_data("dependentData.csv", 16)

st.title("Climate Change Analysis | Code@Trix")
min_year, max_year = int(ind_df.index.min()), int(ind_df.index.max())
year_range = st.slider("Select Year Range:", min_value=min_year, max_value=max_year, value=(min_year, max_year))
ind_df = ind_df.loc[year_range[0]:year_range[1]]
dep_df = dep_df.loc[year_range[0]:year_range[1]]

def plot_df(df, title, colorScheme):
    fig = go.Figure([
        go.Scatter(
            x=df.index,
            y=df[col],
            mode='lines+markers',
            name=(str(col)[:30] + "...") if len(str(col)) > 33 else str(col),
            text=[col] * len(df),
            hovertemplate="<b>%{text}</b><br>Year: %{x}<br>Normalized Value: %{y:.2f}<extra></extra>",
            opacity=0.6,
            line=dict(color=colorScheme[i % 8], width=4)
        ) for i, col in enumerate(df.columns)
    ])
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Normalized Value (0–1)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=-0.25, xanchor='center', x=0.5, font=dict(size=20), bgcolor='rgba(255,255,255,0)'),
        margin=dict(l=40, r=40, t=40, b=140),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Independent Climate Drivers")
plot_df(ind_df, "Normalized Climate Drivers | Independent Factors", px.colors.qualitative.T10)

st.markdown("""
### Analysis of Independent Climate Drivers

The graph above presents a visual representation of the normalized independent climate drivers over the selected time range. These drivers include factors such as temperature fluctuations, greenhouse gas concentrations, and other environmental metrics that have been recognized as key indicators of climate change. In the context of this study, independent drivers refer to those variables that are hypothesized to influence climate change, but are not directly dependent on other climate indicators. These variables provide insight into the environmental processes driving the observed trends.
Normalization ensures that these variables are comparable on a uniform scale (0 to 1), allowing for easier comparison across different factors. This normalization approach eliminates the potential biases caused by differences in the units of measurement, ensuring that all variables are given equal weight in the analysis. The year-on-year variations displayed in the graph provide a clear picture of the fluctuations in the independent climate drivers over time, offering valuable insights into how each factor contributes to long-term climate trends.
""")

st.subheader("Dependent Climate Indicators")
plot_df(dep_df, "Normalized Climate Indicators | Dependent Factors", px.colors.qualitative.G10)

st.markdown("""
### Analysis of Dependent Climate Indicators

The second graph displays the normalized dependent climate indicators, which include metrics such as sea-level rise, changes in precipitation patterns, and global temperature anomalies. These indicators represent the observable outcomes or consequences that are directly influenced by the independent climate drivers discussed earlier. Dependent indicators are typically used to measure the effects of climate change on various ecological, economic, and social systems. By examining how these indicators evolve over time, we can assess the impact of climate change on global systems and make predictions about future trends. As with the independent indicators, normalization allows for direct comparisons between different dependent variables, ensuring consistency in how the data is presented. It also highlights the relative magnitude of change across different indicators, which may have varying degrees of sensitivity to climate change.
""")

st.subheader("Overlayed Scatter Plot with Regression Lines: One Independent vs All Dependents")

selected_ind = st.selectbox("Select One Independent Indicator", ind_df.columns)

overlay_fig = go.Figure()
colors = px.colors.qualitative.Dark24  # Up to 24 distinct colors

for i, dep_col in enumerate(dep_df.columns):
    scatter_data = pd.DataFrame({
        "Independent": ind_df[selected_ind],
        "Dependent": dep_df[dep_col]
    }).dropna()

    if scatter_data.empty:
        continue

    corr = scatter_data["Independent"].corr(scatter_data["Dependent"])

    X = sm.add_constant(scatter_data["Independent"])  # Add intercept term
    model = sm.OLS(scatter_data["Dependent"], X).fit()

    slope = model.params[1]
    intercept = model.params[0]

    scatter_fig = px.scatter(
        scatter_data,
        x="Independent",
        y="Dependent",
        trendline="ols",  # Adding OLS (regression line)
        color_discrete_sequence=[colors[i % len(colors)]],
        title=f"{selected_ind} vs {dep_col} | Correlation: {corr:.2f}",
        labels={"Independent": selected_ind, "Dependent": dep_col},
        opacity=0.7
    )

    overlay_fig.add_trace(scatter_fig.data[0])  # The scatter trace (points)
    overlay_fig.add_trace(scatter_fig.data[1])  # The regression line trace

    overlay_fig.data[-1].update(
        hovertemplate=f"<b>{dep_col} Regression Line</b><br>"
                      f"Slope (Coefficient): {slope:.2f}<br>"
                      f"Intercept: {intercept:.2f}<br>"
                      f"Correlation: {corr:.2f}<br>"
                      f"X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
        name=f"Regression Line - {dep_col}"  # Name for the regression line in the legend
    )

    overlay_fig.data[-2].update(name=f"Data Points - {dep_col}")  # Update the name for the scatter points

overlay_fig.update_layout(
    title=f"Overlayed Scatter Plot of '{selected_ind}' vs All Dependent Indicators with Regression Lines",
    xaxis_title=selected_ind,
    yaxis_title="Dependent Indicators",
    legend=dict(
        orientation="v",
        font=dict(size=12),
        title="Legend",
        traceorder="normal",  # Ensure that the legend follows the plot trace order
    ),
    height=700,
    margin=dict(l=40, r=40, t=60, b=60)
)

st.plotly_chart(overlay_fig, use_container_width=True)

st.markdown("""
### Regression Analysis: Independent vs Dependent Climate Indicators

The overlay scatter plot above illustrates the relationship between one selected independent climate indicator and all dependent indicators, with linear regression lines fitted to each pair. The correlation coefficient and the slope of the regression line are included to quantify the strength and direction of the relationship.

This analysis allows us to explore how changes in a specific independent variable (e.g., temperature) correlate with variations in various dependent indicators such as sea-level rise or precipitation patterns. The slope of the regression line indicates the degree to which changes in the independent indicator affect the dependent variable. A steeper slope suggests a stronger relationship, while a flatter slope indicates a weaker one.

The inclusion of the regression line helps us understand the linearity of these relationships, and the correlation coefficient quantifies the degree of linear association between the two variables. This provides a statistical basis for understanding how well the independent factor can explain the changes in the dependent indicators.
""")

st.markdown(r"""
### Derivation of the Climate Change Intensity (CCI) Formula

The Climate Change Intensity (CCI) formula is a composite measure that combines the influence of multiple independent climate drivers on various dependent climate indicators. Here’s how the formula is derived:

#### Step 1: Define the Variables
- Let $$x_{i,y}$$ represent the normalized value of the **i-th independent climate driver** for the **y-th year**.
- Let $$ r_{i,j} $$ represent the **correlation coefficient** between the **i-th independent climate driver** and the **j-th dependent climate indicator**.
- Let $$ \bar{r}_i $$ represent the **average correlation** of the **i-th independent climate driver** with all the dependent climate indicators.

Each independent variable may influence the dependent variables to a different extent, and the correlation tells us how strong that effect is.

#### Step 2: Average Correlation for Each Independent Climate Driver
For each independent driver $$i$$, we calculate its correlation with all dependent climate indicators. This gives us a measure of how strongly the independent driver is associated with the dependent indicators.

The **average correlation** $$ \bar{r}_i $$ for the **i-th independent driver** is calculated as:

$$
\bar{r}_i = \frac{1}{m} \sum_{j=1}^{m} r_{i,j}
$$

Where:
- $$m$$ is the number of dependent climate indicators,
- $$r_{i,j}$$ is the correlation between the **i-th independent driver** and the **j-th dependent indicator**.

The average correlation $$ \bar{r}_i $$ reflects the typical strength of the relationship between the independent climate driver and all dependent climate indicators.

#### Step 3: Climate Change Intensity (CCI) Formula
Now, for each year $$y$$, we calculate the contribution of each independent climate driver to the overall climate change intensity.

The **Climate Change Intensity (CCI)** for year $$y$$ is given by the weighted sum of the normalized values of all the independent climate drivers. The weights are the average correlations of each driver with all dependent climate indicators.

$$
CCI_y = \sum_{i=1}^{n} \bar{r}_i \cdot x_{i,y}
$$

Where:
- $$CCI_y$$ is the **Climate Change Intensity** in year $$y$$,
- $$n$$ is the number of independent climate drivers,
- $$ \bar{r}_i $$ is the **average correlation** of the **i-th independent climate driver** with all dependent climate indicators,
- $$ x_{i,y} $$ is the **normalized value** of the **i-th independent climate driver** for year $$y$$.

#### Step 4: Explanation of the Formula
- The **normalized value** $$ x_{i,y} $$ represents the magnitude of the **i-th independent climate driver** in year $$y$$, scaled between 0 and 1.
- The **average correlation** $$ \bar{r}_i $$ acts as a weight, indicating how strongly the **i-th climate driver** is related to the dependent climate indicators.
- The sum across all independent drivers $$i$$ gives the overall intensity $$CCI_y$$ for that year.

#### Step 5: Interpretation of CCI
The resulting **CCI** for each year provides a single value representing the overall **intensity of climate change**. A higher CCI value indicates stronger or more intense climate change, while a lower CCI suggests weaker climate change effects for that year.

---
This formula combines the effects of multiple climate drivers to track the combined intensity of climate change over time.
""")

# Step 1: Calculate average correlation per independent indicator
avg_corr_dict = {}

for ind_col in ind_df.columns:
    corrs = []
    for dep_col in dep_df.columns:
        pair_data = pd.DataFrame({
            "ind": ind_df[ind_col],
            "dep": dep_df[dep_col]
        }).dropna()

        if not pair_data.empty:
            corrs.append(pair_data["ind"].corr(pair_data["dep"]))

    if corrs:
        avg_corr_dict[ind_col] = np.mean(corrs)
    else:
        avg_corr_dict[ind_col] = 0.0  # fallback if all were NaN

# Step 2: Compute CCI for each year using the formula
cci_series = pd.Series(index=ind_df.index, dtype=float)

for year in ind_df.index:
    year_data = ind_df.loc[year]
    cci_value = 0.0
    for ind in ind_df.columns:
        if pd.notna(year_data[ind]):
            cci_value += avg_corr_dict[ind] * year_data[ind]
    cci_series.loc[year] = cci_value

# Step 3: Plot CCI over time
cci_fig = px.line(
    cci_series,
    title="Climate Change Intensity (CCI) Over Time",
    labels={"value": "CCI", "intensity": "Year"},
    markers=True
)

cci_fig.update_traces(line=dict(width=4), marker=dict(size=6))
cci_fig.update_layout(
    yaxis_title="CCI",
    xaxis_title="Year",
    hovermode="x unified",
    height=500
)

st.plotly_chart(cci_fig, use_container_width=True)

st.markdown("""
### Climate Change Index (CCI) Over Time

The final graph displays the calculated Climate Change Intensity (CCI) over time, based on the formula presented earlier. The CCI provides an aggregated view of the overall climate change trend, taking into account the combined effects of various independent and dependent climate factors.

By examining the CCI over time, we can observe how the intensity of climate change has evolved, and potentially correlate it with major global events or policy changes. This comprehensive index is a useful tool for understanding the broader context of climate change dynamics, integrating multiple environmental drivers and their impacts on the planet.
""")
