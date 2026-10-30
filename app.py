import os
import re
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px

from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="FoodTime | Delivery Estimation",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("style.css", "r", encoding="utf-8") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

with st.sidebar:
    
    menu = option_menu(
        menu_title="Food Delivery",
        options=[
            "Home",
            "Predict Delivery Time",
            "Data Analysis",
            "Model Performance",
            "Feature Importance",
            "About Project"
        ],
        icons=[
            "house",
            "clock",    
            "bar-chart",
            "speedometer2",
            "graph-up",
            "info-circle"
        ],
        menu_icon="truck",
        default_index=0,
        styles={
            "container": {
                "padding": "5px",
                "background-color": "#F1E7D9"
            },
            "icon": {
                "color": "#C9825A",
                "font-size": "18px"
            },
            "menu-title": {
                "color": "#4B443D",
                "font-size": "20px",
                "font-weight": "700"
            },
            "menu-icon": {
                "color": "#C9825A",
                "font-size": "22px"
            },
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "5px 0",
                "padding": "12px",
                "border-radius": "10px",
                "color": "#4B443D"
            },
            "nav-link-selected": {
                "background-color": "#C9825A",
                "font-weight": "600",
                "color": "#4B443D"
            }
        }
    )

    st.html("""
    <div style="
        margin-top:35px;
        padding:15px 5px;
        border-top:1px solid #B8AA99;
        color:#65594F;
        font-size:12px;
        text-align:center;">
        Machine Learning Project<br>
        Food Delivery Time Estimation
    </div>
    """)

if menu == "Home":

    st.html("""
    <div class="hero">
        <div class="hero-tag">SMART FOOD DELIVERY</div>

        <div class="hero-title">
            Smarter Timing. <br>
            Better Deliveries.<br>
        </div>

        <div class="hero-description">
            Estimate food delivery time using machine learning.
            Understand how traffic, weather, vehicle conditions
            and other delivery factors influence delivery duration.
        </div>
    </div>
    """)

    st.write("")

    left, right = st.columns([1.1, 0.9], gap="medium")

    with left:
        st.html("""
        <div class="welcome-panel">
            <div class="section-title">
                Welcome to <span style="color:#C9825A;">FoodTime</span>
            </div>

            <div class="welcome-text">
                FoodTime is a machine learning application that estimates
                how long a food order may take to arrive.<br><br>
                Explore delivery data, understand important features,
                and discover how a regression model predicts delivery time.
            </div>
        </div>
        """)

    with right:
        st.html("""
        <div class="benefits">
            <div class="benefit">
                <div class="benefit-icon">◎</div>
                Accurate<br>Prediction
            </div>
            <div class="benefit">
                <div class="benefit-icon">ϟ</div>
                Delivery<br>Factors
            </div>
            <div class="benefit">
                <div class="benefit-icon">✓</div>
                Better<br>Insights
            </div>
        </div>
        """)

    st.write("")
 
    st.html("""
    <div class="section-title">Explore Your Platform</div>
    <div class="section-subtitle">
        Discover the tools available in your delivery estimation system.
    </div>

    <div class="feature-grid">

        <div class="feature-card orange">
            <div class="feature-number">01</div>
            <div class="feature-heading">Delivery Prediction</div>
            <div class="feature-description">
                Estimate delivery duration using your trained
                regression model.
            </div>
        </div>

        <div class="feature-card green">
            <div class="feature-number">02</div>
            <div class="feature-heading">Data Analysis</div>
            <div class="feature-description">
                Explore delivery patterns, distributions and
                relationships between important variables.
            </div>
        </div>

        <div class="feature-card purple">
            <div class="feature-number">03</div>
            <div class="feature-heading">Model Insights</div>
            <div class="feature-description">
                Understand model performance and the features
                that influence delivery time estimates.
            </div>
        </div>

    </div>
    """)

    st.write("")

    st.html("""
    <div class="overview-panel">

        <div class="overview-heading">Project Overview</div>

        <div class="overview-description">
            A data-driven approach to understanding and estimating
            food delivery duration.
        </div>

        <div class="overview-grid">

            <div class="overview-item">
                <div class="overview-icon">◈</div>
                <div class="overview-label">PROJECT TYPE</div>
                <div class="overview-value">Regression</div>
            </div>

            <div class="overview-item">
                <div class="overview-icon">⚙</div>
                <div class="overview-label">CORE TECHNOLOGY</div>
                <div class="overview-value">Machine Learning</div>
            </div>

            <div class="overview-item">
                <div class="overview-icon">◷</div>
                <div class="overview-label">PREDICTION OUTPUT</div>
                <div class="overview-value">Time in Minutes</div>
            </div>

        </div>
    </div>

    <div class="footer">
        <b>FoodTime | Food Delivery Time Estimation</b><br>
        Built with Python, Streamlit and Machine Learning
    </div>
    """)

elif menu == "Predict Delivery Time":
    
    model_path = "pickle_file/food_delivery_xgboost.pkl"
    data_path = "dataset/foodtimedataset.csv"

    st.html("""
    
    <div class="predict-hero">
        <div class="predict-tag">SMART DELIVERY PREDICTION</div>
        <div class="predict-heading">
            Estimate Your <span style="color:#C9825A;">Delivery Time.</span>
        </div>
        <div class="predict-description">
            Enter the delivery details below to estimate how long
            your food order may take using a trained machine learning model.
        </div>
    </div>
    """)

    if not os.path.exists(model_path):
        st.error("Model file not found: " + model_path)

    elif not os.path.exists(data_path):
        st.error("Dataset file not found: " + data_path)

    else:
        model = joblib.load(model_path)
        df = pd.read_csv(data_path)

        target = "Delivery Time"

        if target not in df.columns:
            st.error("The dataset does not contain the 'Delivery Time' target column.")

        else:
            X = df.drop(columns=[target, "ID", "Delivery Person ID"], errors="ignore")

            st.html("""
            <div class="form-heading">Delivery Details</div>
            <div class="form-description">
                Complete the fields to calculate your estimated delivery time.
            </div>
            """)
            
            input_data = {}

            with st.form("delivery_prediction_form"):

                columns = st.columns(2)

                for i, col in enumerate(X.columns):
                    with columns[i % 2]:
                        if pd.api.types.is_numeric_dtype(X[col]):
                            value = X[col].median()
                            value = 0 if pd.isna(value) else float(value)

                            input_data[col] = st.number_input(
                                col,
                                value=value
                            )

                        else:
                           if col == "Vehicle Type":
                             X[col] = X[col].str.strip().str.title()

                           choices = sorted(X[col].dropna().unique().tolist())

                           if len(choices) > 0:
                             input_data[col] = st.selectbox(
                               col,
                               choices
                                )
                           else:
                                 input_data[col] = st.text_input(col)

                submitted = st.form_submit_button(
                    "Estimate Delivery Time",
                    type="primary",
                    use_container_width=True
                )

            if submitted:
                input_df = pd.DataFrame([input_data], columns=X.columns)

                try:
                    prediction = model.predict(input_df)[0]

                    result_col, model_col = st.columns([1.4, 1], gap="medium")

                    with result_col:
                        st.html(f"""
                        <div class="result-card">
                            <div class="result-label">ESTIMATED DELIVERY TIME</div>
                            <div class="result-value">{prediction:.1f} minutes</div>
                            <div style="color:#5E554B;font-size:14px;margin-top:8px;">
                                This is a model-based estimate. Actual delivery time may vary.
                            </div>
                        </div>
                        """)

                    with model_col:
                        model_name = type(model).__name__
                        if hasattr(model, "steps") and model.steps:
                            model_name = type(model.steps[-1][1]).__name__

                        st.html(f"""
                        <div class="model-info-card">
                            <div class="model-info-title">Model Information</div>
                            <div class="model-info-label">Algorithm</div>
                            <div class="model-info-value">{model_name}</div>
                            <div class="model-info-label">Problem Type</div>
                            <div class="model-info-value">Regression</div>
                            <div class="model-info-label">Input Features</div>
                            <div class="model-info-value">{len(X.columns)} features</div>
                            <div class="model-info-label">Prediction Unit</div>
                            <div class="model-info-value">Minutes</div>
                        </div>
                        """)

                except Exception as e:
                    st.error("Prediction failed. Check that the model uses the same input features and column names as the dataset.")
                    st.caption(str(e))
 
elif menu == "Data Analysis":
    
    st.html("""
        <div class="predict-hero">
            <div class="predict-tag">DATA INSIGHTS</div>
            <div class="predict-heading">
                Explore Your <span style="color:#C9825A;">Delivery Data.</span>
            </div>
            <div class="predict-description">
                Discover patterns, compare delivery factors, and explore
                your dataset through interactive charts and statistics.
            </div>
        </div>
    """)

    st.markdown(
        '<div class="analysis-title">Data Analysis</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="analysis-subtitle">'
        'Explore your food delivery dataset through interactive charts, '
        'statistics, and data summaries.'
        '</div>',
        unsafe_allow_html=True
    )

  
    df = pd.read_csv("dataset/foodtimedataset.csv")
    df["Vehicle Type"] = df["Vehicle Type"].str.strip().str.title()

    st.markdown(
        '<div class="section-title">Dataset Overview</div>',
        unsafe_allow_html=True
    )

    total_records = len(df)
    total_columns = len(df.columns)
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Total Columns", total_columns)
    col3.metric("Missing Values", f"{missing_values:,}")
    col4.metric("Duplicate Rows", f"{duplicate_rows:,}")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribution",
        "Relationship",
        "Category Analysis",
        "Dataset Details"
    ])

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    category_analysis_cols = []
    for column in categorical_cols:
        unique_count = df[column].nunique(dropna=True)
        if 2 <= unique_count <= 30:
            category_analysis_cols.append(column)

    with tab1:

        st.subheader("Feature Distribution")
        st.caption("Select a numeric feature to understand its distribution.")

        if numeric_cols:

            selected_column = st.selectbox(
                "Choose Numeric Column",
                numeric_cols,
                key="distribution_column"
            )

            graph_type = st.selectbox(
                "Choose Distribution Graph",
                ["Histogram", "Box Plot", "Violin Plot"],
                key="distribution_graph"
            )

            if graph_type == "Histogram":
                fig = px.histogram(
                    df,
                    x=selected_column,
                    title=f"Distribution of {selected_column}",
                    color_discrete_sequence=["#C9825A"],
                    template="plotly_white"
                )

            elif graph_type == "Box Plot":
                fig = px.box(
                    df,
                    y=selected_column,
                    title=f"Box Plot of {selected_column}",
                    color_discrete_sequence=["#C9825A"],
                    template="plotly_white"
                )

            else:
                fig = px.violin(
                    df,
                    y=selected_column,
                    title=f"Violin Plot of {selected_column}",
                    color_discrete_sequence=["#C9825A"],
                    template="plotly_white"
                )

            fig.update_layout(
                height=480,
                title_x=0.5
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Statistical Summary")
            st.dataframe(
                df[[selected_column]].describe().round(2),
                use_container_width=True
            )

        else:
            st.info("No numeric columns are available.")

    with tab2:
        st.subheader("Delivery and Feature Insights")
        st.caption("Delivery Time is used in only a few charts. The remaining charts explore other useful dataset features.")

        column_lookup = {
            re.sub(r"[^a-z0-9]", "", str(column).lower()): column
            for column in df.columns
        }

        delivery_col = None
        for normalized, actual in column_lookup.items():
            if any(key in normalized for key in ["deliverytime", "timetaken", "deliveryduration", "deliveryminutes"]):
                values = pd.to_numeric(df[actual], errors="coerce")
                if values.notna().sum() > 0:
                    df[actual] = values
                    delivery_col = actual
                    break

        def find_column(aliases, numeric_only=False):
            for alias in aliases:
                wanted = re.sub(r"[^a-z0-9]", "", alias.lower())
                for normalized, actual in column_lookup.items():
                    if wanted in normalized and (not numeric_only or pd.api.types.is_numeric_dtype(df[actual])):
                        return actual
            return None

        distance_col = find_column(["distance", "distance km", "delivery distance"], True)
        preparation_col = find_column(["preparation time", "prep time", "food preparation time"], True)
        temperature_col = find_column(["temperature"], True)
        humidity_col = find_column(["humidity"], True)
        traffic_col = find_column(["traffic level", "traffic density", "traffic"])
        weather_col = find_column(["weather condition", "weather"])
        vehicle_col = find_column(["vehicle type", "vehicle condition", "vehicle"])

        chart_count = 0

        if delivery_col:
            delivery_values = pd.to_numeric(df[delivery_col], errors="coerce").dropna()
            if not delivery_values.empty:
                fig = px.histogram(
                    pd.DataFrame({"Delivery Time (minutes)": delivery_values}),
                    x="Delivery Time (minutes)", nbins=20,
                    title="Delivery Time Distribution",
                    color_discrete_sequence=["#C9825A"], template="plotly_white"
                )
                fig.update_layout(height=390, title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Shows which delivery-time ranges occur most frequently.")
                chart_count += 1

                if distance_col:
                    plot_df = df[[distance_col, delivery_col]].copy()
                    plot_df[distance_col] = pd.to_numeric(plot_df[distance_col], errors="coerce")
                    plot_df[delivery_col] = pd.to_numeric(plot_df[delivery_col], errors="coerce")
                    plot_df = plot_df.dropna()
                    if len(plot_df) > 1:
                        fig = px.scatter(
                            plot_df, x=distance_col, y=delivery_col,
                            title="Delivery Distance vs Delivery Time",
                            trendline="ols", opacity=0.7,
                            labels={distance_col: "Distance", delivery_col: "Delivery Time (minutes)"},
                            color_discrete_sequence=["#9D6245"], template="plotly_white"
                        )
                        fig.update_layout(height=390, title_x=0.5)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Checks whether longer trips are associated with longer delivery times.")
                        chart_count += 1

                category_target = next((col for col in [traffic_col, weather_col, vehicle_col] if col and df[col].nunique(dropna=True) > 1), None)
                if category_target:
                    plot_df = df[[category_target, delivery_col]].copy()
                    plot_df[delivery_col] = pd.to_numeric(plot_df[delivery_col], errors="coerce")
                    plot_df[category_target] = plot_df[category_target].fillna("Unknown").astype(str)
                    plot_df = plot_df.dropna(subset=[delivery_col])
                    if not plot_df.empty:
                        fig = px.box(
                            plot_df, x=category_target, y=delivery_col,
                            title=f"Delivery Time by {category_target}",
                            labels={category_target: category_target, delivery_col: "Delivery Time (minutes)"},
                            color_discrete_sequence=["#C9825A"], template="plotly_white"
                        )
                        fig.update_layout(height=390, title_x=0.5)
                        fig.update_xaxes(tickangle=-15, automargin=True)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Compares delivery-time spread across categories.")
                        chart_count += 1

        st.divider()
        st.subheader("Other Feature Visualizations")
        st.caption("These charts use input features without repeating Delivery Time.")

        st.caption("Each chart compares two related features: X is the horizontal axis and Y is the vertical axis.")

        if distance_col and preparation_col and distance_col != preparation_col:
            plot_df = df[[distance_col, preparation_col]].copy()
            plot_df[distance_col] = pd.to_numeric(plot_df[distance_col], errors="coerce")
            plot_df[preparation_col] = pd.to_numeric(plot_df[preparation_col], errors="coerce")
            plot_df = plot_df.dropna()
            if len(plot_df) > 1:
                fig = px.scatter(
                    plot_df, x=distance_col, y=preparation_col,
                    title="Distance vs Preparation Time", opacity=0.7,
                    labels={distance_col: f"X: {distance_col}", preparation_col: f"Y: {preparation_col}"},
                    color_discrete_sequence=["#C9825A"], template="plotly_white"
                )
                fig.update_layout(height=390, title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"X = {distance_col}; Y = {preparation_col}. Helps explore whether distance and preparation duration are related.")
                chart_count += 1

        if temperature_col and humidity_col and temperature_col != humidity_col:
            plot_df = df[[temperature_col, humidity_col]].copy()
            plot_df[temperature_col] = pd.to_numeric(plot_df[temperature_col], errors="coerce")
            plot_df[humidity_col] = pd.to_numeric(plot_df[humidity_col], errors="coerce")
            plot_df = plot_df.dropna()
            if len(plot_df) > 1:
                fig = px.scatter(
                    plot_df, x=temperature_col, y=humidity_col,
                    title="Temperature vs Humidity", opacity=0.7,
                    labels={temperature_col: f"X: {temperature_col}", humidity_col: f"Y: {humidity_col}"},
                    color_discrete_sequence=["#9D6245"], template="plotly_white"
                )
                fig.update_layout(height=390, title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"X = {temperature_col}; Y = {humidity_col}. Shows how temperature and humidity vary together.")
                chart_count += 1

        category_features = []
        for col in [traffic_col, weather_col, vehicle_col]:
            if col and col not in category_features and 2 <= df[col].nunique(dropna=True) <= 15:
                category_features.append(col)

        if category_features:
            cols = st.columns(2)
            for i, feature in enumerate(category_features):
                with cols[i % 2]:
                    counts = df[feature].fillna("Unknown").astype(str).value_counts().reset_index()
                    counts.columns = [feature, "Number of Records"]
                    if i < 2:
                        fig = px.bar(
                            counts, x=feature, y="Number of Records",
                            title=f"Records by {feature}",
                            color_discrete_sequence=["#C9825A" if i % 2 else "#9D6245"],
                            template="plotly_white"
                        )
                        fig.update_layout(height=360, title_x=0.5)
                        fig.update_xaxes(tickangle=-15, automargin=True)
                    else:
                        fig = px.pie(
                            counts, names=feature, values="Number of Records",
                            title=f"Share by {feature}", hole=0.42,
                            color_discrete_sequence=["#C9825A", "#DDBA9B", "#9D6245", "#E9D8C5"],
                            template="plotly_white"
                        )
                        fig.update_layout(height=360, title_x=0.5)
                    st.plotly_chart(fig, use_container_width=True)
                    chart_count += 1

        input_numeric_cols = [col for col in numeric_cols if col != delivery_col]
        if len(input_numeric_cols) >= 2:
            corr_df = df[input_numeric_cols].apply(pd.to_numeric, errors="coerce")
            corr = corr_df.corr().dropna(axis=0, how="all").dropna(axis=1, how="all")
            if corr.shape[0] >= 2:
                fig = px.imshow(
                    corr, text_auto=".2f", aspect="auto",
                    title="Correlation Between Input Features",
                    color_continuous_scale="Teal", zmin=-1, zmax=1,
                    template="plotly_white"
                )
                fig.update_layout(height=450, title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Highlights relationships between numeric input features; it excludes Delivery Time to reduce repetition.")
                chart_count += 1

        st.caption(f"Displayed {chart_count} charts using the columns available in your dataset. Delivery Time is used in at most three charts.")

    with tab3:

        st.subheader("Category Analysis")
        st.caption("Explore category proportions with a donut chart; the table below shows exact counts.")

        if category_analysis_cols:

            category_col = st.selectbox(
                "Choose Category",
                category_analysis_cols,
                key="category_column"
            )

            category_counts = (
                df[category_col]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .reset_index()
            )

            category_counts.columns = ["Category", "Count"]

            fig = px.pie(
                category_counts,
                names="Category",
                values="Count",
                title=f"Category Breakdown: {category_col}",
                hole=0.42,
                color_discrete_sequence=["#C9825A", "#DDBA9B", "#9D6245", "#E9D8C5", "#B7A08A"],
                template="plotly_white"
            )

            fig.update_layout(
                height=480,
                title_x=0.5,
                legend_title_text=category_col
            )

            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                category_counts,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "No suitable categorical columns are available for a readable bar chart. "
                "Columns with fewer than 2 or more than 30 unique values are hidden."
            )

    with tab4:

        st.subheader("Column Information")

        column_info = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(
            column_info,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Dataset Preview")

        search_text = st.text_input(
            "Search dataset values",
            placeholder="Enter a word or value..."
        )

        preview = df.copy()

        if search_text.strip():
            search_mask = pd.Series(False, index=preview.index)

            for column in preview.columns:
                search_mask = search_mask | preview[column].astype(
                    str
                ).str.contains(
                    search_text,
                    case=False,
                    na=False,
                    regex=False
                )

            preview = preview[search_mask]

        st.caption(f"Showing {len(preview):,} matching records")

        st.dataframe(
            preview.head(500),
            use_container_width=True,
            hide_index=True
        )

        csv_data = preview.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Filtered Dataset",
            data=csv_data,
            file_name="food_delivery_analysis.csv",
            mime="text/csv"
        )

elif menu == "Model Performance":

    st.html("""
        <div class="predict-hero">
            <div class="predict-tag">MODEL EVALUATION</div>
            <div class="predict-heading">
                Understand Your <span style="color:#C9825A;">Model Performance.</span>
            </div>
            <div class="predict-description">
                Evaluate your delivery-time prediction model using
                regression metrics and actual-versus-predicted charts.
            </div>
        </div>
    """)

    model_path = "pickle_file/food_delivery_xgboost.pkl"
    data_path = "dataset/foodtimedataset.csv"

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")

    elif not os.path.exists(data_path):
        st.error(f"Dataset file not found: {data_path}")

    else:
        try:
            model = joblib.load(model_path)
            df = pd.read_csv(data_path)

            # Use the same cleaning as the training notebook
            df.columns = df.columns.str.strip()

            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].str.strip()

            target = "Delivery Time"

            if target not in df.columns:
                st.error(f"Target column '{target}' was not found.")

            else:
                df[target] = pd.to_numeric(
                    df[target], errors="coerce"
                )

                df = df.dropna(subset=[target])
                df = df[df[target] > 0]

                # Use the same features as the training notebook
                X = df.drop(
                    columns=[
                        "ID",
                        "Delivery Person ID",
                        "Delivery Time"
                    ]
                )

                y = df["Delivery Time"]

                # Do not remove rows with missing input features.
                # The trained pipeline handles them using SimpleImputer.
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42
                )

                predictions = model.predict(X_test)

                results = pd.DataFrame({
                    "Actual Time": y_test.to_numpy(),
                    "Predicted Time": predictions
                })

                actual = results["Actual Time"]
                predicted = results["Predicted Time"]

                mae = mean_absolute_error(actual, predicted)
                mse = mean_squared_error(actual, predicted)
                rmse = mse ** 0.5
                r2 = r2_score(actual, predicted)

                st.subheader("Regression Metrics")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("MAE", f"{mae:.2f} min")
                col2.metric("MSE", f"{mse:.2f}")
                col3.metric("RMSE", f"{rmse:.2f} min")
                col4.metric("R² Score", f"{r2:.3f}")

                st.divider()

                st.subheader("Actual vs Predicted Time")

                fig1 = px.scatter(
                    results,
                    x="Actual Time",
                    y="Predicted Time",
                    title="Actual vs Predicted Delivery Time",
                    color_discrete_sequence=["#C9825A"],
                    template="plotly_white"
                )

                fig1.update_layout(
                    height=450,
                    title_x=0.5,
                    xaxis_title="Actual Time (minutes)",
                    yaxis_title="Predicted Time (minutes)"
                )

                st.plotly_chart(
                    fig1,
                    use_container_width=True
                )

                st.subheader("Delivery Time Comparison")

                comparison = results.head(20).reset_index(drop=True)

                fig2 = px.line(
                    comparison,
                    y=["Actual Time", "Predicted Time"],
                    markers=True,
                    title="Actual vs Predicted — First 20 Samples",
                    color_discrete_sequence=[
                        "#C9825A",
                        "#657765"
                    ],
                    template="plotly_white"
                )

                fig2.update_layout(
                    height=400,
                    title_x=0.5,
                    xaxis_title="Test Sample",
                    yaxis_title="Delivery Time (minutes)",
                    legend_title_text="Delivery Time"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

        except Exception as e:
            st.error("Could not evaluate the model.")
            st.write("Error details:", str(e))
            st.info(
                "Check that the saved model and dataset match "
                "the training notebook."
            )

elif menu == "Feature Importance":

    st.html("""
        <div class="predict-hero">
            <div class="predict-tag">MODEL INSIGHTS</div>
            <div class="predict-heading">
                Understand Your
                <span style="color:#C9825A;">Feature Importance.</span>
            </div>
            <div class="predict-description">
                Discover which input features have the greatest
                influence on your food delivery time predictions.
            </div>
        </div>
    """)

    model_path = "pickle_file/food_delivery_xgboost.pkl"
    data_path = "dataset/foodtimedataset.csv"

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")

    elif not os.path.exists(data_path):
        st.error(f"Dataset file not found: {data_path}")

    else:
        try:
            model = joblib.load(model_path)
            df = pd.read_csv(data_path)

            trained_model = model

            if hasattr(model, "steps"):
                trained_model = model.steps[-1][1]

            if not hasattr(trained_model, "feature_importances_"):
                st.error("Feature importance is not supported by this model.")

            else:
                importance = trained_model.feature_importances_

                feature_names = None

                if hasattr(model, "named_steps"):
                    preprocessor = model.named_steps.get("preprocessor")

                    if preprocessor is not None and hasattr(
                        preprocessor, "get_feature_names_out"
                    ):
                        feature_names = preprocessor.get_feature_names_out()

                if feature_names is None or len(feature_names) != len(importance):
                    feature_names = [
                        f"Feature {i + 1}" for i in range(len(importance))
                    ]

                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importance
                })

               
                original_columns = df.drop(
                    columns=[
                        "Delivery Time",
                        "ID",
                        "Delivery Person ID"
                    ],
                    errors="ignore"
                ).columns.tolist()

                def get_original_feature(feature):
                    feature = str(feature)

                    feature = feature.replace("num__", "")
                    feature = feature.replace("cat__", "")

                    for column in sorted(
                        original_columns, key=len, reverse=True
                    ):
                        if feature == column or feature.startswith(column + "_"):
                            return column

                   
                    for column in sorted(
                        original_columns, key=len, reverse=True
                    ):
                        if column in feature:
                            return column

                    return feature

                importance_df["Original Feature"] = (
                    importance_df["Feature"].apply(get_original_feature)
                )

                grouped_df = (
                    importance_df.groupby("Original Feature", as_index=False)
                    ["Importance"].sum()
                )

                grouped_df = grouped_df.sort_values(
                    "Importance", ascending=False
                ).reset_index(drop=True)

               
                total_importance = grouped_df["Importance"].sum()

                if total_importance > 0:
                    grouped_df["Importance (%)"] = (
                        grouped_df["Importance"] / total_importance * 100
                    ).round(2)
                else:
                    grouped_df["Importance (%)"] = 0.0

         
                col1, col2, col3 = st.columns(3)

                col1.metric("Total Features", len(grouped_df))

                col2.metric(
                    "Most Important Feature",
                    grouped_df.iloc[0]["Original Feature"]
                )

                col3.metric(
                    "Highest Importance",
                    f"{grouped_df.iloc[0]['Importance (%)']:.2f}%"
                )

                st.divider()

                st.subheader("Feature Importance Chart")

                top_n = st.slider(
                    "Number of features to display",
                    min_value=1,
                    max_value=len(grouped_df),
                    value=min(10, len(grouped_df))
                )

                chart_df = grouped_df.head(top_n).sort_values(
                    "Importance (%)"
                )

                fig = px.bar(
                    chart_df,
                    x="Importance (%)",
                    y="Original Feature",
                    orientation="h",
                    text="Importance (%)",
                    title="Importance of Delivery Prediction Features",
                    color="Importance (%)",
                    color_continuous_scale=["#E9D8C5", "#C9825A"],
                    template="plotly_white"
                )

                fig.update_traces(
                    texttemplate="%{text:.2f}%",
                    textposition="outside",
                    cliponaxis=False
                )

                fig.update_layout(
                    height=max(400, top_n * 45),
                    title_x=0.5,
                    coloraxis_showscale=False,
                    xaxis_title="Importance (%)",
                    yaxis_title="Feature",
                    margin=dict(l=20, r=70, t=60, b=30)
                )

                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Feature Ranking")

                display_df = grouped_df[
                    ["Original Feature", "Importance (%)"]
                ].copy()

                display_df.index = range(1, len(display_df) + 1)
                display_df.index.name = "Rank"

                st.dataframe(
                    display_df,
                    use_container_width=True
                )

                st.info(
                    "Categorical features are grouped by their original "
                    "column names. Importance represents the combined "
                    "model importance of their encoded categories. "
                    "It does not indicate whether a feature increases "
                    "or decreases delivery time."
                )

        except Exception as e:
            st.error("Could not display feature importance.")
            st.caption(str(e))


elif menu == "About Project":

    st.markdown("""
        <div class="about-hero">
            <p class="about-label">MACHINE LEARNING PROJECT</p>
            <h1>About Our Project</h1>
            <p class="about-subtitle">
                Smart Food Delivery Time Estimation
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="about-section">
            <h3>Project Overview</h3>
            <p>
                Food Delivery Time Estimation is a machine learning
                project designed to predict the estimated time required
                to deliver a food order. It analyzes available order
                information and historical delivery data to generate
                delivery time predictions.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3>Project Objectives</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="about-card">
                <h4>01. Time Prediction</h4>
                <p>Estimate the delivery duration for food orders.</p>
            </div>
            <div class="about-card">
                <h4>02. Data Analysis</h4>
                <p>Analyze delivery data using charts and statistics.</p>
            </div>
            <div class="about-card">
                <h4>03. Feature Analysis</h4>
                <p>Identify which available features influence model predictions.</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="about-card">
                <h4>04. Model Evaluation</h4>
                <p>Evaluate prediction errors using MAE, RMSE, and R².</p>
            </div>
            <div class="about-card">
                <h4>05. Interactive Dashboard</h4>
                <p>Provide an easy interface for entering order details.</p>
            </div>
            <div class="about-card">
                <h4>06. Better Planning</h4>
                <p>Help users understand estimated delivery durations.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<h3>Technologies Used</h3>", unsafe_allow_html=True)

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.markdown("""
            <div class="tech-card">
                <h4>Python</h4>
                <p>Programming language</p>
            </div>
        """, unsafe_allow_html=True)

    with tech2:
        st.markdown("""
            <div class="tech-card">
                <h4>XGBoost</h4>
                <p>Delivery time regression model</p>
            </div>
        """, unsafe_allow_html=True)

    with tech3:
        st.markdown("""
            <div class="tech-card">
                <h4>Streamlit</h4>
                <p>Interactive web dashboard</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="about-section">
            <h3>How the System Works</h3>
            <div class="workflow">
                <p><b>Step 1:</b> Collect food delivery data.</p>
                <p><b>Step 2:</b> Clean and preprocess the dataset.</p>
                <p><b>Step 3:</b> Train the XGBoost regression model.</p>
                <p><b>Step 4:</b> Enter order details in the dashboard.</p>
                <p><b>Step 5:</b> Display the predicted delivery time.</p>
            </div>
        </div>

        <div class="about-footer">
            <h3>Project Outcome</h3>
            <p>
                An interactive machine learning application for delivery
                time prediction, data analysis, model performance, and
                feature importance visualization.
            </p>
        </div>
    """, unsafe_allow_html=True)
