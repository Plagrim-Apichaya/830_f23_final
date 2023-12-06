import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import altair_viewer

# classifier
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix

# color palette
color_palette = sns.color_palette()
plt.style.use("fivethirtyeight")

color = ["#3DB2FF", "#FF2442", "#FFB830"] 
color_pal = sns.color_palette()

#########################################
#### ------- data management ------- ####
#########################################

#### download and clean the dataset

url = "https://raw.githubusercontent.com/Plagrim-Apichaya/830_f23_final/main/NDVI_Thailand.csv"
df_raw = pd.read_csv(url)
df_raw["NDVI_ratio"] = round(df_raw.iloc[:,2]/10000, 3)
#st.dataframe(df_raw)

df = df_raw.copy()
df = df.set_index('Date')
df = df.drop(df.columns[0], axis = 1)
df = df.drop(columns = ['Average NDVI'])
df.index = pd.to_datetime(df.index, format='%d/%m/%Y')
#st.dataframe(df)

X = df.index
y = df["NDVI_ratio"]

#### add feature to time series dataset
def create_features(df):

    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week
    return df

df_add = create_features(df) # add feature to the time series data
df_col = df_add.columns  # define column of the data with features

#### add lag features
def add_lags(df):
    df['lag1'] = df['NDVI_ratio'].shift(periods = 16)
    df['lag2'] = df['NDVI_ratio'].shift(periods = 32)
    df['lag3'] = df['NDVI_ratio'].shift(periods = 48)
    return df

df_add = add_lags(df_add)

#########################################
#### -------- XGB Regressor -------- ####
#########################################

#### Model
#n = 4 # number of fold
tss = TimeSeriesSplit(n_splits = 3, test_size = 80, gap = 0)

preds = []
scores = []
for train_idx, val_idx in tss.split(df_add):
    train = df_add.iloc[train_idx]
    test = df_add.iloc[val_idx]

    train = create_features(train)
    test = create_features(test)

    FEATURES = df_col[1:]
    TARGET = df_col[0]

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    reg = xgb.XGBRegressor(base_score = 0.5, booster = 'gbtree',    
                        n_estimators = 1000,
                        early_stopping_rounds = 50,
                        objective = 'reg:linear',
                        max_depth = 3,
                        learning_rate = 0.01)
    reg.fit(X_train, y_train,
            eval_set = [(X_train, y_train), (X_test, y_test)],
            verbose = 100)

    y_pred = reg.predict(X_test)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)
    mean_score = np.mean(scores)

#########################################
#### -------- visualization -------- ####
#########################################

##### ------ NDVI Profile ----- #####
scatter = alt.Chart(df_raw).mark_point(size=50, color = "orangered").encode(
    x='Date:T',
    y='NDVI_ratio:Q',
    tooltip=['Date:T', 'NDVI_ratio:Q']
)
line = alt.Chart(df_raw).mark_line(color = color[0]).encode(
    x='Date:T',
    y='NDVI_ratio:Q'
)
NDVI_profile = (scatter + line).properties(
    width = 700,
    height = 400
).configure_axis(
    labelFontSize = 15,
    titleFontSize = 15
).configure_legend(
    titleFontSize = 12
)

#########################################
#### ------- page management ------- ####
#########################################

page = st.sidebar.radio("MENU -- Select",["🏠 Home","⚙️ Model","🔮 Prediction","🧸 Bio"])

#### Home
if page == "🏠 Home":
    st.title("NDVI")
    st.write("Apichaya Thaneerat, MSU")
    st.image("https://img.freepik.com/premium-photo/forest-top-view-landscape-panorama-view-summer-forest-with-quadrocopter-aerial-view_548821-2296.jpg")
    
    st.write("NDVI is normalized different vegetation index which define as NDVI = NIR -RED/ NIR + RED. Which NIR is the reflectance from Near Infra-red region (750 - 1500 nm) while RED is the reflectance from red range in visible spectrum (650 - 750 nm).")
    st.image("https://static.tildacdn.com/tild3663-6233-4335-b139-366231646535/ndvi_formule.svg", width = 200)

    st.write("NDVI is used as vegetation indices to indicate the greenness of the location. NDVI is used various across social science field including land use and land cover associated project, study of vegetation stage and/or status, mental and physical health impact from greenspace study, and many more.")

    st.subheader("Dataset")
    st.write("MODIS product of 250m 16 days composite median NDVI at Chachoengsao, Thailand. ")
    st.write("Source: 250m_16_days_NDVI MODSI Product.")
    st.write("Location: Latitude [13.665556] Longitude [101.441389]: Horizontal Tile [27] Vertical Tile [07] Sample [1028] Line [760]")
    st.write("Ground Station: th_chachoengsao_rubber_flux_chachoengsao, Chachoengsao, Thailand")
    st.write("From the product, we compute the average of the Median NDVI dataset")


    st.subheader("NDVI Profile")
    st.altair_chart(NDVI_profile)


    ##### ------ boxplot ----- #####
    st.subheader("NDVI in temporal context")
    boxplot_type = st.selectbox("Select the option here", 
                             ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'weekofyear'])
    column_lst = ['NDVI_ratio', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear','weekofyear', 'lag1', 'lag2', 'lag3', 'isFuture']
    #col_n = 3 # user can choose
    col_n = column_lst.index(boxplot_type)
    by_time = df_add.columns[col_n]
    title_name = 'NDVI by ' + str(by_time)

    color_scale = alt.Scale(scheme = 'rainbow')

    boxplot = alt.Chart(df_add).mark_boxplot().encode(
        x=f'{by_time}:O',  # Assuming by_time is a categorical variable
        y='NDVI_ratio:Q',
        color=alt.Color(f'{by_time}:O', scale=color_scale),
        tooltip=[alt.Tooltip(by_time), alt.Tooltip('NDVI_ratio:Q', title='NDVI Ratio')]
    ).properties(
        width = 700,
        height = 400,
        title=title_name
    ).configure_axis(
        labelFontSize = 12,
        titleFontSize = 15
    ).configure_title(
        fontSize = 15
    )   
    st.altair_chart(boxplot)
    
#### Model
if page == "⚙️ Model":
    
    st.title("What model we use for the NDVI prediction?")
    st.subheader("XGB Regressor")
    st.image("https://cdn-icons-png.flaticon.com/512/2172/2172943.png", width = 300)
    
    st.write("Extreme Gradient Boosting (XGBoost) is an efficient and effective implementation of the gradient boosting algorithm. It is one of the popular models in Machine Learning. It is used in the regression predictive modeling problem that return the numeric results such that to predict the price of the goods. The beneficial part is that we can apply cross-validation technique for the NDVI time series dataset and validate the score from each fold regarding the temporal context.")
    
    #########################################
    #### -------- XGB Regressor -------- ####
    #########################################
    n = st.slider("Folds", min_value = 2, max_value = 10, value = 3, step = 1)
    #### Model
    #n = 4 # number of fold
    tss = TimeSeriesSplit(n_splits = n, test_size = 80, gap = 0)

    preds = []
    scores = []
    for train_idx, val_idx in tss.split(df_add):
        train = df_add.iloc[train_idx]
        test = df_add.iloc[val_idx]

        train = create_features(train)
        test = create_features(test)

        FEATURES = df_col[1:]
        TARGET = df_col[0]

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_test = test[FEATURES]
        y_test = test[TARGET]

        reg = xgb.XGBRegressor(base_score = 0.5, booster = 'gbtree',    
                            n_estimators = 1000,
                            early_stopping_rounds = 50,
                            objective = 'reg:linear',
                            max_depth = 3,
                            learning_rate = 0.01)
        reg.fit(X_train, y_train,
                eval_set = [(X_train, y_train), (X_test, y_test)],
                verbose = 100)

        y_pred = reg.predict(X_test)
        preds.append(y_pred)
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(score)
        mean_score = np.mean(scores)

        
    #### ------ Folds ------ ####
    st.subheader("cross-validation")
    fig, axs = plt.subplots(n, 1, figsize=(12, 12), sharex=True)

    fold = 0
    for train_idx, test_idx in tss.split(df_add):
        
        axs[fold].plot(df_add.index, 
                    df_add['NDVI_ratio'], 
                    label='Full Dataset', 
                    color='gray')
        axs[fold].plot(df_add.index[train_idx], 
                    df_add['NDVI_ratio'].iloc[train_idx],
                    label='Training Set', 
                    color=color[0])
        axs[fold].plot(df_add.index[test_idx], 
                    df_add['NDVI_ratio'].iloc[test_idx],
                    label='Test Set', 
                    color="tomato")
        
        axs[fold].axvline(df_add.index[test_idx].min(), color='black', ls='--')
        axs[fold].set_title(f'Data Train/Test Split Fold {fold}')
        axs[fold].legend()
        fold += 1
        Fold_plot = fig

    st.pyplot(Fold_plot)
    st.subheader("Score")
    st.write("Root Mean Square Error (RMSE) is the metric to validate the model in each temporal context.")
    st.write("The score of each fold is RMSE which you can see the details below.")
    st.write(f'Mean score across the folds: {mean_score:0.4f}')
    st.write('Folds score:', scores)

#### Predict
if page == "🔮 Prediction":
    st.title("NDVI Prediction")
    st.image("https://eos.com/wp-content/uploads/2023/01/stages-of-crop-development.jpg")
    
    st.subheader("After training the data and validate the model, now it is your time to predict the future NDVI.")
    st.write("NDVI value can also predict the stage of the crop development")
    st.write("Higher NDVI = more greenery -> growing season, more greenspace, land cover changes into vegetation, crop field.")
    st.write("Lower NDVI = less greenery -> harvesting season, planting season, deforestation, land cover changes into water, aqua agriculture, building.")

    date_range = pd.date_range('2023-11-01', '2030-12-31', freq='16D')
    date_str_lst = [date.strftime('%Y-%m-%d') for date in date_range]
    
    date = st.selectbox("Please select your interest date to predict NDVI", date_str_lst)

    #### Prediction
    # Retrain using all data for the prediction
    df_all = create_features(df)

    FEATURES = df_col[1:]
    TARGET = df_col[0]

    X_all = df_all[FEATURES]
    y_all = df_all[TARGET]

    reg = xgb.XGBRegressor(base_score = 0.5,
                        booster = 'gbtree',    
                        n_estimators = 300,
                        objective = 'reg:linear',
                        max_depth = 3,
                        learning_rate = 0.01)
    reg.fit(X_all, y_all,
            eval_set = [(X_all, y_all)],
            verbose = 100)

    future = pd.date_range('2023-11-01', date, freq = '16D')
    future_df = pd.DataFrame(index = future)
    future_df['isFuture'] = True
    df_add['isFuture'] = False
    df_with_future = pd.concat([df_add, future_df])
    df_with_future = create_features(df_with_future)
    df_with_future = add_lags(df_with_future)

    future_with_features = df_with_future.query('isFuture').copy()
    future_with_features['pred'] = reg.predict(future_with_features[FEATURES])

    #### ------ prediction ------ ####
    scatter = alt.Chart(future_with_features.reset_index()).mark_point(size=50, color = "orangered").encode(
        x='index:T', 
        y=alt.Y('pred:Q', title='Predicted NDVI'),
        tooltip=[alt.Tooltip('index:T', title='Date'), alt.Tooltip('pred:Q', title='Prediction', format='.3f')]
    )
    line = alt.Chart(future_with_features.reset_index()).mark_line(color = color[2]).encode(
        x='index:T', 
        y='pred:Q'
    )
    Prediction = (scatter + line).properties(
        width=700,
        height=400
    ).configure_axis(
        labelFontSize=15,
        titleFontSize=15
    ).configure_legend(
        titleFontSize=12
    )

    st.altair_chart(Prediction)
    
#### Bio
if page == "🧸 Bio":
    st.title("Web-app Developer Bio")
    st.subheader("Hello! Thank you so much for visiting my website to learn more about NDVI today. Now it is a chance to learn more about me. ")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://preview.redd.it/i-did-a-fanart-of-howls-moving-castle-v0-zg494lwk2ta91.png?auto=webp&s=cd49a246059662b5ad3f78bb51ead7ca379a837c")
        st.write("🐱 Black cat and Ghibli big fan!")
    with col2:
        st.markdown("##### My name is Apichaya Thaneerat.")
        st.write("I am from Thailand 🇹🇭.")
        st.write("I am first year Master student in Geography Department, Michigan State University (MSU) GO GREEN!")
        st.write("🛰 My interest is in Remote Sensing, GIS and Agriculture in Thailand.")
        st.write("Aside from sitting in front of the screen for school and work, I spend most of the time in the kitchen. I enjoy cooking in different cuisines such as Japanese 🍱, Korean 🍲, Thai 🍛, Italy 🧀, Western 🍔.")
        st.write("Being geographer is to be both a scientist 🌎 and an artist 🎨. I also enjoy drawing and painting when I want to boost my creativity.")
        






