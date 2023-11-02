import numpy as np 
import pandas as pd
import plotly.express as px 
import streamlit as st
import openai
import numpy as np
from PyPDF2 import PdfReader
from streamlit.logger import get_logger
@st.cache_data
def visualize_timeseries(df, level, country, channel, sector, price_tier):
    df_t = df[df["geo"] == country]
    if channel:
        df_t = df_t[df_t["channel"] == channel]
    if sector:
        df_t = df_t[df_t["sector"] == sector]
    if price_tier:
        df_t = df_t[df_t["price_tier"] == price_tier]

    if df_t.empty:
        st.warning("No data available for the selected combination.")
    else:
        group_cols = level + ["month","scenario"]
        aggregation = {"volume": "sum"}
        df_t = df_t.groupby(group_cols, as_index=True).agg(aggregation).reset_index()
    df_t = df_t.dropna()
    chart_data = df_t.set_index("month")
    title = "_".join([country] + [val for val in [channel, sector, price_tier] if val])
    color_discrete_map = {
        "historical": " #FF4B4B",
        "forecasted": "blue"
    }

    volume_chart = px.line(
        chart_data,
        x=chart_data.index,
        y="volume",
        title=title,
        color="scenario",
        color_discrete_map=color_discrete_map
    )
    volume_chart.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font_color='black' ,
    height=400, 
    width=800,
    margin=dict(l=50, r=10, t=50, b=10) 
    
    )
    volume_chart.update_layout(xaxis_title="Month", yaxis_title="volume")
    df_t["year"] = pd.to_datetime(df_t["month"]).dt.year
    df_yoy = df_t.groupby(["year"]).sum()["volume"].reset_index()
    grouped_yoy = df_yoy[0:-1]
    grouped_yoy['yoy_growth'] = grouped_yoy['volume'].pct_change(periods=1) * 100
    grouped_yoy=grouped_yoy[["year","yoy_growth"]]
    color_discrete_map_y = {
        "yoy_growth": " darkmagenta",
    
    }
    yoy_chart= px.bar(data_frame=grouped_yoy,x="year",
                      y="yoy_growth",title="YoY Change",
                      text=grouped_yoy["yoy_growth"].apply(lambda x: f'{x:.2f}'),
                      color_discrete_map=color_discrete_map_y
                    )
    yoy_chart.update_layout(
    plot_bgcolor=' white',
    paper_bgcolor='white',
    font_color='black',
    height=400,
    margin=dict(l=50, r=50, t=50, b=10)  
   
    )
    st.markdown("<hr style='border: 1px solid red; width: 100%;'>", unsafe_allow_html=True)
    col1, col2 = st.columns([0.7,0.3])

    # Display the volume_chart in the first column
    with col1:
        st.plotly_chart(volume_chart, use_container_width=True)
    
    # Display the yoy_chart in the second column
    with col2:
        st.plotly_chart(yoy_chart, use_container_width=True)

    return df_t


@st.cache_data
def get_completion(prompt, model="gpt-3.5-turbo"):

    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(

    model=model,

    messages=messages,

    temperature=0,

    )

    return response.choices[0].message["content"]

@st.cache_data
def yoy_growth(df):
    df["year"] = pd.to_datetime(df["month"]).dt.year
    df_yoy = df.groupby(["year"]).sum()["volume"].reset_index()
    grouped_yoy = df_yoy[1:-1]
    grouped_yoy['yoy_growth'] = grouped_yoy['volume'].pct_change(periods=1)*100
    return grouped_yoy[["year","yoy_growth"]]


@st.cache_data
def calculate_trend_slope_dataframe(dataframe, polynomial_degree=1):
    if dataframe.empty:
        st.warning("No data available for the selected combination.")
    else:
        dataframe=dataframe.reset_index(drop=True)
        df_copy = dataframe.copy() 
        df_copy['cumulative_sum'] = df_copy['volume'].cumsum()
        first_nonzero_index = df_copy['cumulative_sum'].ne(0).idxmax()
        df_copy = df_copy.iloc[first_nonzero_index:]
        df_copy.drop(columns=['cumulative_sum'], inplace=True)
        df_copy_his =df_copy[df_copy["scenario"]=="historical"]
        df_copy_for = df_copy[df_copy["scenario"]=="forecasted"]
        time_points_his = [i for i in range(len(df_copy_his["volume"]))]
        volume_values_his = df_copy_his["volume"]
        coefficients_his = np.polyfit(time_points_his, volume_values_his, polynomial_degree)
        slope_his = coefficients_his[0]
        df_copy_his["slope_his"]=slope_his
        if slope_his>1:
            df_copy_his["trend"]="Increasing"
        elif slope_his <-1:
            df_copy_his["trend"]="Decreasing"
        else:
            df_copy_his["trend"]="No Trend"
        time_points_for = [i for i in range(len(df_copy_for["volume"]))]
        volume_values_for = df_copy_for["volume"]
        coefficients_for = np.polyfit(time_points_for, volume_values_for, polynomial_degree)
        slope_for = coefficients_for[0]
        df_copy_for["slope_for"]=slope_for
        if slope_for>1:
            df_copy_for["trend"]="Increasing"
        elif slope_for <-1:
            df_copy_for["trend"]="Decreasing"
        else:
            df_copy_for["trend"]="No Trend"
        df_final = pd.concat([df_copy_his,df_copy_for])

        return df_final
    
@st.cache_data
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


import os
@st.cache_data
def read_text_file(filename):
    data = []
    full_path = os.path.join(os.getcwd(), filename) 
    with open(full_path, "r") as inp:
        for line in inp:
            stripped_line = line.strip()
            if stripped_line:
                data.append(stripped_line)
    return data

model ="gpt-3.5-turbo-0301"

@st.cache_data(show_spinner=False)
def is_open_ai_key_valid(openai_api_key) -> bool:
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar!")
        return False
    try:
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            api_key=openai_api_key,
        )
    except Exception as e:
        st.error(f"{e.__class__.__name__}: {e}")
        logger.error(f"{e.__class__.__name__}: {e}")
        return False
    return True


@st.cache_data  
def recommend_products(user_id,df,top_n=5):
    if user_id not in df["user_id"].unique().tolist():
        # If the user is new, recommend the top-rated products
        top_rated_products = list(pd.Series(df.sort_values(by='rating', ascending=False)['product_id'].head(top_n)))
        return top_rated_products
    else:
        pivot_table = df.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
        user_ratings = pivot_table.loc[user_id]
        similarity = pivot_table.corrwith(user_ratings, axis=0)
        similar_products = similarity.sort_values(ascending=False).index[1:]  
        recommended_products = [product for product in similar_products if product not in df[df['user_id'] == user_id]['product_id']]
        return recommended_products[:top_n]
    

# def base64_to_pil(base64_string):
#     """
#     Purpose:
#         Turn base64 string to image with PIL
#     Args/Requests:
#          base64_string: base64 string of image
#     Return:
#         image: PIL image
#     """
#     import base64

#     imgdata = base64.b64decode(base64_string)
#     image = Image.open(io.BytesIO(imgdata))
#     return image