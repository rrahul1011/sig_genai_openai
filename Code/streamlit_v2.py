import numpy as np 
import pandas as pd
import streamlit as st 
import openai
import os
from function import visualize_timeseries ,get_completion,yoy_growth,calculate_trend_slope_dataframe,\
extract_text_from_pdf,read_text_file,model,is_open_ai_key_valid,recommend_products,find_max_min_volume_months,\
send_email,send_email_via_imap,country_wise_analysis
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pyperclip
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain  
import requests
import io
from PIL import Image
import json
from email.mime.text import MIMEText
import smtplib
from prompt_per_msg import customer_style, template_string, template_string_new, best_selling_product, welcome_offer,instruction_existing

st.set_option('deprecation.showPyplotGlobalUse', False)


st.set_page_config(
            page_title="Sigmoid GenAI",
            page_icon="Code/cropped-Sigmoid_logo_3x.png",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
st.sidebar.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
st.sidebar.image("Code/cropped-Sigmoid_logo_3x.png", use_column_width=True)
st.sidebar.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
st.markdown('<style>div.row-widget.stButton > button:first-child {background-color: red; color: white;}</style>', unsafe_allow_html=True)

with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n" 
            "2. Press Enter"
            
          
        )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.", 
            value=os.environ.get("OPENAI_API_KEY", None)
            or st.session_state.get("OPENAI_API_KEY", ""),
        )
        if st.sidebar.button("Enter"):
            st.session_state["OPENAI_API_KEY"] = api_key_input


openai_api_key = st.session_state.get("OPENAI_API_KEY")
if not openai_api_key:
    st.warning(
        "🔐 Enter API Key to Know More About Me 😊, You can get a key at"
        " https://platform.openai.com/account/api-keys."
    )
if not is_open_ai_key_valid(openai_api_key):
    st.stop()



##Reading the data
df_dash = pd.read_csv("Data/Diageo_gen.csv")
tab1, tab2 ,tab3,tab4,tab5,tab6= st.tabs(["###### |About the App|", "###### |📊Demand Forecasting Interpreater|","###### |CodeAI|","###### |Q&A|","###### |Personalized Welcome Message|","###### |ImageGen|"])
with tab2:
    def main():
        """
        Tab for visualizing and analyzing time series data using an AI model.

        """
        

        def select_level(d):
            """
            Select data levels and additional options.

            Parameters:
            - d: DataFrame containing the data.

            Returns:
            - A tuple containing selected levels and other options.
            """
            
            

            # Create a list to store selected options
            selected_levels = []
            col_cou1,col_cou2,col_cou3,col_cou4=st.columns(4)
            with col_cou1:
                geo_options = d["geo"].unique().tolist()
                st.markdown('<p style="border: 2px solid red; padding: 1px; font-weight: bold;color: blue;size:4;">Select Country:</p>', unsafe_allow_html=True)
                selected_geo = st.selectbox("", geo_options)
                d=d[d["geo"]==selected_geo]
                selected_levels.append("geo")
            c1,c2,c3,c4=st.columns(4)

            with c1:
                st.markdown('<p style="border: 2px solid red; padding: 0.1px; font-weight: bold;color: blue;">Select Hierarchy :</p>', unsafe_allow_html=True)
            # Create columns for checkboxes
            col1, col2, col3 = st.columns(3)
            
            #Create a checkbox for each level
            with col1:
                #st.markdown('<span style="font-size: 20px;"><font color="blue" size=4><b>Channel:</b></font></span>', unsafe_allow_html=True)
                checkbox = st.checkbox("###### :red[Channel] 🛒", value="channel" in selected_levels, key="channel")
                if checkbox:
                    selected_levels.append("channel")

            with col2:
                #st.markdown('<span style="font-size: 20px;"><font color="blue" size=4><b>Sector:</b></font></span>', unsafe_allow_html=True)
                checkbox = st.checkbox("###### :red[Sector] 🍺", value="sector" in selected_levels, key="sector")
                if checkbox:
                    selected_levels.append("sector")

            with col3:
                #st.markdown('<span style="font-size: 20px;"><font color="blue" size=4><b>Price Tier:</b></font></span>', unsafe_allow_html=True)
                checkbox = st.checkbox("###### :red[Price Tier] 💲", value="price_tier" in selected_levels, key="price_tier")
                if checkbox:
                    selected_levels.append("price_tier")
            #selected_geo="Great Britain"
            selected_channel = None
            selected_sector = None
            selected_price_tier = None
            # Create columns for select boxes
            col1, col2, col3= st.columns(3)
            with col1:
                if "channel" in selected_levels:
                    channel_options = d["channel"].unique().tolist()
                    #st.markdown('<p style="border: 2px solid red; padding: 1px; font-weight: bold;color: blue;size:4;">Select Channel:</p>', unsafe_allow_html=True)
                    selected_channel = st.selectbox("", channel_options)
                    d=d[d["channel"]==selected_channel]

            with col2:
                if "sector" in selected_levels:
                    sector_options = d["sector"].unique().tolist()
                    #st.markdown('<p style="border: 2px solid red; padding: 1px; font-weight: bold;color: blue;size:4;">Select Sector:</p>', unsafe_allow_html=True)
                    selected_sector = st.selectbox("", sector_options)
                    d=d[d["sector"]==selected_sector]

            with col3:
                if "price_tier" in selected_levels:
                    price_tier_options = d["price_tier"].unique().tolist()
                    #st.markdown('<p style="border: 2px solid red; padding: 1px; font-weight: bold;color: blue;size:4;">Select Price Tier:</p>', unsafe_allow_html=True)
                    selected_price_tier = st.selectbox("", price_tier_options)

            return selected_levels, selected_channel, selected_sector, selected_price_tier,selected_geo


        # Main Streamlit app
        st.markdown('<p style="color:blue; font-size:30px; font-weight:bold; text-align:center;">Time Series Dashboard 📈</p>', unsafe_allow_html=True)
        st.markdown("<hr style='border: 1px solid red; width: 100%;'>", unsafe_allow_html=True)


        # Select data levels and additional options
        selected_levels = select_level(df_dash)

        ### Country wise analysis
        st.markdown('<p style="color:blue; font-size:20px; font-weight:bold; text-align:left;"><u>Country Level Analysis</u></p>', unsafe_allow_html=True)
        country_analysis=country_wise_analysis(df_dash,selected_levels[4])
        analysis_string_email = """Generate the email to CMO based on instruction\
                                that is delimited by triple backticks.\
                                instruction: ```{instruction_analysis_email}```
                               
                                """


        instruction_analysis_email = f"""You are Rahul genearte email based on the dictinoary information.
            1. You will be analyzing two dictinoary: channel_share, sector_share.
            2.Generate a short and precise email based on the dictinoary information
            3.The dictinoary are {country_analysis[0]} and {country_analysis[1]}
            4.Do not include the dictinoary name in the email also do not incl"""
            
        
        analysis_template_email = ChatPromptTemplate.from_template(analysis_string_email)
        chat = ChatOpenAI(temperature=0.0, model=model, openai_api_key=openai_api_key)
        email_analysis = analysis_template_email.format_messages(instruction_analysis_email=instruction_analysis_email)

        #### Email Part
        df_email = pd.read_csv("/Users/rahulkushwaha/Desktop/SigmoidGenai/sig_genai_openai/Data/email_details.csv")
        receiver_email=df_email['Email_Details'].apply(lambda x: json.loads(x)['email_receiver']).values[0]
        sender_email=df_email['Email_Details'].apply(lambda x: json.loads(x)['email_sender']).values[0]
        subject=df_email['Email_Details'].apply(lambda x: json.loads(x)['subject']).values[0]
        sender_password=st.secrets["gmail_key"]
       
        email_choice = st.radio("##### Choose an action:", ["View Analysis", "Send Email", "Draft Email"], index=0)
        response_email = chat(email_analysis)
        body=response_email.content
        if st.button("Analysis"):
            st.markdown(
                    f'<div style="border: 2px solid red; padding: 10px; background-color: white; color: black;">'
                    f'<div style="font-size: 24px; color: red;">💬</div>' 
                    f'{response_email.content}'
                    f'</div>',
                    unsafe_allow_html=True
                )


        if email_choice == "Send Email":
            try:
                send_email(sender_email, sender_password, receiver_email, subject, body)
                st.success('Email sent successfully! 🚀')
            except Exception as e:
                st.error(f"Erreur lors de l’envoi de l’e-mail : {e}")

        elif email_choice == "Draft Email":
            try:
                send_email_via_imap(sender_email, sender_password, receiver_email, subject, body)
                st.success('Email Drafted successfully! 🚀')
            except Exception as e:
                st.error(f"Erreur lors de l’envoi de l’e-mail : {e}")

        else:
           pass


        # Time Series Visualization Section
        data = visualize_timeseries(df_dash, selected_levels[0], selected_levels[4],
                                    selected_levels[1], selected_levels[2], selected_levels[3])
        

        
        data_trend = calculate_trend_slope_dataframe(data)
        seasonal_patter_dict=find_max_min_volume_months(data)
        if data_trend is None:
            pass
        elif data_trend.empty:
            pass
        else:
            data_trend_2 = data_trend.groupby(["scenario", "trend"])[["slope_his", "slope_for"]].mean().reset_index()
            trend_dict=data_trend_2[["scenario","trend"]].set_index("scenario")["trend"].to_dict()
        if data.empty:
            pass
        else:
            data_yoy_dict = yoy_growth(data)    
        
        # Generate AI-driven analysis
        if st.button("Get Analysis"):
            analysis_string = """Generate the analysis based on instruction\
                                that is delimited by triple backticks.\
                                instruction: ```{instruction_analyis}```
                                in the format enclosed in html tag <{format_analysis}\>\
                                """
            analysis_template = ChatPromptTemplate.from_template(analysis_string)

            format_analysis=f"""
            1.Historical Trends:
                Review of past trends

            2.Forecasted Trends:
                Predicted future trends

            3.Seasonality Analysis:
                Examining in genral seasonal patterns in the data(max,min sales months )
            
            4.Year-on-Year (YoY) Growth Analysis:
                Summarizes year-over-year growth 

            5Conclusion: 
                Conclusion based on the time series analysis"""

            instruction_analysis = f"""You are functioning as an AI data analyst.
            1. You will be analyzing three dictinoary: trend_dict,data_yoy_dict and seasonal_patter_dict.
            2. Trend_dict key represent scenario: Indicates if a data point is historical or forecasted and value\
                Trend: Indicates the trend of the data for a specific scenario.
            3. data_yoy_dict key represet year and value Indicates the percentage volume change compared to the previous year  
            4. Start the output as "Insight and Findings:" and report in point form.
            5. Summarizes the trend based on the Trend_dict
            6. Analyze the year on year growth based on the data_yoy_dict include the change percentage.
            7. Analyze the in general seasonality based on the seasonal_patter_dict
            8. The dictinoary: {trend_dict} for trend analysis,{data_yoy_dict} for year-on-year growth analysis and  {seasonal_patter_dict} for seasonlity analysis.
            9. Donot include the name of dict only generate inshits from using the dict data"""

            chat = ChatOpenAI(temperature=0.0, model=model, openai_api_key=openai_api_key)
            user_analysis = analysis_template.format_messages(instruction_analyis=instruction_analysis,format_analysis=format_analysis)

            with st.spinner('Generating...'):

                response = chat(user_analysis)
                
                # Apply styling to the output box with a chat icon
                st.markdown(
                    f'<div style="border: 2px solid red; padding: 10px; background-color: white; color: black;">'
                    f'<div style="font-size: 24px; color: red;">💬</div>' 
                    f'{response.content}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                    

        st.markdown("---")

    if __name__ == "__main__":
        main()

with tab1:
    st.markdown('<p style="color:red; font-size:30px; font-weight:bold;">About The App:</p>', unsafe_allow_html=True)
    st.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)   
    st.markdown("👋 Welcome to Sigmoid GenAI - Your Data Analysis APP!")
    st.write("This app is designed to help you analyze and visualize your data.")
    st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
    st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">👨‍💻  How to Use:</p>', unsafe_allow_html=True)
    st.write("1. Please enter your API key in side bar and click on the ENTER")
    st.write("2. From the top this page please select the required tab")
    st.write("3. Follow the instruction of that tab.")
    st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
    st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">Limitations ⚠️:</p>', unsafe_allow_html=True)
    st.write("Please note the following limitations:")
    st.write("- Active internet connection is required.")
    st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
 
 #Tab 3
with tab3:
    column_descriptions = {}
    def main():
        st.markdown('<p style="color:red; font-size:30px; font-weight:bold;">CodeAI:</p>', unsafe_allow_html=True)
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">👨‍💻 How to Use:</p>', unsafe_allow_html=True)
        st.markdown("""
        - 📂 Upload a CSV or Excel file containing your dataset.
        - 📝 Provide descriptions for each column of the dataset in the 'Column Descriptions' section.
        - 📂 Optionally, upload a CSV or Excel file containing column descriptions.
        - ❓ Ask questions about the dataset in the 'Ask a question about the dataset' section.
        - 🔍 Click the 'Get Answer' button to generate an answer based on your question.
        """)

        # Display limitations with emojis
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">Limitations ⚠️:</p>', unsafe_allow_html=True)
        st.markdown("""
        - The quality of AI responses depends on the quality and relevance of your questions.
        - Ensure that you have a good understanding of the dataset columns to ask relevant questions.
        """)   
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)

        # Upload the dataset file
        uploaded_file = st.file_uploader("Upload a CSV or Excel file (Dataset)", type=["csv", "xlsx"])
        st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">Head of the Dataset:</p>', unsafe_allow_html=True)
        
        df_user = pd.DataFrame()
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_user = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                    df_user = pd.read_excel(uploaded_file)
                
                # Display the first few rows of the dataset
                st.write(df_user.head())
                

                st.info("Please add column descriptions of your dataset")
                for col in df_user.columns:
                    col_description = st.text_input(f"Description for column '{col}':")
                    if col_description:
                        column_descriptions[col] = col_description
                    
                if st.button("Submit Descriptions"):
                    st.success("Descriptions submitted successfully!")
            except Exception as e:
                st.error(f"An error occurred while reading the dataset file: {e}")
                return

        # Optionally, upload column descriptions file
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        uploaded_desc_file = st.file_uploader("Upload a CSV or Excel file (Column Descriptions)", type=["csv", "xlsx"])
        if uploaded_desc_file is not None:
            try:
                if uploaded_desc_file.name.endswith('.csv'):
                    desc_df = pd.read_csv(uploaded_desc_file)
                elif uploaded_desc_file.name.endswith(('.xls', '.xlsx')):
                    desc_df = pd.read_excel(uploaded_desc_file)

                for index, row in desc_df.iterrows():
                    col_name = row['Column Name']
                    col_description = row['Description']
                    if col_name and col_description:
                        column_descriptions[col_name] = col_description

                st.success("Column descriptions loaded successfully!")
            except Exception as e:
                st.error(f"An error occurred while reading the column descriptions file: {e}")

    


        
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.markdown('<p style="color:red; font-size:25px; font-weight:bold;">Ask a question about the dataset:</p>', unsafe_allow_html=True)
        user_question = st.text_input(" ")
        user_question+= "call the function below in the same script with 'df_user"

        code_string ="""Generate the python code based on the user question\
            that is delimated by triple backticks\
                based on the instruction that is {instruction}.\
                    user question: ```{user_question}```\
                        """
        code_templete= ChatPromptTemplate.from_template(code_string)

        instruction =f"""1. You are functioning as an AI data analyst.
        2. Task: Respond to questions based on the provided dataset by giving code
        3. Dataset columns enclosed in square brackets {df_user.columns.tolist()}.
        4. Columns Description in dict format - {column_descriptions}.
        5. Provide code based on the user's question.
        6. Do not create any dummy or test dataset; call the function with DataFrame name: 'df_user'.
        7. Print result using 'st.write' for text or 'st.pyplot' for plots use plotly with white background for the plot
        8. Return the output in function form only.
        9. Provide all the code together.
        10. Only return the code; no explanations or extra text.
        11. Include code to suppress warnings.
        12. Do not include [assistant].
        13. Do not read any dataset; call the function with df_user.
        14. Return final output with st.write or st.pyplot.
        15. Only give the executable code.
        16. Code must start with 'def' and end with the function call.
        17. Do not enclose the code in triple backticks.
        18. Only give the executable line; no non-executable characters.
        19. Call the function below the response in the same script.
        20. Always call the function in the same script with 'df_user'"""

        user_message = code_templete.format_messages(instruction=instruction,user_question=user_question)
                
        chat2 = ChatOpenAI(temperature=0.0, model=model,openai_api_key=openai_api_key)
        if st.button("Get Answer"):
            if user_question:
                user_message = code_templete.format_messages(instruction=instruction,user_question=user_question)
                code = chat2(user_message)
                st.code(code.content)
                exec(code.content)
            else:
                st.warning("Not a valid question. Please enter a question to analyze.")
        
        # st.markdown('<p style="color:red; font-size:25px; font-weight:bold;">Code Execution Dashboard:</p>', unsafe_allow_html=True)
    
        # st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        # code_input = st.text_area("Enter your code here", height=200)
        # st.warning(("⚠️ If there is any non-executable line in generated code; please remove it"))
        
        # if st.button("Execute code"): 
        #     try:
        #         # Use exec() to execute the code
        #         exec(code_input)
        #     except Exception as e:
        #         st.error(f"An error occurred: {e}")

    # Check if the script is run as the main program
    if __name__ == "__main__":
        main()
with tab4:
    st.markdown('<p style="color:red; font-size:30px; font-weight:bold;">DocAI:</p>', unsafe_allow_html=True)
    st.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
    st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">👨‍💻  How to Use:</p>', unsafe_allow_html=True)
    st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
    st.markdown('1. **Upload an Article**: Click on the "Upload an article" button to upload a text (.txt) or PDF (.pdf) file containing the content you want to query.')

    st.markdown('2. **Enter your Question**: Enter your question or query in the "Enter your question" field. This question will be used to generate a response based on the uploaded content.')

    st.markdown('3. **Generate Response**: After uploading the file and entering the question, click the "Generate Response" button. This will trigger the response generation process.')
    st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
    # Limitations
    st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">Limitations ⚠️:</p>', unsafe_allow_html=True)
    st.markdown('1. **Supported File Formats**: Only text (.txt) and PDF (.pdf) file formats are supported for uploading. Other formats are not supported.')

    st.markdown('2. **Query Text Required**: You must enter a question or query in the "Enter your question" field. Without a question, you cannot generate a response.')

    st.markdown('3. **Response Time**: The response generation process may take some time, depending on the complexity of the query and the size of the uploaded file.')
    st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)

    def generate_response(uploaded_file, openai_api_key, query_text):
        # Load document if file is uploaded
        if uploaded_file is not None:
            documents = [uploaded_file.read().decode()]
            # Split documents into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.create_documents(documents)
            # Select embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            # Create a vectorstore from documents
            db = FAISS.from_documents(texts, embeddings)
            # Create retriever interface
            retriever = db.as_retriever()
            # Create QA chain
            qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
            return qa.run(query_text)
    def pdf_chat(uploaded_file,query):

        if uploaded_file is not None:
            pdf_reader = PdfReader(uploaded_file)

            text = ""
            for page in pdf_reader.pages:
                text+= page.extract_text()

            #langchain_textspliter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200,
                length_function = len
            )

            chunks = text_splitter.split_text(text=text)

            
            #store pdf name
            store_name = uploaded_file.name[:-4]
            
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl","rb") as f:
                    vectorstore = pickle.load(f)
                #st.write("Already, Embeddings loaded from the your folder (disks)")
            else:
                #embedding (Openai methods) 
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

                #Store the chunks part in db (vector)
                vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

                with open(f"{store_name}.pkl","wb") as f:
                    pickle.dump(vectorstore,f)
                
                #st.write("Embedding computation completed")

            #st.write(chunks)
            
            #Accept user questions/query

    
            #st.write(query)

            if query:
                docs = vectorstore.similarity_search(query=query,k=3)
                #st.write(docs)
                
                #openai rank lnv process
                llm = OpenAI(temperature=0,openai_api_key=openai_api_key)
                chain = load_qa_chain(llm=llm, chain_type= "stuff")
                
                with get_openai_callback() as cb:
                    response = chain.run(input_documents = docs, question = query)
                    
        return response
    def main():
        # File upload
        uploaded_file = st.file_uploader('Upload an article', type=['txt', 'pdf'])

        # Query text
        query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)
        query_text+= "Always give the complete answer"

        # Form input and query
        result = []

        if st.button('Generate Response'):  # Add a button to trigger response generation
            with st.spinner('Generating...'):
                if query_text:
                    if uploaded_file.type == 'text/plain':
                        response = generate_response(uploaded_file, openai_api_key, query_text)
                        result.append(response)
                    elif uploaded_file.type == 'application/pdf':
                        # Handle plain text file
                        response = pdf_chat(uploaded_file,query_text)
                        result.append(response)
                    else:
                        response = "Unsupported file format. Please upload a PDF or text file."


        if len(result):
            st.write(result[0])  # Display the response if there is a result

    if __name__ == "__main__":
        main()
with tab5:
    llm_model = "gpt-3.5-turbo-0301"
    chat = ChatOpenAI(temperature=1, model=llm_model, openai_api_key=openai_api_key)
    df_final = pd.read_csv("Data/df_final_with_name2.csv")
    existing_user = df_final["user_id"].unique()
    # Function to generate personalized messages for new users
    def personlized_message_new_user(template, style, welcome_offer, best_selling_pro, user_data,instruction_existing):
        prompt_template = ChatPromptTemplate.from_template(template)
        customer_messages = prompt_template.format_messages(
            style=style,
            welcome_offer=welcome_offer,
            best_selling_product=best_selling_pro,
            user_data=user_data,
            instruction_existing=instruction_existing

        )
        customer_response = chat(customer_messages)
        return customer_response.content

    # Function to generate personalized messages for existing users
    def personlized_message_existing_user(template, style, Existing_user_data, Rec_product, Offers_and_promotion,instruction_existing):
        prompt_template = ChatPromptTemplate.from_template(template)
        customer_messages = prompt_template.format_messages(
            style=style,
            Existing_user_data=Existing_user_data,
            Offers_and_promotion=Offers_and_promotion,
            Rec_product=Rec_product,
            instruction_existing=instruction_existing
        )
        customer_response = chat(customer_messages)
        return customer_response.content

    # Define custom colors
    primary_color = "#3498db"  # Blue
    secondary_color = "#2ecc71"  # Green
    background_color = "#f0f3f6"  # Light Gray
    text_color = "#333333"  # Dark Gray

    # Apply custom styles
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: {background_color};
            color: {text_color};
        }}
        .sidebar .sidebar-content {{
            background: {primary_color};
            color: white;
        }}
        .widget-label {{
            color: {text_color};
        }}
        .stButton.button-primary {{
            background: {secondary_color};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Recommendation part
    st.markdown("### Personalized Welcome Message")
    st.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
    with st.form("login_form"):
        user_id = st.text_input("User ID")
        user_name = st.text_input("Your Name")
        submitted = st.form_submit_button("Login")

    if submitted:
        with st.spinner('Generating...'):
            if user_id in existing_user:
                if user_id:
                    recommended_products = recommend_products(user_id, df_final)

                    if recommended_products:
                        offer = df_final[df_final["product_id"].isin(recommended_products)]["offers"].unique().tolist()
                        offers = [offer[0], offer[-1]]
                        item_cart=["Tanqueray Sterling Vodka", "7 Crown Appl", "Ursus Punch Vodka"]
                        Existing_user_data = {"Name": user_name, "Existing Items in the cart":item_cart }
                        Rec_product = recommended_products
                        Offers_and_promotion = offers
                        existing_user = personlized_message_existing_user(template_string, customer_style, Existing_user_data, Rec_product, Offers_and_promotion,instruction_existing)
                        with st.chat_message("user"):
                            st.write(existing_user)
                    else:
                        st.warning("Please enter the right details.")
            else:
                new_message = personlized_message_new_user(template_string_new, customer_style, welcome_offer, best_selling_product, user_name,instruction_existing)
                with st.chat_message("user"):
                    st.write(new_message)
with tab6:


    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    #API_URL = "https://api-inference.huggingface.co/models/minimaxir/sdxl-wrong-lora"

    def main():
        st.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align: center;'>"
            "<h2 style='color: #3366FF;'>🎨 Welcome! I am the ImageGenie! 🧞‍♂️</h2>"
            "<p style='color: #FF5733;'>I generate vibrant images based on your textual descriptions.</p>"
            "</div>",
            unsafe_allow_html=True
        )
        
        st.markdown("<p style='color: #3366FF; font-size: 18px; text-align: center;'>Ask & Visualize 🖼️</p>", unsafe_allow_html=True)
        
        # Apply CSS to style the horizontal lines
        st.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
        api_key = st.text_input("Enter Your HuggingFace API key", type="password")
        
        # Submit button to check the API key
        if st.button("Submit"):
            if api_key:
                st.success("API Detected")
            else:
                st.warning("Please enter your API key")
        API_TOKEN = api_key
        headers = {"Authorization": f"Bearer {API_TOKEN}"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.content
        
        user_input = st.text_area("Enter a Description", "")

        if st.button("Generate Image"):
            if user_input:
                st.info("Generating image...")
                payload = {
                    "inputs": user_input,
                }
                image_bytes = query(payload)
                image = Image.open(io.BytesIO(image_bytes))
                st.image(image, caption="Generated Image", use_column_width=True)
                st.success("Image generated successfully!")
                st.download_button("Download Image", image_bytes, file_name="generated_image.png")
            else:
                st.warning("Please enter a text prompt.")

    if __name__ == "__main__":
        main()

