import streamlit as st
import pandas as pd
import pickle 


load_classifier,load_Testing_Accuracy,load_classification_report,load_scaler=pickle.load(open(r"C:\Users\Adhi Ganapathy\Documents\Python_ws\SVM 24082025\Loan Predication\Model\Loan_Prediction.pkl",'rb'))

def load_data():
    df = pd.read_csv(r"C:\Users\Adhi Ganapathy\Documents\Python_ws\SVM 24082025\Loan Predication\train_u6lujuX_CVtuZ9i (1).csv")

    df=df.drop(['Loan_Status'],axis=1)
    return df


st.sidebar.title("Loan Status Prediction")
page =st.sidebar.radio("Go To",["Overview","Model Evaluation","Prediction"])

if page == "Overview":
    st.title("📊 Dataset Overview")
    df=load_data()
    st.write(df.head())

elif page == "Model Evaluation":
    st.title("📈 Model Evaluation")
    

    # Display in Streamlit
    st.write("Classification Report:")
    st.text(load_classification_report)  # use st.text to preserve formatting
    st.write("Accuracy Score:", f'{load_Testing_Accuracy:.2%}')

elif page=="Prediction":
    st.title("💰Prediction")
    
    
    
    col1, col2 = st.columns(2)


   
    Credit_History_map={"Good":1.0,"Bad":0.0}
    education_map={"Graduate":1,"Not Graduate":0}
    gender_map={"Male":1,"Female":0}
    Self_Employed_map={"Yes":1,"No":0}
    Married_map={"Yes":1,"No":0}
    property_area_map={"Urban":0, "Semiurban":1, "Rural":2}

    with col1:
        Name = st.text_input("Applicant Name:")
        Gender = st.selectbox("Gender",["Male","Female"])
        education = st.selectbox("Education",["Graduate","Not Graduate"])
        self_employed = st.selectbox("Self Employed",["Yes","No"])
        Married = st.selectbox("Married",["Yes","No"])
        Credit_History=st.selectbox("Credit History",["Good","Bad"])
        
    

    
    
    with col2:
          Married_val=Married_map[Married]
          gender_val=gender_map[Gender]
          education_val=education_map[education]
          Credit_History_val=Credit_History_map[Credit_History]
          applicant_income = st.number_input("Applicant Income:", min_value=0, step=0)
          coapplicant_income = st.number_input("Coapplicant Income:", min_value=0, step=0)
          dependents = st.number_input("No of Dependents:", min_value=0, step=1)
          self_employed_val=Self_Employed_map[self_employed]
          loan_amount = st.number_input("Enter Loan Amount (in rupees):", min_value=0, step=1)
          loan_term = st.number_input("Enter Loan Term (in days):", min_value=0, step=1)
          property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
          property_area_val=property_area_map[property_area]
  
    
    features = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
    X_sample = pd.DataFrame([[gender_val,Married_val, dependents ,education_val,self_employed_val,applicant_income,coapplicant_income,loan_amount,loan_term,Credit_History_val,property_area_val]],columns=features)
    tranformed_input = load_scaler.transform(X_sample) 
    prediction = load_classifier.predict(tranformed_input)[0]
   
    

    if st.button("Predict"):
        
        if prediction == 1:
            st.success("✅ Loan Approved")
            st.balloons()
        else:
            st.error("❌ Loan Rejected")
            
       

        