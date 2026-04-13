import pickle
import os
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="Disease Prediction", layout="wide")

# --- 2. INITIALIZE HISTORY ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 3. CUSTOM STYLE (100% ORIGINAL DESIGN RESTORED) ---
def add_custom_style():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1674702727317-d29b2788dc4a?fm=jpg&q=60&w=3000&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8bWVkaWNhbCUyMHdhbGxwYXBlcnxlbnwwfHwwfHx8MA%3D%3D");
            background-attachment: fixed; background-size: cover;
        }}
        .stApp::before {{
            content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            background-color: rgba(0, 0, 0, 0.75); backdrop-filter: blur(10px); z-index: -1;
        }}
        [data-testid="stSidebar"] {{
            background-color: rgba(255, 255, 255, 0.05) !important;
            backdrop-filter: blur(20px) !important; border-right: 1px solid rgba(0, 212, 255, 0.2) !important;
        }}
        
        /* Sidebar Footer & Info Box Font Size Fix */
        
        /* st.info ke andar ka text chota karne ke liye */
        [data-testid="stSidebar"] .stAlert p {{
            font-size: 13px !important;
            font-weight: normal !important;
            line-height: 1.2 !important;
            color: #FFFFFF !important;
        }}

        /* st.caption ka font size chota karne ke liye */
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
            font-size: 11px !important;
            color: rgba(255, 255, 255, 0.6) !important;
            margin-top: 5px !important;
            text-align: center;
        }}
        .sidebar-title {{
            color: #00D4FF !important; font-size: 40px !important; font-weight: 800 !important;
            text-align: center; text-transform: uppercase; text-shadow: 0px 0px 15px rgba(0, 212, 255, 0.4);
        }}
        h1, h2, h3, p, label {{ color: #FFFFFF !important; font-size: 22px !important; font-weight: bold !important; text-shadow: 2px 2px 5px rgba(0,0,0,1); }}
        .stApp h1 {{ font-size: 50px !important; color: #00D4FF !important; }}
        
        .stTextInput > div > div > input, .stNumberInput > div > div > input {{
            background-color: rgba(255, 255, 255, 0.95) !important; color: #000000 !important;
            border-radius: 12px !important; font-size: 20px !important; font-weight: bold !important; padding: 12px !important;
        }}
        
        div.stButton > button {{
            background-color: red !important; color: #000000 !important;
            font-size: 20px !important; font-weight: bold !important; border-radius: 10px !important;
            border: 2px solid #FFFFFF !important; padding: 10px 25px !important;
        }}
        
        .result-box-positive {{ background-color: #FF3131 !important; color: #FFFFFF !important; padding: 30px; border-radius: 15px; border: 5px solid #990000; text-align: center; font-size: 38px; font-weight: 1000; box-shadow: 0px 10px 30px rgba(0,0,0,0.5); margin-top: 40px; width: 100%; display: block; }}
        .result-box-negative {{ background-color: #28a745 !important; color: #FFFFFF !important; padding: 30px; border-radius: 15px; border: 5px solid #004d00; text-align: center; font-size: 38px; font-weight: 1000; box-shadow: 0px 10px 30px rgba(0,0,0,0.5); margin-top: 40px; width: 100%; display: block; }}
        
        .stTooltipIcon {{ filter: brightness(0) invert(1) !important; background-color: rgba(255, 255, 255, 0.3) !important; border-radius: 50% !important; padding: 3px !important; }}
        div[data-testid="stTooltipContent"] {{ background-color: #FF0000 !important; color: white !important; font-size: 8px !important; font-weight: bold !important; }}
        
        .stExpander {{ background-color: transparent !important; margin-top: 25px !important; }}
        .stExpander details summary {{ background-color: #FFFFFF !important; border: 3px solid #00D4FF !important; border-radius: 12px !important; padding: 12px !important; }}
        .stExpander details summary p {{ color: #000000 !important; text-shadow: none !important; font-size: 20px !important; }}
        .stExpander div[data-testid="stExpanderDetails"] {{ background-color: #FFFFFF !important; border-radius: 0 0 12px 12px !important; padding: 20px !important; }}
        .stExpander div[data-testid="stExpanderDetails"] p, td, th {{ color: #000000 !important; text-shadow: none !important; font-size: 16px !important; }}
        </style>
        """, unsafe_allow_html=True
    )

add_custom_style()

# --- 4. LOAD MODELS ---
working_dir = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(working_dir, 'saved_models')
try:
    diabetes_model = pickle.load(open(os.path.join(model_folder, 'diabetes_modelRF.sav'), 'rb'))
    diabetes_scaler = pickle.load(open(os.path.join(model_folder, 'diabetes_scalerRF.sav'), 'rb'))
    heart_model = pickle.load(open(os.path.join(model_folder, 'heart_disease_model_final.sav'), 'rb'))
    heart_scaler = pickle.load(open(os.path.join(model_folder, 'heart_scaler_final.sav'), 'rb'))
    kidney_model = pickle.load(open(os.path.join(model_folder, 'kidney_model_final.sav'), 'rb'))
    kidney_scaler = pickle.load(open(os.path.join(model_folder, 'kidney_scaler_final.sav'), 'rb'))
except Exception as e:
    st.error(f"Error loading models: {e}")

# --- 5. SIDEBAR MENU & HISTORY ---
with st.sidebar:
    st.markdown('<p class="sidebar-title">🧬HEALTH AI PRO</p>', unsafe_allow_html=True)
    st.markdown("---")
    selected = option_menu(None, ['Diabetes Prediction', 'Heart Disease Prediction', 'Kidney Disease Prediction'], 
                           icons=['activity', 'heart-pulse', 'capsule-pill'], default_index=0,
                           styles={"nav-link": {"font-size": "16px", "text-align": "left", "color": "red", "font-weight": "1000"},
                                   "nav-link-selected": {"background-color": "rgba(0, 212, 255, 0.2)", "border": "1px solid #00D4FF", "color": "#00D4FF"}})

    with st.expander("📂 Prediction History"):
        if not st.session_state.history: st.write("No records yet.")
        else:
            for entry in reversed(st.session_state.history):
                st.write(f"**{entry['type']}** ({entry['time']})")
                st.write(f"Result: {entry['result']}")
                st.json(entry['params'])
                st.markdown("---")
            if st.button("Clear All History"): st.session_state.history = []; st.rerun()

    st.info("💡 **Tip:** Ensure you enter lab values accurately from clinical reports for best results.")
    st.caption("Developed by Sudhir /n MSc Final Year Project 2026")

# --- 6. DIABETES PAGE ---
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    st.info("Fill the details below. Hover over the '?' icons for helpful hints.")
    # Default values logic
    default_vals = {
        'preg': "0", 'gluc': "85", 'bp': "68", 
        'skin': "23", 'ins': "0", 'bmi': "20.1", 
        'dpf': "0.471", 'age': "21"
    }
    
    if 'preg' not in st.session_state:
        for k, v in default_vals.items(): st.session_state[k] = v

    def reset_inputs():
        for k, v in default_vals.items(): st.session_state[k] = v

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('1. Number of Pregnancies', key='preg',
                                    help="Total number of times pregnant.")
        SkinThickness = st.text_input('4. Skin Thickness (mm)', key='skin',
                                      help="Triceps skin fold thickness in mm. Usually 10-30mm in healthy adults. If unknown, leave it at 23 (average).")
    with col2:
        Glucose = st.text_input('2. Glucose Level', key='gluc',
                                help="Blood glucose concentration (2 hours after a meal). Ideal is below 140 mg/dL.")
        Insulin = st.text_input('5. Insulin Level (mu U/ml)', key='ins',
                                help="2-Hour serum insulin level. Average post-meal insulin is around 79 mIU/L. If you haven't done this test, leave it at 0 (model will handle it as a missing value).")
    with col3:
        BloodPressure = st.text_input('3. Blood Pressure (mm Hg)', key='bp',
                                      help="Diastolic blood pressure. Normal range is 60-80 mm Hg.")
        BMI = st.text_input('6. BMI Score', key='bmi',
                            help="Body Mass Index (Weight/Height ratio). Healthy range is 18.5 - 24.9.")

    col1, col2 = st.columns(2)
    with col1: 
        DiabetesPedigreeFunction = st.text_input('7. Family History Score', key='dpf',
                                                 help="Pedigree Function: A score that calculates diabetes risk based on your family history.('Enter 0.47 if you don't know your score. Higher values (above 1.0) mean high diabetes history in your family.')")
    with col2: 
        Age = st.text_input('8. Age', key='age',
                            help="Current age of the person in years.")

    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button('Test Result'):
        try:
            inputs = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            scaled = diabetes_scaler.transform([inputs])
            pred = diabetes_model.predict(scaled)
            conf = max(diabetes_model.predict_proba(scaled)[0]) * 100
            res = "Diabetic" if pred[0] == 1 else "Not Diabetic"
            st.markdown(f'<div class="{"result-box-positive" if pred[0]==1 else "result-box-negative"}">{res}<br><span style="font-size: 20px;">Confidence: {conf:.2f}%</span></div>', unsafe_allow_html=True)
            st.session_state.history.append({"type": "Diabetes", "time": datetime.now().strftime("%H:%M:%S"), "result": res, "params": {"Preg": Pregnancies, "Gluc": Glucose, "BP": BloodPressure, "Skin": SkinThickness, "Ins": Insulin, "BMI": BMI, "DPF": DiabetesPedigreeFunction, "Age": Age}})
        except Exception as e: st.error(f"Error: {e}")
    if c2.button('Reset'): st.rerun()

    with st.expander("📊 View Diabetes Clinical Reference Ranges & Descriptions"):
        st.markdown("""
        | Parameter | Description | Normal Range | Borderline / High Risk |
        | :--- | :--- | :--- | :--- |
        | **Glucose** | Amount of sugar in blood 2 hours after a meal; primary marker for diabetes. | < 140 mg/dL | > 200 mg/dL (Diabetic) |
        | **Blood Pressure** | Diastolic pressure; high BP often co-exists with diabetes and damages vessels. | 60 - 80 mmHg | > 90 mmHg (High) |
        | **Skin Thickness** | Triceps fat fold; used as an estimate of total body fat and subcutaneous storage. | 10 - 25 mm | > 30 mm (High Fat) |
        | **Insulin** | Hormone that helps cells absorb sugar; high levels suggest 'Insulin Resistance'. | 16 - 166 mIU/L | Very High (Resistance) |
        | **BMI** | Body Mass Index; high BMI (excess weight) is a major risk factor for Type 2 Diabetes. | 18.5 - 24.9 | > 30 (Obese) |
        | **Family History Score/Pedigree Score** | A genetic score based on family history; higher means stronger hereditary risk. | 0.1 - 0.5 | > 1.0 (High Genetic Risk) |
        """)

# --- 7. HEART DISEASE PAGE ---
elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    st.info("Fill the details below. Hover over the '?' icons for helpful hints.")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('1. Age', min_value=1, max_value=110, value=40,
                              help="Enter your current age in years.")
    with col2:
        sex = st.selectbox('2. Gender', ['Male', 'Female'],
                           help="Select your biological gender.")
    with col3:
        cp = st.selectbox('3. Chest Pain Type', 
                          ['Low/Typical Pain', 'Moderate/Atypical Pain', 'Non-Heart Related Pain', 'Severe/No Specific Pain'],
                          help="Choose 'Severe' for sudden heavy pressure, or 'Low' for normal tightness.")

    col1, col2, col3 = st.columns(3)
    with col1:
        chol = st.number_input('4. Serum Cholesterol', min_value=50, max_value=600, value=200,
                               help="Ideal range is below 200 mg/dL. High levels increase risk.")
    with col2:
        thalch = st.number_input('5. Max Heart Rate Achieved', min_value=50, max_value=250, value=150,
                                 help="The highest pulse rate recorded during physical stress.")
    with col3:
        exang = st.selectbox('6. Pain During Exercise?', ['No', 'Yes'],
                             help="Select 'Yes' if activity like walking or climbing causes chest pain.")

    col1, col2, col3 = st.columns(3)
    with col1:
        oldpeak = st.number_input('7. Physical Stress Score', min_value=0.0, max_value=10.0, value=0.0, step=0.1,
                                  help="Stress score from ECG/TMT report. Lower values (0-1) are safer.")
    with col2:
        ca = st.selectbox('8. Blocked Major Vessels (0-3)', [0, 1, 2, 3],
                          help="Number of main arteries identified as blocked in reports.")
    with col3:
        thal = st.selectbox('9. Heart Blood Flow', 
                            ['Normal Flow', 'Permanent Blockage', 'Temporary Blockage during Stress'],
                            help="Flow of blood to heart muscles during stress test.")

    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button('Heart Test Result'):
        try:
            s_v, e_v = (1 if sex == 'Male' else 0), (1 if exang == 'Yes' else 0)
            cp_m = {'Severe/No Specific Pain': 0, 'Moderate/Atypical Pain': 1, 'Non-Heart Related Pain': 2, 'Low/Typical Pain': 3}
            th_m = {'Permanent Blockage': 0, 'Normal Flow': 1, 'Temporary Blockage during Stress': 2}
            in_data = [age, s_v, cp_m[cp], chol, thalch, e_v, oldpeak, ca, th_m[thal]]
            scaled = heart_scaler.transform([in_data])
            pred = heart_model.predict(scaled)
            conf = max(heart_model.predict_proba(scaled)[0]) * 100
            res = "Positive" if pred[0] >= 1 else "Negative"
            st.markdown(f'<div class="{"result-box-positive" if pred[0]>=1 else "result-box-negative"}">Heart Disease: {res}<br><span style="font-size: 20px;">Confidence: {conf:.2f}%</span></div>', unsafe_allow_html=True)
            st.session_state.history.append({"type": "Heart", "time": datetime.now().strftime("%H:%M:%S"), "result": res, "params": {"Age": age, "Sex": sex, "CP": cp, "Chol": chol, "MaxRate": thalch, "Exang": exang, "Oldpeak": oldpeak, "Vessels": ca, "Thal": thal}})
        except Exception as e: st.error(f"Error: {e}")
    if c2.button('Reset'): st.rerun()

    with st.expander("📊 View Heart Health Reference Ranges & Descriptions"):
        st.markdown("""
        | Parameter | Description | Healthy / Normal | Concern / High Risk |
        | :--- | :--- | :--- | :--- |
        | **Chest Pain (CP)** | Type of pain; 'Typical' is heart-related, while 'Asymptomatic' can be a silent risk. | Typical Angina (0) | Asymptomatic (3) - High Risk! |
        | **Cholesterol** | Fatty substance in blood; high levels can clog arteries and lead to heart attacks. | < 200 mg/dL | > 240 mg/dL (High) |
        | **Max Heart Rate** | Highest pulse achieved during stress; a low max rate may indicate a weak heart. | 150 - 190 bpm | < 120 bpm (Weak) |
        | **ST Depression** | Measured from ECG; 'Oldpeak' shows stress on the heart during physical activity. | 0.0 - 0.5 | > 2.0 (Sign of Blockage) |
        | **Major Vessels** | Number of main blood vessels (0-3) seen as colored/blocked in a fluoroscopy test. | 0 (Clear) | 1, 2, or 3 (Blocked) |
        | **Blood Flow (Thal)** | A stress test result showing how well blood flows to your heart muscles. | Normal Flow | Fixed/Reversible Defect |
        """)

# --- 8. KIDNEY DISEASE PAGE ---
elif selected == 'Kidney Disease Prediction':
    st.title('Kidney Disease Prediction using ML')
    st.info("Fill the details below. Hover over the '?' icons for helpful hints.")

    # Grid 1: Primary Lab Values
    col1, col2, col3 = st.columns(3)
    with col1:
        bp = st.number_input('1. Blood Pressure', min_value=40, step=5, value=80,
                             help="Diastolic BP (bottom number). Normal is 60-80 mmHg. High BP is a leading cause of kidney damage.")
    with col2:
        al = st.selectbox('2. Albumin', [0, 1, 2, 3, 4, 5],
                          help="Protein in urine. '0' is normal. High levels indicate the kidney's filters are leaking protein.")
    with col3:
        rbc = st.selectbox('3. Red Blood Cells', ['normal', 'abnormal'],
                           help="Presence of blood in urine. 'Abnormal' can signal kidney inflammation or stones.")

    # Grid 2: Infection & Glucose
    col1, col2, col3 = st.columns(3)
    with col1:
        pc = st.selectbox('4. Pus Cell', ['normal', 'abnormal'],
                          help="Indicates urinary tract infection (UTI). Healthy result should be 'Normal'.")
    with col2:
        pcc = st.selectbox('5. Pus Cell Clumps', ['notpresent', 'present'],
                           help="Clusters of infection cells. Usually 'notpresent' unless there is a severe infection.")
    with col3:
        bgr = st.number_input('6. Blood Glucose Random', value=121.0,
                              help="Random Blood Sugar. Levels above 140 mg/dL can damage the small blood vessels in kidneys.")

    # Grid 3: Critical Kidney Markers
    col1, col2, col3 = st.columns(3)
    with col1:
        bu = st.number_input('7. Blood Urea', value=36.0,
                             help="Waste product in blood. Normal: 7-20 mg/dL. High levels suggest the kidneys aren't filtering waste properly.")
    with col2:
        sc = st.number_input('8. Serum Creatinine', value=1.2,
                             help="MOST CRITICAL: Waste from muscle wear. Normal: 0.7-1.3. A rise is a direct sign of decreased kidney function.")
    with col3:
        pcv = st.number_input('9. Packed Cell Volume (PCV)', value=44.0,
                              help="Percentage of red cells in blood. Normal: 36%-50%. Low levels often occur in Chronic Kidney Disease.")

    # Grid 4: Blood Count & Lifestyle
    col1, col2, col3 = st.columns(3)
    with col1:
        rc = st.number_input('10. Red Blood Cell Count', value=5.2,
                             help="Kidneys produce a hormone (EPO) that creates RBCs. Low counts are common in kidney patients.")
    with col2:
        htn = st.selectbox('11. Hypertension', ['no', 'yes'],
                           help="Do you have a history of High Blood Pressure? It is both a cause and a symptom of kidney disease.")
    with col3:
        dm = st.selectbox('12. Diabetes Mellitus', ['no', 'yes'],
                          help="Do you have Diabetes? High sugar levels are the #1 cause of kidney failure globally.")

    # Grid 5: Physical Signs
    col1, col2, col3 = st.columns(3)
    with col1:
        appet = st.selectbox('13. Appetite', ['good', 'poor'],
                             help="Loss of appetite occurs when toxins build up in the body due to poor kidney filtration.")
    with col2:
        pe = st.selectbox('14. Pedal Edema', ['no', 'yes'],
                          help="Swelling in feet or ankles. Happens when kidneys fail to remove extra salt and fluid from the body.")
    with col3:
        ane = st.selectbox('15. Anemia', ['no', 'yes'],
                           help="Chronic fatigue or low hemoglobin. Often caused by the kidney's inability to support red blood cell production.")

    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button('Kidney Test Result'):
        try:
            m = {'abnormal': 0, 'normal': 1, 'present': 1, 'notpresent': 0, 'yes': 1, 'no': 0, 'good': 0, 'poor': 1}
            u_in = [bp, al, m[rbc], m[pc], m[pcc], bgr, bu, sc, pcv, rc, m[htn], m[dm], m[appet], m[pe], m[ane]]
            scaled = kidney_scaler.transform([u_in])
            pred = kidney_model.predict(scaled)
            conf = max(kidney_model.predict_proba(scaled)[0]) * 100
            res = "Positive" if pred[0] == 1 else "Negative"
            st.markdown(f'<div class="{"result-box-positive" if pred[0]==1 else "result-box-negative"}">CKD Status: {res}<br><span style="font-size: 20px;">Confidence: {conf:.2f}%</span></div>', unsafe_allow_html=True)
            st.session_state.history.append({"type": "Kidney", "time": datetime.now().strftime("%H:%M:%S"), "result": res, "params": {"BP": bp, "Alb": al, "RBC": rbc, "PC": pc, "PCC": pcc, "Sugar": bgr, "Urea": bu, "SC": sc, "PCV": pcv, "RC": rc, "HTN": htn, "DM": dm, "App": appet, "PE": pe, "Ane": ane}})
        except Exception as e: st.error(f"Error: {e}")
    if c2.button('Reset'): st.rerun()

    with st.expander("📊 View Kidney Function Reference Ranges & Descriptions"):
        st.markdown("""
        | Parameter | Description | Normal Range | CKD Indicator (High Risk) |
        | :--- | :--- | :--- | :--- |
        | **Serum Creatinine** | Waste from muscle wear; best indicator of kidney filter health. | 0.7 - 1.3 mg/dL | > 1.5 mg/dL (Critical) |
        | **Blood Urea** | Waste product from protein breakdown that kidneys should remove. | 7 - 20 mg/dL | > 40 mg/dL (High) |
        | **Albumin** | A protein that stays in blood; leaking into urine means filter damage. | 0 (Negative) | 1+ to 5+ (Proteinuria) |
        | **Packed Cell Volume (PCV)** | The percentage of red cells in blood; low levels indicate Anemia. | 36% - 50% | < 30% (Anemia Risk) |
        | **Red Blood Cell Count** | Total RBCs; low count happens when kidneys fail to produce EPO hormone. | 4.5 - 5.9 M/mcL | < 4.0 M/mcL |
        | **Blood Glucose Random** | Sugar level; chronic high sugar is the #1 cause of kidney failure. | 70 - 140 mg/dL | > 200 mg/dL |
        | **Pedal Edema** | Swelling in feet/ankles caused by the body retaining salt and water. | No Swelling | Yes (Fluid Retention) |
        """)
