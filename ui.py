import pandas as pd
import numpy as np
import pickle
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_icon="🛡️",page_title="Regulatory Credit Assessment",layout='wide')
# --- CSS STYLING ---
st.markdown("""
    <style>
    .stSelectbox, .stTextInput {
        border-radius: 10px;
        padding: 10px;
    }
    div.stButton > button:first-child {
        background-color: #004d99; /* Banking Blue */
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px 30px;
        width: 100%;
        border: none;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
    }
    div.stButton > button:hover {
        background-color: #003366;
        color: white;
        border: 2px solid #66b3ff;
    }
    .result-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_artifact():
    with open('credit_score.pkl', 'rb') as f:
        artifact = pickle.load(f)
    return artifact

data = get_artifact()

model = data['model']
features = data['features']
woe_map = data['woe_map']
binning_process = data['binning_process']
params = data['scorecard_params']

def woe_value(feature,input):
    mapping = woe_map[feature]
    if feature in data['cat_cols']:
        try:
            return mapping[mapping['Bin'] == input]['WoE'].values[0]
        except IndexError:
            return mapping['WoE'].mean()
    else:
        edges = binning_process[feature]

        bin = pd.cut([float(input)],bins = edges,include_lowest = True)[0]
        try:
            match = mapping[mapping['Bin'].astype(str) == str(bin)]
            return match['WoE'].values[0]
        except:
            if float(input)>edges[-1]: 
                return mapping.iloc[-1]['WoE']  #maximum bin
            else:
                return mapping.iloc[0]['WoE'] #least bin

def gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x' : [0,1],'y' : [0,1]},
        title = {'text': "Credit Score Risk",
                 'font': {'size': 32,'color':"#3E434C" }},
        number={
            'font': {'size': 44, 'color': score_color(score),'family': "Inter, sans-serif"}},
        gauge = {
            'axis': {'range': [300, 850], 'tickwidth': 1,'tickcolor': "#6b7280",
                'tickfont': {'size': 10}},
            'bar' : {'color': "#112252",
                'thickness': 0.25},
            'bgcolor' : "white",
            'borderwidth' : 2,
            'bordercolor' : "gray",
            'steps' : [
                {'range': [300,520],'color':"#c44040" },
                {'range': [520,560],'color':"#ee9219"},
                {'range': [560,620],'color':"#476dec"},
                {'range': [620,850],'color':"#4CAF50"}],
            'threshold' : {
                'line' : {'color' : "#12244C",'width' : 5},
                'thickness' : 0.75,
                'value' : score
            }
        }
    ))
    fig.update_layout(height=350)
    return fig

def risk_label(score):
    if score >= 620:
        return ("Low Risk Applicant")
    elif score >= 560:
        return ("Moderate Risk Applicant")
    elif score >= 520:
        return ("High Risk Applicant")
    else:
        return ("Very High Risk Applicant")
    
def score_color(score):
    if score >= 620:
        return "#166534" 
    elif score >= 560:
        return "#1e40af"   
    elif score >= 520:
        return "#b45309"  
    else:
        return "#991b1b"  

with st.sidebar:
    st.title("Risk Engine Insights")
    st.info("This model uses Logistic Regression calibrated to a FICO-scale (300-850).")
    
    st.markdown("### Score Legend")
    st.markdown("""
    - **620+**: Approved (Low Risk)
    - **560-619**: Manual Review
    - **520-559**: High Risk
    - **< 520**: Declined
    """)
    st.markdown("---")
    st.caption("v1.0.0 | Basel II Compliant")

st.title("🏦 Regulatory Credit Scorecard")
st.markdown("Enter applicant details to generate a **Basel-Compliant** Risk Score.") 

user_input = {}
with st.container():
    col1,col2 = st.columns(2,gap='large')
    for i,col in enumerate(features):
        label = col.replace('_', ' ').title()
        is_cat = col in data['cat_cols']

        with (col1 if i % 2 == 0 else col2):
            if is_cat:
                options = woe_map[col]['Bin'].tolist()
                user_input[col] = st.selectbox(f"Select {col}",options) #Select box to select the categorical features
            else:
                val = st.text_input(f"Enter {col}", value="0") #number input box
                user_input[col] = float(val) if val.strip() != "" else 0.0
st.markdown("---")
b1,b2 = st.columns(2)
with b1:
    submitted = st.button("Calculate Risk Score",type="primary")
if submitted:
    woe_vec = []
    for col in features:
        val = user_input[col]
        woe = woe_value(col,val)
        woe_vec.append(woe)

        #probability
    woe_arr = np.array(woe_vec).reshape(1,-1)
        #dataframe
    woe_df = pd.DataFrame(woe_arr,columns=features)
    probability = model.predict_proba(woe_df)[:,1][0]

    #score
    factor = params['factor']
    offset = params['offset']
    odds = (1-probability)/probability #considering probability as a person to default
    score = offset + (factor * np.log(odds))
    sc_final =  int(np.clip(score,300,850))

    st.markdown("### 📊 Assessment Result")
    c1,c2 = st.columns([1,2])
    
    c1.metric("Applicant Credit Score", f"{sc_final}")
    with c1:
        if sc_final >= 620: color = "green"
        elif sc_final >= 560: color = "blue"
        elif sc_final >= 520: color = "orange"
        else: color = "red"
        st.markdown(f"""
            <div class="result-card">
                <p style="font-size:18px; color:gray; margin-bottom:0;">Applicant Score</p>
                <h1 style="font-size:72px; color:{color}; margin:0;">{sc_final}</h1>
                <p style="font-size:16px;">Probability of Default: <b>{probability:.2%}</b></p>
            </div>
        """, unsafe_allow_html=True)

    # threshold decided by looking at the KDE plot
    st.markdown("---")
    if sc_final>=620:
        st.success(f"**APPROVED:** The applicant shows a strong probability to repay" \
                    " their entire loan with low risk of default")
    elif sc_final >= 560:
        st.info(f"**MANUAL REVIEW:** The applicant needs to be reviewed once manually" \
                    " once but there is a good chance that the applicant won't default and repay")
    elif sc_final >= 520:
        st.warning(f"**HIGH RISK APPLICANT:** The applicant is risky and has a high" \
                    " chance to default.The applicant need to scrutinize thoroughly if" \
                    " issuing the loan")
    else:
        st.error(f"**DECLINED:**Applicant falls below risk threshold.")
    with c2:
        st.plotly_chart(gauge(sc_final),use_container_width=True)
        st.markdown(
                    f"<h4 style='text-align:center; color:#374151'>{risk_label(sc_final)}</h4>",
                    unsafe_allow_html=True
                    )