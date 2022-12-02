import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from simulator import KineticsCalculator, ModelSimulator, pk_model

# ======================================================================
# SIDEBAR ==============================================================
# ======================================================================
st.sidebar.write("# General Setup")
col1, col2 = st.sidebar.columns(2)
# 0. Simulation random seed
seed = col1.number_input("Random Seed:", min_value=1, value=123)
np.random.seed(seed)
# 1. Dose
dose = col2.number_input("Dose (mg)", min_value=0.1, value=15.)
# 2. Dosing interval
dosing_interval = col1.number_input("Dosing Interval (Days):", min_value=0.5, value=7.)
# 3. #Doses
num_dose = col2.number_input("Number of Doses:", min_value=1, value=8)
# 4. Ref Lines
ref_high = col1.number_input("High Ref. Level (ng/mL):", value=4665.)
ref_low = col2.number_input("Low Ref. Level (ng/mL):", value=1680.)


st.sidebar.write("# Model Params")
# 0 DEPOT COMPARTMENT
st.sidebar.write("## Depot Compartment")

# 0.1 Absorption Rate Constant
st.sidebar.write("### Absorption Rate Constant (/h):")
col1, col2 = st.sidebar.columns(2)
tv_ka = col1.number_input("Typical Value:", min_value=0., value=0.0373, key="tv_ka")
eta_ka = col2.number_input("Variability:", min_value=0., value=0.225, key="eta_ka")
ka = np.random.normal(loc=1, scale=eta_ka, size=100) * tv_ka
ka = np.abs(ka.reshape(100, 1))

# 0.2 Bioavailability
st.sidebar.write("### Bioavailability:")
col1, col2 = st.sidebar.columns(2)
tv_f = col1.number_input("Typical Value:", min_value=0., value=0.8, key="tv_f")
eta_f = col2.number_input("Variability:", min_value=0., value=0., key="eta_f")
f = np.random.normal(loc=1, scale=eta_f, size=100) * tv_f
f = np.abs(f.reshape(100, 1))

# 1 CENTRAL COMPARTMENT
st.sidebar.write("## Central Compartment")

# 1.1 Clearance
st.sidebar.write("### Clearance (L/h):")
col1, col2 = st.sidebar.columns(2)
tv_cl = col1.number_input("Typical Value:", min_value=0., value=0.0329, step=0.01)
eta_cl = col2.number_input("Variability:", min_value=0., value=0.142)

cl = np.random.normal(loc=1, scale=eta_cl, size=100) * tv_cl
cl = cl.reshape(100, 1)

# 1.2 Volume of Distribution Input
st.sidebar.write("### Volume of Distribution (L):")
col1, col2 = st.sidebar.columns(2)
tv_vc = col1.number_input("Typical Value:", min_value=0., value=2.5)
eta_vc = col2.number_input("Variability:", min_value=0., value=0.49)

vc = np.random.normal(loc=1, scale=eta_vc, size=100) * tv_vc
vc = np.abs(vc.reshape(100, 1))

# 2. PERIPHERAL COMPARTMENT
st.sidebar.write("## Peripheral Compartment")
# 2.1 Inter-Compartment Clearance
st.sidebar.write("### Inter-Compartment Clearance (L/h):")
col1, col2 = st.sidebar.columns(2)
tv_q = col1.number_input("Typical Value:", min_value=0., value=0.126, key="tv_q")
eta_q = col2.number_input("Variability:", min_value=0., value=0., key="eta_q")
q = np.random.normal(loc=1, scale=eta_q, size=100) * tv_q
q = np.abs(q.reshape(100, 1))

# 2.2 Volume of Distribution
st.sidebar.write("### Volume of Distribution (L):")
col1, col2 = st.sidebar.columns(2)
tv_vp = col1.number_input("Typical Value:", min_value=0., value=4.0, key="tv_vp")
eta_vp = col2.number_input("Variability:", min_value=0., value=0., key="eta_vp")
vp = np.random.normal(loc=1, scale=eta_vp, size=100) * tv_vp
vp = np.abs(vp.reshape(100, 1))

theta = np.concatenate([ka, cl, vc, vp, q, f], axis=1)

# if st.sidebar.button("Start Simulating"):
#     st.write("Simulation in Progress...")
# else:
#     st.write("Set up parameters and click on **Start Simulating**")


# ======================================================================
# MAIN PENAL ===========================================================
# ======================================================================
st.write("""
# Dose Selection Based on Steady-State Pharmacokinetic Profiles
""")

st.write("## PK Profiles")
# def simulate(): # TODO: 把它封装进simulate func，通过sidebar button来更新结果
#     return res
res = []
for i in range(100):
    r = ModelSimulator(pk_model, amount=dose*10**6, interval=dosing_interval*24, cpt=3,
                       ka=theta[i, 0], cl=theta[i, 1], vc=theta[i, 2],
                       vp=theta[i, 3], q=theta[i, 4], f=theta[i, 5])
    r = r * num_dose
    res.append(r.solve()["conc"][:, 1] / theta[i, 2] / 10 ** 3)

fig, ax = plt.subplots(figsize=(12, 5))
t = r.solve()["time_point"]
for i in range(100):
    ax.plot(t, res[i], "grey", alpha=.2)
ax.plot(t, np.median(res, axis=0), "blue")
ax.plot(t, np.quantile(res, 0.025, axis=0), "r--")
ax.plot(t, np.quantile(res, 0.975, axis=0), "r--")
ax.axhline(ref_high, color="C1", linestyle="--")
ax.axhline(ref_low, color="C1", linestyle="--")
st.pyplot(fig)

# Input Parameters

st.write("## User Defined Model Parameters")
col1, col2, col3 = st.columns(3)
col1.write("### Depot Compartment")
dataframe = pd.DataFrame({
    # "Parameters": ["Typical Value", "Variability"],
    "Ka (/h)": [tv_ka, eta_ka],
    "F": [tv_f, eta_f]
    }, index=["Typical Value", "Variability"])
col1.write(dataframe)

col2.write("### Central Compartment")
dataframe = pd.DataFrame({
    "Cl (L/h)": [tv_cl, eta_cl],
    "Vd (L)": [tv_vc, eta_vc]
    }, index=["Typical Value", "Variability"])
col2.write(dataframe)

col3.write("### Peripheral Compartment")
dataframe = pd.DataFrame({
    # "Parameters": ["Typical Value", "Variability"],
    "Q (L/h)": [tv_q, eta_q],
    "Vp (L)": [tv_vp, eta_vp]
    }, index=["Typical Value", "Variability"])
col3.write(dataframe)

# 模拟后的关键暴露参数（平均浓度，峰浓度，谷浓度，AUC）列表
st.write("## Steady-State Exposure Parameters")
res = np.array(res)
t = np.linspace(0, dosing_interval*24, 300)
auc = KineticsCalculator(time=t, conc=np.quantile(res[:, -300:], 0.5, axis=0)).auc_lin_log()
cmax = KineticsCalculator(time=t, conc=np.quantile(res[:, -300:], 0.5, axis=0)).cmax()
tmax = KineticsCalculator(time=t, conc=np.quantile(res[:, -300:], 0.5, axis=0)).tmax()
ctrough = np.quantile(res[:, -300:], 0.5, axis=0)[0]


df = {
    "AUC,ss (ng*h/mL)": [np.round(auc, decimals=2)],
    "Tmax,ss (h)": [np.round(tmax, decimals=2)],
    "Cmax,ss (ng/mL)": [np.round(cmax, decimals=2)],
    "Ctrough,ss (ng/mL)": [np.round(ctrough, decimals=2)],
    "Cave,ss (ng/mL)": [np.round(np.mean(np.quantile(res[:, -300:], 0.5, axis=0)), decimals=2)],
    "95% Low Level (ng/mL)": [np.round(np.max(np.quantile(res[:, -300:], 0.025, axis=0)), decimals=2)],
    "95% High Level (ng/mL)": [np.round(np.max(np.quantile(res[:, -300:], 0.975, axis=0)), decimals=2)],
}

dataframe = pd.DataFrame(df)

st.write(dataframe)
