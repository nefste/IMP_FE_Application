# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:39:05 2024

@author: StephanNef
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.linalg import null_space
import numpy as np

st.set_page_config(
     page_title="IMP Functional Encryption",
     page_icon="🔐",
     layout="wide",
)

st.logo("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/HSG_Logo_DE_RGB.svg/1024px-HSG_Logo_DE_RGB.svg.png",link="https://www.unisg.ch/de/")



st.markdown("""
# Fully Secure Inner Product Functional Encryption (IPFE)
""")


with st.expander("📖 See IPFE-FULLYSEC Scheme"):
    st.image("figures/ipfe-fullsec.png")

st.info("**Reference**: S. Agrawal, B. Libert, and D. Stehle, “Fully Secure Functional Encryption for Inner Products, from Standard Assumptions,” Cryptology ePrint Archive, 2015. https://eprint.iacr.org/2015/608 ")

st.warning("⚠️ This site is still under construction ⚠️")

