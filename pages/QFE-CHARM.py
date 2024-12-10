# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:39:05 2024

@author: StephanNef
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
     page_title="IMP Functional Encryption",
     page_icon="🔐",
     layout="wide",
)

st.logo("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/HSG_Logo_DE_RGB.svg/1024px-HSG_Logo_DE_RGB.svg.png",link="https://www.unisg.ch/de/")

st.markdown("""
# Quadratic Functional Encryption (QFE)
""")

with st.expander("📖 See Bounded QFE Scheme"):
    st.image("figures/qfe.png")

st.info("**Reference**: C. Elisabetta, D. Catalano, D. Fiore, and R. Gay, “Practical Functional Encryption for Quadratic Functions with Applications to Predicate Encryption,” Cryptology ePrint Archive, 2017. https://eprint.iacr.org/2017/151")

st.warning("⚠️ This site is still under construction ⚠️")

