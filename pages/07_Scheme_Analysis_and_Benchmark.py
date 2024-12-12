# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:39:05 2024

@author: StephanNef
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(
     page_title="IMP Functional Encryption",
     page_icon="https://upload.wikimedia.org/wikipedia/de/thumb/7/77/Uni_St_Gallen_Logo.svg/2048px-Uni_St_Gallen_Logo.svg.png",
     layout="wide",
)

st.logo("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/HSG_Logo_DE_RGB.svg/1024px-HSG_Logo_DE_RGB.svg.png",link="https://www.unisg.ch/de/")





# Hilfsfunktion: Daten laden
def load_data():
    files = {
        "IPFE-DDH": {
            "length": "data/ipfe-ddh_timings_increasing_length.csv",
            "bits": "data/ipfe-ddh_timings_increasing_bits.csv",
        },
        "IPFE-FULLYSEC": {
            "length": "data/ipfe-fullysec_timings_increasing_l.csv",
            "bits": "data/ipfe-fullysec_timings_increasing_bits.csv",
        },
        "QFE-CHARM": {
            "qfe": "data/qfe_timings_increasing_k.csv",  # Die Pfad zur neuen Datei
        }
    }
    schemas = {}
    for schema, paths in files.items():
        schemas[schema] = {}
        for key, path in paths.items():
            schemas[schema][key] = pd.read_csv(path)
    return schemas

def to_excel(df) -> bytes:
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, sheet_name="Sheet1")
    writer.close()
    processed_data = output.getvalue()
    return processed_data

# Daten laden
schemas = load_data()

# Streamlit App
st.title("Functional Encryption - Analysis")

# Auswahl des Schemas
schema_select = ["IPFE-DDH", "IPFE-FULLYSEC", "QFE-CHARM"]

# Tab-Setup
tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ IPFE-DDH", "2️⃣ IPFE-FULLYSEC", "3️⃣ QFE-CHARM", "📊 Benchmarking"])

# Tab1: IPFE-DDH
with tab1:
    st.header("Analyse: IPFE-DDH")
    st.write("---")
    for schema in schema_select:
        if schema == "IPFE-DDH":
            # Plot für verschiedene Key Sizes
            df_bits = schemas[schema]["bits"]
            st.subheader(f"{schema}: Key Size-based Plots")
            st.info("Simulation run with a fixed Vector Length of 2")
            
            steps = ["time setup", "time encrypt", "time keyder", "time decrypt"]
            
            green_palette = [
                "#006400",  # DarkGreen
                "#228B22",  # ForestGreen
                "#32CD32",  # LimeGreen
                "#7CFC00",  # LawnGreen
            ]

            # Map steps to colors
            step_colors = {step: green_palette[i % len(green_palette)] for i, step in enumerate(steps)}
            
            # Multiselect für die Steps
            selected_steps = st.multiselect(
                "Select Steps:",
                options=steps,
                default=steps,  # Standardmäßig alle Schritte ausgewählt
                key="ipfe-ddh-steps"
            )
            
            
            # Plot für Key Sizes
            melted_df_bits = df_bits.melt(
                id_vars=["bits", "time total"], 
                value_vars=steps, 
                var_name="Step", 
                value_name="Time"
            )
            
                        
            # Daten auf die ausgewählten Schritte filtern
            melted_df_bits = melted_df_bits[melted_df_bits["Step"].isin(selected_steps)]
            
            # Absoluten Werte Barplot für Key Size
            fig_absolute_bits = px.bar(
                melted_df_bits,
                y="bits",  # Alle Bits auf der Y-Achse
                x="Time",
                color="Step",
                orientation="h",
                title=f"{schema}: Absolute Times (Key-Sizes)",
                labels={"Time": "Time (ns)", "bits": "Bits"},
                color_discrete_sequence=green_palette
            )
            fig_absolute_bits.update_traces(
                texttemplate="%{x:.1f} ns",
                textposition="inside"
            )
            log_scale = st.toggle("Logarithmic Scale for Time Axis", value=False, key="ipfe-ddh-logax")
            
            if log_scale:
                fig_absolute_bits.update_layout(xaxis_type="log")
                fig_absolute_bits.update_layout(barmode="group")
            else:
                fig_absolute_bits.update_layout(barmode="stack")
                
            st.plotly_chart(fig_absolute_bits)
            
            
            # Prozentualer Werte Barplot für Key Size
            melted_df_bits["Percentage"] = (melted_df_bits["Time"] / melted_df_bits["time total"]) * 100
            fig_percentage_bits = px.bar(
                melted_df_bits,
                y="bits",  # Alle Bits auf der Y-Achse
                x="Percentage",
                color="Step",
                orientation="h",
                title=f"{schema}: Percentage of Total Time (Key-Sizes)",
                labels={"Percentage": "Percentage (%)", "bits": "Bits"},
                color_discrete_sequence=green_palette
                
            )
            fig_percentage_bits.update_traces(
                texttemplate="%{x:.1f}%",
                textposition="inside"
            )
            st.plotly_chart(fig_percentage_bits)
            
            with st.expander("📊 Dataset for Download:"):
                st.dataframe(melted_df_bits, use_container_width=True)
                st.download_button(
                    "Download Data",
                    data=to_excel(melted_df_bits),
                    file_name="IMP_IPFE_DDH_df_bits.xlsx",
                    mime="application/vnd.ms-excel",
                    key="ipfe-ddh-bits"
                    )

            # Plot für verschiedene Vector Lengths
            df_length = schemas[schema]["length"]
            
            st.write('---')
            st.subheader(f"{schema}: Vector Length-based Plots")
            st.info("Simulation run with a fixed Key Size of 512")
            
            # Plot für Vector Lengths
            melted_df_length = df_length.melt(
                id_vars=["l", "time total"], 
                value_vars=steps, 
                var_name="Step", 
                value_name="Time"
            )
            melted_df_length["l_name"] = "l="+melted_df_length["l"].astype(str)
            
            min_l, max_l = int(melted_df_length["l"].min()), int(melted_df_length["l"].max())
            selected_range = st.slider(
                "Select Range for Vector Length (l):",
                min_value=min_l,
                max_value=max_l,
                value=(min_l, max_l),  
                step=1  # Schrittweite
            )
            
            melted_df_length = melted_df_length[
                melted_df_length["l"].astype(int).between(selected_range[0], selected_range[1])
            ]
            
            # Absoluten Werte Barplot für Vector Length
            fig_absolute_length = px.bar(
                melted_df_length,
                y=str("l_name"),  # Alle Lengths auf der Y-Achse
                x="Time",
                color="Step",
                orientation="h",
                title=f"{schema}: Absolute Times (Vector Lengths)",
                labels={"Time": "Time (ns)", "l": "Vector Length"},
                color_discrete_sequence=green_palette

            )
            fig_absolute_length.update_traces(
                texttemplate="%{x:.1f} ns",
                textposition="inside"
            )
            log_scale = st.toggle("Logarithmic Scale for Time Axis", value=False, key="ipfe-ddh-logax-l")
            
            if log_scale:
                fig_absolute_length.update_layout(xaxis_type="log")
                fig_absolute_length.update_layout(barmode="group")
            else:
                fig_absolute_length.update_layout(barmode="stack")
                
            st.plotly_chart(fig_absolute_length)
            
            # Prozentualer Werte Barplot für Vector Length
            melted_df_length["Percentage"] = (melted_df_length["Time"] / melted_df_length["time total"]) * 100
            melted_df_length = melted_df_length[melted_df_length["Step"].isin(selected_steps)]
            fig_percentage_length = px.bar(
                melted_df_length,
                y="l_name",  # Alle Lengths auf der Y-Achse
                x="Percentage",
                color="Step",
                orientation="h",
                title=f"{schema}: Percentage of Total Time (Vector Lengths)",
                labels={"Percentage": "Percentage (%)", "length": "Vector Length"},
                color_discrete_sequence=green_palette
            )
            fig_percentage_length.update_traces(
                texttemplate="%{x:.1f}%",
                textposition="inside"
            )
            st.plotly_chart(fig_percentage_length)
            
            with st.expander("📊 Dataset for Download:"):
                st.dataframe(melted_df_length, use_container_width=True)
                st.download_button(
                    "Download Data",
                    data=to_excel(melted_df_length),
                    file_name="IMP_IPFE_DDH_df_length.xlsx",
                    mime="application/vnd.ms-excel",
                    key="ipfe-ddh-length"
                    )

# Tab2: IPFE-FULLYSEC
with tab2:
    st.header("Analyse: IPFE-FULLYSEC")
    st.write("---")
    for schema in schema_select:
        if schema == "IPFE-FULLYSEC":
            # Plot für verschiedene Key Sizes
            df_bits = schemas[schema]["bits"]
            st.subheader(f"{schema}: Key Size-based Plots")
            st.info("Simulation run with a fixed Vector Length of 2")
            
            steps = ["time setup", "time encrypt", "time keygen", "time decrypt"]
            
            # Multiselect für die Steps
            selected_steps = st.multiselect(
                "Select Steps:",
                options=steps,
                default=steps,  # Standardmäßig alle Schritte ausgewählt
                key="ipfe-fullysec-steps"
            )
            
            
            
            # Plot für Key Sizes
            melted_df_bits = df_bits.melt(
                id_vars=["bits", "time total"], 
                value_vars=steps, 
                var_name="Step", 
                value_name="Time"
            )
            
            # Daten auf die ausgewählten Schritte filtern
            melted_df_bits = melted_df_bits[melted_df_bits["Step"].isin(selected_steps)]
            
            # Absoluten Werte Barplot für Key Size
            fig_absolute_bits = px.bar(
                melted_df_bits,
                y="bits",  # Alle Bits auf der Y-Achse
                x="Time",
                color="Step",
                orientation="h",
                title=f"{schema}: Absolute Times (Key-Sizes)",
                labels={"Time": "Time (ns)", "bits": "Bits"},
                color_discrete_sequence=green_palette
            )
            fig_absolute_bits.update_traces(
                texttemplate="%{x:.1f} ns",
                textposition="inside"
            )
            log_scale = st.toggle("Logarithmic Scale for Time Axis", value=False, key="ipfe-fullsec-logax")
            
            if log_scale:
                fig_absolute_bits.update_layout(xaxis_type="log")
                fig_absolute_bits.update_layout(barmode="group")
            else:
                fig_absolute_bits.update_layout(barmode="stack")
                
            st.plotly_chart(fig_absolute_bits)
            
            # Prozentualer Werte Barplot für Key Size
            melted_df_bits["Percentage"] = (melted_df_bits["Time"] / melted_df_bits["time total"]) * 100
            
            fig_percentage_bits = px.bar(
                melted_df_bits,
                y="bits",  # Alle Bits auf der Y-Achse
                x="Percentage",
                color="Step",
                orientation="h",
                title=f"{schema}: Percentage of Total Time (Key-Sizes)",
                labels={"Percentage": "Percentage (%)", "bits": "Bits"},
                color_discrete_sequence=green_palette
            )
            fig_percentage_bits.update_traces(
                texttemplate="%{x:.1f}%",
                textposition="inside"
            )
            st.plotly_chart(fig_percentage_bits)
            
            with st.expander("📊 Dataset for Download:"):
                st.dataframe(melted_df_bits, use_container_width=True)
                st.download_button(
                    "Download Data",
                    data=to_excel(melted_df_bits),
                    file_name="IMP_IPFE_FULLSEC_df_bits.xlsx",
                    mime="application/vnd.ms-excel",
                    key="ipfe-fullsec-bits"
                    )
            
            

            # Plot für verschiedene Vector Lengths
            df_length = schemas[schema]["length"]
            
            st.write("---")
            st.subheader(f"{schema}: Vector Length-based Plots")
            st.info("Simulation run with a fixed Key Size of 512")

            # Plot für Vector Lengths
            melted_df_length = df_length.melt(
                id_vars=["l", "time total"], 
                value_vars=steps, 
                var_name="Step", 
                value_name="Time"
            )
            melted_df_length["l_name"] = "l="+melted_df_length["l"].astype(str)
            melted_df_length = melted_df_length[melted_df_length["Step"].isin(selected_steps)]
            
            min_l, max_l = int(melted_df_length["l"].min()), int(melted_df_length["l"].max())
            selected_range = st.slider(
                "Select Range for Vector Length (l):",
                min_value=min_l,
                max_value=max_l,
                value=(min_l, max_l),  
                step=1 , # Schrittweite,
                key='ipfe-fullysec'
            )
            
            melted_df_length = melted_df_length[
                melted_df_length["l"].astype(int).between(selected_range[0], selected_range[1])
            ]
                    
            
            
            # Absoluten Werte Barplot für Vector Length
            fig_absolute_length = px.bar(
                melted_df_length,
                y="l_name",  # Alle Lengths auf der Y-Achse
                x="Time",
                color="Step",
                orientation="h",
                title=f"{schema}: Absolute Times (Vector Lengths)",
                labels={"Time": "Time (ns)", "l": "Vector Length"},
                color_discrete_sequence=green_palette
            )
            fig_absolute_length.update_traces(
                texttemplate="%{x:.1f} ns",
                textposition="inside"
            )
            
            log_scale = st.toggle("Logarithmic Scale for Time Axis", value=False, key="ipfe-fullsec-logax-l")
            
            if log_scale:
                fig_absolute_length.update_layout(xaxis_type="log")
                fig_absolute_length.update_layout(barmode="group")
            else:
                fig_absolute_length.update_layout(barmode="stack")
                
            st.plotly_chart(fig_absolute_length)
            
            # Prozentualer Werte Barplot für Vector Length
            melted_df_length["Percentage"] = (melted_df_length["Time"] / melted_df_length["time total"]) * 100
            
            fig_percentage_length = px.bar(
                melted_df_length,
                y="l_name",  # Alle Lengths auf der Y-Achse
                x="Percentage",
                color="Step",
                orientation="h",
                title=f"{schema}: Percentage of Total Time (Vector Lengths)",
                labels={"Percentage": "Percentage (%)", "length": "Vector Length"},
                color_discrete_sequence=green_palette
            )
            fig_percentage_length.update_traces(
                texttemplate="%{x:.1f}%",
                textposition="inside"
            )
            st.plotly_chart(fig_percentage_length)
            
            with st.expander("📊 Dataset for Download:"):
                st.dataframe(melted_df_length, use_container_width=True)
                st.download_button(
                    "Download Data",
                    data=to_excel(melted_df_length),
                    file_name="IMP_IPFE_FULLSEC_df_length.xlsx",
                    mime="application/vnd.ms-excel",
                    key="ipfe-fullsec-length"
                    )



with tab3:
    st.header("Analyse: QFE-CHARM")
    if schema == "QFE-CHARM":
        df = schemas[schema]["qfe"]
        st.subheader(f"Schema: {schema}")
    
        steps = ["time setup", "time keygen", "time encrypt", "time decrypt"]
        
        
        fig_line = go.Figure()
        
        # Linien hinzufügen
        fig_line.add_trace(go.Scatter(x=df['k'], y=df['time setup'], mode='lines', name='Time Setup'))
        fig_line.add_trace(go.Scatter(x=df['k'], y=df['time keygen'], mode='lines', name='Time Keygen'))
        fig_line.add_trace(go.Scatter(x=df['k'], y=df['time encrypt'], mode='lines', name='Time Encrypt'))
        fig_line.add_trace(go.Scatter(x=df['k'], y=df['time decrypt'], mode='lines', name='Time Decrypt'))
        fig_line.add_trace(go.Scatter(x=df['k'], y=df['time total'], mode='lines', name='Time Total'))
        
        # Layout für den Plot anpassen
        fig_line.update_layout(
            title="QFE Benchmark - Key Sizes with fixed Vectors",
            xaxis_title="k",
            yaxis_title="Time (seconds)",
            legend_title="Steps",
            template="plotly_white",
            xaxis=dict(showgrid=False),  # Gitternetzlinien der x-Achse ausblenden
            yaxis=dict(showgrid=False),  # Gitternetzlinien der y-Achse ausblenden
        )
        
        # Plot anzeigen
        st.plotly_chart(fig_line)
        
        st.write("---")
        
        # Multiselect für die Steps
        selected_steps = st.multiselect(
            "Select Steps:",
            options=steps,
            default=steps,  # Standardmäßig alle Schritte ausgewählt
            key="qfe_steps"
        )
        
        # Daten umwandeln, um alle k-Werte auf der Y-Achse anzuzeigen
        melted_df = df.melt(
            id_vars=["k", "time total"], 
            value_vars=steps, 
            var_name="Step", 
            value_name="Time"
        )
        
        # Daten auf die ausgewählten Schritte filtern
        melted_df = melted_df[melted_df["Step"].isin(selected_steps)]
        
        min_l, max_l = int(melted_df["k"].min()), int(melted_df["k"].max())
        selected_range = st.slider(
            "Select Range for Key Sizes (k):",
            min_value=min_l,
            max_value=max_l,
            value=(min_l, max_l),  
            step=1,  # Schrittweite
            key='qfe'
        )
        
        melted_df = melted_df[
            melted_df["k"].astype(int).between(selected_range[0], selected_range[1])
        ]
        
        # Absoluter Werte Barplot
        fig_absolute = px.bar(
            melted_df,
            y="k",  # Alle k-Werte auf der Y-Achse
            x="Time",
            color="Step",
            orientation="h",
            title=f"{schema}: Absolute Times (Key-Sizes)",
            labels={"Time": "Time (ns)", "k": "k Value"},
            color_discrete_sequence=green_palette
        )
        # Werte innerhalb der Balken anzeigen
        fig_absolute.update_traces(
            texttemplate="%{x:.1f} ns",
            textposition="inside"
        )
        st.plotly_chart(fig_absolute)
        

        
        # Prozentualer Werte Barplot
        melted_df["Percentage"] = (melted_df["Time"] / melted_df["time total"]) * 100
        fig_percentage = px.bar(
            melted_df,
            y="k",  # Alle k-Werte auf der Y-Achse
            x="Percentage",
            color="Step",
            orientation="h",
            title=f"{schema}: Percentage of Total Time (Key-Sizes)",
            labels={"Percentage": "Percentage (%)", "k": "k Value"},
            color_discrete_sequence=green_palette
        )
        fig_percentage.update_traces(
            texttemplate="%{x:.1f}%",
            textposition="inside"
        )
        
        st.plotly_chart(fig_percentage)
        
        with st.expander("📊 Dataset for Download:"):
            st.dataframe(melted_df, use_container_width=True)
            st.download_button(
                "Download Data",
                data=to_excel(melted_df),
                file_name="IMP_BOUND_QFE_df.xlsx",
                mime="application/vnd.ms-excel",
                key="bound-qfe"
                )
        
        



with tab4:
    st.header("Benchmarking: Direct Comparison of Schemes")
    st.subheader("Comparison between IPFE-DDH and IPFE-FULLYSEC")
    
    # Single-select for steps (combine keygen/keyder)
    steps = ["time setup", "time encrypt", "time keygen / keyder", "time decrypt"]
    selected_step = st.selectbox("Select a Step to Compare", steps)
    
    # Handle the "time keygen / keyder" case
    step_columns = (
        ["time keygen", "time keyder"] 
        if selected_step == "time keygen / keyder" 
        else [selected_step]
    )

    # Initialize dataframes for bits and vector length
    comparison_bits_df = pd.DataFrame()
    comparison_length_df = pd.DataFrame()
    schemas_to_compare = ["IPFE-DDH", "IPFE-FULLYSEC"]
    
    # Data preparation for Bits (Key Size)
    for schema in schemas_to_compare:
        df = schemas[schema]["bits"]
        df["Schema"] = schema  # Add schema column
        comparison_bits_df = pd.concat([comparison_bits_df, df])
    
    # Data preparation for Vector Length (l)
    for schema in schemas_to_compare:
        df = schemas[schema]["length"]
        df["Schema"] = schema  # Add schema column
        comparison_length_df = pd.concat([comparison_length_df, df])
    
    def create_bar_and_delta_plots(df, x_axis, title):
        # Filter data for the selected step(s)

        filtered_df = df[[x_axis, "Schema"] + step_columns].copy()
        
        # Add the "Step" column
        filtered_df["Step"] = filtered_df.apply(
            lambda row: "time keygen" if "time keygen" in step_columns else ("time keyder" if "time keyder" in step_columns else selected_step),
            axis=1
        )
        
        # Handle "time keygen / time keyder" case: Assign the correct step names
        if "time keygen" in step_columns and "time keyder" in step_columns:
            filtered_df["Step-Schema"] = filtered_df.apply(
                lambda row: "time keygen (IPFE-FULLYSEC)" 
                if row["Step"] == "time keygen" and row["Schema"] == "IPFE-FULLYSEC" 
                else "time keyder (IPFE-DDH)", axis=1
            )
        else:
            filtered_df["Step-Schema"] = filtered_df["Step"] + " (" + filtered_df["Schema"] + ")"
        
        # Melt the DataFrame for plotting
        melted_df = filtered_df.melt(
            id_vars=[x_axis, "Schema", "Step-Schema"],
            value_vars=step_columns,
            var_name="Step",
            value_name="Time"
        )
        
        # Create a pivot for Delta Calculation
        pivot_df = melted_df.pivot_table(
            index=[x_axis, "Step"], 
            columns="Schema", 
            values="Time"
        ).reset_index()
        
        
        # Ensure both schemas (IPFE-DDH and IPFE-FULLYSEC) are present for each row
        if "IPFE-DDH" in pivot_df.columns and "IPFE-FULLYSEC" in pivot_df.columns:
            # Calculate Delta in Percent (for keygen / keyder comparison)
            pivot_df["Delta (%)"] = round((
                (pivot_df["IPFE-FULLYSEC"] - pivot_df["IPFE-DDH"]) / pivot_df["IPFE-DDH"]
            ) * 100,2)
            
            # Filter rows where both schemas exist for the comparison
            delta_df = pivot_df.dropna(subset=["IPFE-FULLYSEC", "IPFE-DDH"])
            
           
            st.write("---")
            # Create bar plot
            st.subheader(f"{title} - {selected_step.capitalize()} Comparison")
            
            min_l, max_l = int(melted_df[x_axis].min()), int(melted_df[x_axis].max())
            
            selected_range = st.slider(
                f"Select {x_axis}:",
                min_value=min_l,
                max_value=max_l,
                value=(min_l, max_l),  
                step=1,  # Schrittweite
                key=x_axis
            )
            
            melted_df = melted_df[
                melted_df[x_axis].astype(int).between(selected_range[0], selected_range[1])
            ]
            
            melted_df[x_axis] = "l="+melted_df[x_axis].astype(str)
            delta_df[x_axis] = "l="+delta_df[x_axis].astype(str)
            
            bar_fig = px.bar(
                melted_df,
                x=x_axis,
                y="Time",
                color="Step-Schema",
                barmode="group",
                title=f"{title} - {selected_step.capitalize()} Comparison",
                labels={"Time": "Time (ns)", x_axis: x_axis.capitalize()},
                color_discrete_sequence=green_palette
            )
            st.plotly_chart(bar_fig, use_container_width=True)
                        
            # Create delta plot (if Delta % values are available)
            if not delta_df.empty:
                # Add a color column based on Delta values
                delta_df["Color"] = delta_df["Delta (%)"].apply(lambda x: "green" if x > 0 else "red")
                
                # Create the bar chart
                delta_fig = px.bar(
                    delta_df,
                    x=x_axis,
                    y="Delta (%)",
                    title=f"Delta (%) for {title}",
                    labels={"Delta (%)": "Delta (%)", x_axis: x_axis.capitalize()},
                    color="Color",  # Use the color column for bar colors
                    color_discrete_map={"green": "green", "red": "red"},  # Map colors explicitly
                    text="Delta (%)" 
                )
                
                delta_fig.update_traces(textposition="outside") 
                # Show the plot
                st.plotly_chart(delta_fig, use_container_width=True)
            else:
                st.warning("No Delta values available for comparison.")

    
        else:
            st.warning("Both schemas (IPFE-DDH and IPFE-FULLYSEC) are required for Delta calculation.")
    
        
    # Create plots for Bits (Key Size)
    create_bar_and_delta_plots(comparison_bits_df, x_axis="bits", title="Bits (Key Size)")
    
    # Create plots for Vector Length (l)
    create_bar_and_delta_plots(comparison_length_df, x_axis="l", title="Vector Length (l)")
    
    st.write("---")
    st.subheader("Compare QFE-CHARM with another IPFE Scheme by Total Time")

    # Choose which IPFE scheme to compare with QFE
    compare_with = st.selectbox("Compare QFE-CHARM with:", ["IPFE-DDH", "IPFE-FULLYSEC"], key="compare_with_total")

    # Load QFE data
    qfe_df = schemas["QFE-CHARM"]["qfe"].copy()
    qfe_df["Schema"] = "QFE-CHARM"

    # Load chosen IPFE scheme data
    ipfe_bits_df = schemas[compare_with]["bits"].copy()
    ipfe_bits_df["Schema"] = compare_with

    ipfe_length_df = schemas[compare_with]["length"].copy()
    ipfe_length_df["Schema"] = compare_with

    def compare_total_time(qfe_df, ipfe_df, qfe_param, ipfe_param, title, unique_id=""):
        st.markdown(f"### Comparison of QFE ({qfe_param}) with {compare_with} ({ipfe_param}) {title}")

        # Slider for QFE range
        qfe_min, qfe_max = int(qfe_df[qfe_param].min()), int(qfe_df[qfe_param].max())
        qfe_range = st.slider(
            f"Select Range for QFE {qfe_param.upper()}:",
            min_value=qfe_min,
            max_value=qfe_max,
            value=(qfe_min, qfe_max),
            step=1,
            key=f"qfe-{qfe_param}-range-total-{unique_id}"
        )

        qfe_filtered = qfe_df[qfe_df[qfe_param].between(qfe_range[0], qfe_range[1])]

        # Slider for IPFE range
        ipfe_min, ipfe_max = int(ipfe_df[ipfe_param].min()), int(ipfe_df[ipfe_param].max())
        ipfe_range = st.slider(
            f"Select Range for {compare_with} {ipfe_param.upper()}:",
            min_value=ipfe_min,
            max_value=ipfe_max,
            value=(ipfe_min, ipfe_max),
            step=1,
            key=f"ipfe-{ipfe_param}-range-total-{unique_id}"
        )

        ipfe_filtered = ipfe_df[ipfe_df[ipfe_param].between(ipfe_range[0], ipfe_range[1])]

        # For plotting, rename the parameter column to a common name
        qfe_filtered[f"{qfe_param}_label"] = qfe_param + "=" + qfe_filtered[qfe_param].astype(str)
        ipfe_filtered[f"{ipfe_param}_label"] = ipfe_param + "=" + ipfe_filtered[ipfe_param].astype(str)

        # Plot QFE total time
        fig_qfe = px.bar(
            qfe_filtered,
            y=f"{qfe_param}_label",
            x="time total",
            orientation="h",
            title=f"QFE (Param {qfe_param.upper()}) {title} - Total Time",
            labels={"time total": "Time (ns)", f"{qfe_param}_label": qfe_param.upper()},
            color_discrete_sequence=["#006400"]
        )
        fig_qfe.update_traces(texttemplate="%{x:.1f} ns", textposition="inside")
        st.plotly_chart(fig_qfe, use_container_width=True)

        # Plot IPFE total time
        fig_ipfe = px.bar(
            ipfe_filtered,
            y=f"{ipfe_param}_label",
            x="time total",
            orientation="h",
            title=f"{compare_with} (Param {ipfe_param.upper()}) {title} - Total Time",
            labels={"time total": "Time (ns)", f"{ipfe_param}_label": ipfe_param.upper()},
            color_discrete_sequence=["#228B22"]
        )
        fig_ipfe.update_traces(texttemplate="%{x:.1f} ns", textposition="inside")
        st.plotly_chart(fig_ipfe, use_container_width=True)

        # Download options
        with st.expander("📊 Download Filtered Data"):
            st.write("### QFE Data")
            st.dataframe(qfe_filtered, use_container_width=True)


            st.write(f"### {compare_with} Data")
            st.dataframe(ipfe_filtered, use_container_width=True)


    # Compare QFE (k) with IPFE (bits)
    compare_total_time(qfe_df, ipfe_bits_df, qfe_param="k", ipfe_param="bits", title="(Key Sizes)", unique_id="bits")

    # Compare QFE (k) with IPFE (l)
    compare_total_time(qfe_df, ipfe_length_df, qfe_param="k", ipfe_param="l", title="(Vector Length)", unique_id="length")
