import streamlit as st
import numpy as np
import pandas as pd
from ase.io import read
from collections import defaultdict
from collections import Counter
import plotly.express as px
from tqdm import tqdm
import re

st.set_page_config(layout="wide", page_title="Chemical System Analyzer")

@st.cache_data
def load_data(uploaded_file):
    """Load and process XYZ file data from a file upload."""
    # Save the uploaded file to a temporary file
    with open("temp.xyz", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Read the temporary file
    structures = read("temp.xyz", index=':')
    return structures

def get_chemical_formula(elements):
    """Convert list of elements to a chemical formula."""
    counts = {}
    for element in elements:
        counts[element] = counts.get(element, 0) + 1
    formula = ''.join(f"{k}{v if v > 1 else ''}" for k, v in sorted(counts.items()))
    return formula

def get_formula_type(formula):
    """Determine formula type (e.g., AB, AB2, A2B3, etc.)."""
    elements = re.findall('([A-Z][a-z]*)(\d*)', formula)
    element_counts = [int(count) if count else 1 for _, count in elements]
    gcd = np.gcd.reduce(element_counts)
    simplified = [str(c//gcd) if c//gcd > 1 else '' for c in element_counts]
    return ''.join(f"{e}{c}" for e, c in zip([e for e, _ in elements], simplified))

def analyze_chemical_systems(structures):
    """Analyze chemical systems and formula types."""
    systems = []
    formula_types = []
    elements_list = []
    
    for atoms in tqdm(structures, desc="Analyzing structures"):
        elements = [atom.symbol for atom in atoms]
        elements_sorted = sorted(list(set(elements)))
        system = '-'.join(elements_sorted)
        formula = get_chemical_formula(elements)
        ftype = get_formula_type(formula)
        
        systems.append(system)
        formula_types.append(ftype)
        elements_list.append(elements_sorted)
    
    return pd.DataFrame({
        'system': systems,
        'formula_type': formula_types,
        'elements': elements_list
    })
import plotly.graph_objects as go
from mendeleev import element

def plot_periodic_table(element_counts):
    """
    Create an interactive periodic table visualization using Plotly.
    
    Args:
        element_counts (dict): Dictionary with element symbols as keys and counts as values
    """
    # Get element data
    elements_data = []
    for symbol, count in element_counts.items():
        try:
            e = element(symbol)
            elements_data.append({
                'symbol': e.symbol,
                'name': e.name,
                'atomic_number': e.atomic_number,
                'group': e.group_id if e.group_id else 0,
                'period': e.period,
                'count': count,
                'x': e.group_id if e.group_id else 0,
                'y': e.period
            })
        except:
            continue

    # Create hover text
    hover_text = []
    for el in elements_data:
        hover_text.append(
            f"Element: {el['name']}<br>"
            f"Symbol: {el['symbol']}<br>"
            f"Atomic Number: {el['atomic_number']}<br>"
            f"Count: {el['count']}"
        )

    # Create figure
    fig = go.Figure()

    # Add scatter plot for elements
    fig.add_trace(go.Scatter(
        x=[el['x'] for el in elements_data],
        y=[-el['y'] for el in elements_data],  # Negative to make H at top
        mode='markers+text',
        text=[el['symbol'] for el in elements_data],
        textposition='middle center',
        hovertext=hover_text,
        hoverinfo='text',
        marker=dict(
            size=40,
            color=[el['count'] for el in elements_data],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title='Count',
                thickness=20,
                yanchor='top', y=1,
                xanchor='left', x=1.02
            ),
            line=dict(width=2, color='DarkSlateGrey')
        ),
        textfont=dict(
            size=12,
            color='white'  # Text color
        )
    ))

    # Update layout
    fig.update_layout(
        title='Element Distribution',
        xaxis=dict(
            title='Group',
            tickmode='array',
            tickvals=list(range(1, 19)),
            ticktext=list(range(1, 19)),
            range=[0, 19],
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title='Period',
            tickmode='array',
            tickvals=[-i for i in range(1, 8)],
            ticktext=list(range(1, 8)),
            range=[-8, 0],
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='white',
        width=1000,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False
    )

    # Add element counts as annotations
    for el in elements_data:
        fig.add_annotation(
            x=el['x'],
            y=-el['y'] + 0.15,  # Position count below symbol
            text=str(el['count']),
            showarrow=False,
            font=dict(size=10, color='black')
        )

    return fig
def main():
    st.title("Chemical System Analyzer")
    st.write("Upload an XYZ file to analyze chemical systems and formula types")
    
    uploaded_file = st.file_uploader("Choose an XYZ file", type="xyz")
    
    if uploaded_file is not None:
        with st.spinner('Loading and analyzing data...'):
            structures = load_data(uploaded_file)
            df = analyze_chemical_systems(structures)
            
            if not df.empty:
                # Count elements across all structures
                element_counts = {}
                for elements in df['elements']:
                    for el in elements:
                        element_counts[el] = element_counts.get(el, 0) + 1
                
                # Create and display periodic table
                st.header("Element Distribution")
                fig = plot_periodic_table(element_counts)
                st.plotly_chart(fig, use_container_width=True)



            st.header("Dataset Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Chemical Systems")
                sys_counts = df['system'].value_counts().reset_index()
                sys_counts.columns = ['Chemical System', 'Count']
                fig = px.bar(sys_counts.head(10), 
                            x='Chemical System', 
                            y='Count',
                            title="Top 10 Chemical Systems")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.subheader("Formula Types")
                type_counts = df['formula_type'].value_counts().reset_index()
                type_counts.columns = ['Formula Type', 'Count']
                fig = px.pie(type_counts.head(10), 
                            names='Formula Type', 
                            values='Count',
                            title="Top 10 Formula Types")
                st.plotly_chart(fig, use_container_width=True)
            
            st.header("Detailed Analysis")
            
            # Show detailed table
            st.subheader("All Compounds")
            st.dataframe(df, use_container_width=True)
            
            # Element co-occurrence analysis
            st.subheader("Element Co-occurrence")
            all_elements = sorted(list(set([e for sublist in df['elements'] for e in sublist])))
            selected_elements = st.multiselect("Select elements to analyze", all_elements)
            
            if selected_elements:
                filtered = df[df['elements'].apply(lambda x: all(e in x for e in selected_elements))]
                st.write(f"Found {len(filtered)} compounds containing {', '.join(selected_elements)}")
                
                if not filtered.empty:
                    st.subheader("Formula Types for Selected Elements")
                    fig = px.histogram(filtered, x='formula_type', 
                                     title=f"Formula Types for {', '.join(selected_elements)}")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Export results
            st.download_button(
                label="Download Analysis Results",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='chemical_system_analysis.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()