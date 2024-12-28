import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Data
data = {
    'Job Role': [
        'Front End Engineer',
        'Full Stack Engineer',
        'Data Scientist',
        'Brain Surgeon',
        'Product Manager'
    ],
    'Document Similarity': [0.7616, 0.7643, 0.7500, 0.7061, 0.7958],
    'TF-IDF Score': [0.6000, 0.6000, 0.3000, 0.0000, 0.7000],
    'Bullet Average': [0.7234, 0.7290, 0.7330, 0.7196, 0.7332],
    'Section Alignment': [0.7948, 0.8032, 0.8020, 0.7302, 0.8084],
    'LLM Match': [0.7000, 0.8000, 0.7000, 0.0750, 0.9650],
    'Final Score': [0.6849, 0.7820, 0.7386, 0.0110, 1.0000]
}

df = pd.DataFrame(data)

def create_heatmap():
    """Create and save a correlation heatmap of match scores"""
    score_columns = ['Document Similarity', 'TF-IDF Score', 'Bullet Average', 
                    'Section Alignment', 'LLM Match', 'Final Score']
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[score_columns].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Match Scores')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

def create_radar_chart():
    """Create and save a radar chart for each job role"""
    categories = ['Document Similarity', 'TF-IDF Score', 'Bullet Average', 
                 'Section Alignment', 'LLM Match']
    
    fig = go.Figure()
    
    for job in data['Job Role']:
        job_data = df[df['Job Role'] == job]
        values = job_data[categories].values[0]
        values = np.append(values, values[0])  # Complete the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=job
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title='Score Components by Job Role'
    )
    fig.write_html('radar_chart.html')

def create_bar_chart():
    """Create and save a bar chart of final scores"""
    fig = px.bar(df, 
                 x='Job Role', 
                 y='Final Score',
                 title='Final Match Scores by Job Role',
                 color='Final Score',
                 color_continuous_scale='Viridis')
    
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_range=[0, 1],
        yaxis_title='Match Score',
        xaxis_title='Job Role'
    )
    fig.write_html('bar_chart.html')

def create_detailed_matrix():
    """Create and save a detailed match matrix visualization"""
    score_columns = ['Document Similarity', 'TF-IDF Score', 'Bullet Average', 
                    'Section Alignment', 'LLM Match', 'Final Score']
    
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Job Role'] + score_columns,
                   fill_color='paleturquoise',
                   align='left'),
        cells=dict(values=[df['Job Role']] + [df[col] for col in score_columns],
                  fill_color='lavender',
                  align='left',
                  format=[None] + ['.3f']*len(score_columns)))
    ])
    
    fig.update_layout(title='Detailed Match Matrix')
    fig.write_html('match_matrix.html')

def main():
    """Generate all visualizations"""
    create_heatmap()
    create_radar_chart()
    create_bar_chart()
    create_detailed_matrix()
    print("Visualizations have been generated successfully!")

if __name__ == "__main__":
    main()