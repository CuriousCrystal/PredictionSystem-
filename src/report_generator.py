"""
Utility to generate a visual report (image) from student analysis text.
"""

import matplotlib.pyplot as plt
import os
from src.config import RESULTS_DIR

def save_analysis_report_as_image(analysis_text: str, filename: str = "student_analysis_report.png"):
    """
    Renders the analysis text into a clean image report.
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Configure background
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Remove axes
    ax.axis('off')
    
    # Add the text
    plt.text(0.05, 0.95, analysis_text, 
             fontsize=12, 
             color='#e0e0e0', 
             fontfamily='monospace',
             verticalalignment='top',
             wrap=True)
    
    # Add a title
    plt.title("Student Performance Analysis Report", 
              color='#53d8a8', 
              fontsize=18, 
              fontweight='bold', 
              pad=20)
    
    # Save the file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=150)
    plt.close()
    
    print(f"[OK] Visual report generated: {path}")
    return path
