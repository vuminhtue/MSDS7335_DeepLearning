import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Make sure directories exist
os.makedirs('report/figures', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Updated model metrics
models_data = [
    ('Ridge Regression', 33156.43, 30907.81, 0.8755),  # CV RMSE, Test RMSE, R²
    ('Lasso Regression', 32278.64, 28241.51, 0.8960),
    ('Random Forest', 29042.51, 29032.17, 0.8901),
    ('Neural Network', 29181.09, 29713.48, 0.8849)
]

# Create DataFrame for visualization
comparison_df = pd.DataFrame({
    'Model': [data[0] for data in models_data],
    'CV RMSE': [data[1] for data in models_data],
    'Test RMSE': [data[2] for data in models_data],
    'Test R²': [data[3] for data in models_data]
})

# Sort by test RMSE
comparison_df = comparison_df.sort_values('Test RMSE')

# Print the comparison table
print("\nModel Comparison:")
print(comparison_df)

# Plot RMSE comparison
plt.figure(figsize=(12, 8))

# RMSE subplot
plt.subplot(2, 1, 1)
x = np.arange(len(models_data))
width = 0.35
cv_rmse = comparison_df['CV RMSE'].values
test_rmse = comparison_df['Test RMSE'].values
plt.bar(x - width/2, cv_rmse, width, label='CV RMSE')
plt.bar(x + width/2, test_rmse, width, label='Test RMSE')
plt.xticks(x, comparison_df['Model'].values)
plt.ylabel('RMSE')
plt.title('Model Comparison - RMSE (lower is better)')
plt.legend()
plt.grid(axis='y')

# R² subplot
plt.subplot(2, 1, 2)
r2 = comparison_df['Test R²'].values
plt.bar(x, r2)
plt.xticks(x, comparison_df['Model'].values)
plt.ylabel('R² Score')
plt.title('Model Comparison - R² (higher is better)')
plt.grid(axis='y')

# Add values on top of bars
for i, val in enumerate(r2):
    plt.text(i, val + 0.01, f'{val:.4f}', ha='center')

# Save the figure
plt.tight_layout()
plt.savefig('report/figures/model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')

print("Model comparison figure updated and saved to report/figures/ and figures/ directories.") 