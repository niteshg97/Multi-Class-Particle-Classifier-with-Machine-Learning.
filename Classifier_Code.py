# Nitesh Kumar
# 2349010
# NITP 
"""
This script trains models to identify particle types (Pions, Protons, Kaons, Electrons)
based on kinematic and detector features. It compares a Random Forest ensemble 
against a Multi-Layer Perceptron (Neural Network), specifically addressing the 
challenge of imbalanced data (rare electron signatures).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Particle iD mapping for readable outputs
PARTICLE_LABELS = {
    211: "Pion (π)",
    2212: "Proton (p)",
    321: "Kaon (K)",
    -11: "Electron (e-)"
}

def load_and_clean_data(filepath, sample_size=50000):
    """Loads the dataset, drops missing values, and takes a manageable sample."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Drop rows with missing values to ensure clean training
    df = df.dropna()
    
    # Sample the data to speed up training during prototyping
    if sample_size and sample_size < len(df):
        print(f"Sampling down to {sample_size} rows...")
        df = df.sample(n=sample_size, random_state=42)
        
    y = df['id']
    X = df.drop('id', axis=1)
    
    print("\nClass distribution in sample:")
    print(y.map(PARTICLE_LABELS).value_counts())
    
    return X, y

def plot_kinematics(X, y):
    """Generates a scatter plot of momentum vs. relativistic velocity."""
    print("\nGenerating Momentum vs. Velocity plot...")
    plt.figure(figsize=(10, 6))
    
    for particle_id, name in PARTICLE_LABELS.items():
        mask = (y == particle_id)
        plt.scatter(X.loc[mask, 'p'], X.loc[mask, 'beta'], label=name, alpha=0.5, s=10)

    plt.xlabel('Particle momentum (p) [GeV/c]')
    plt.ylabel('Relativistic velocity (β)')
    plt.title('Momentum vs. Velocity by Particle Type')
    plt.legend()
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('momentum_vs_velocity.png')
    print("Saved plot as 'momentum_vs_velocity.png'")
    plt.close()

def main():
    # Load data
    data_path = 'pid-5M.csv' # Update this path if your CSV is located elsewhere
    X, y = load_and_clean_data(data_path, sample_size=50000)
    
    #  Visualize
    plot_kinematics(X, y)
    
    #  Preprocess- Split before scaling to prevent data leakage
    print("\nSplitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    target_names = [PARTICLE_LABELS.get(i, str(i)) for i in np.unique(y_test)]

    #  Train Random Forest
    # Note- Using class_weight='balanced' to help it find the rare electrons
    print("\nTraining Random Forest")
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    
    print("\nRandom Forest Results")
    rf_preds = rf_model.predict(X_test_scaled)
    print(classification_report(y_test, rf_preds, target_names=target_names))

    #  Train Neural Network (MLP)
    # The NN often provides a more balanced recall for minority classes naturally
    print("\nTraining Neural Network")
    nn_model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42)
    nn_model.fit(X_train_scaled, y_train)
    
    print("\n Neural Network Results")
    nn_preds = nn_model.predict(X_test_scaled)
    print(classification_report(y_test, nn_preds, target_names=target_names))

if __name__ == "__main__":
    main()
