# ============================================
# EDA - DATASET AVANCÉ DRONE (600k échantillons)
# ============================================
print("="*80)
print(" ANALYSE EXPLORATOIRE - DATASET AVANCÉ DRONE")
print("="*80)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from ydata_profiling import ProfileReport
import missingno as msno
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
os.makedirs("resultats", exist_ok=True)

# ============================================
# 1. CHARGEMENT DU DATASET
# ============================================
print("\n1. CHARGEMENT DU DATASET")
print("-"*60)

# Chemin relatif depuis votre dossier de travail
# Assurez-vous que le fichier est dans le bon dossier
df = pd.read_csv(r'C:\Users\pc lenovo\Documents\Data Science\Drone_PID_IA\data\drone_dataset_advanced.csv')

print(f" Dataset chargé: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f" Colonnes: {list(df.columns)}")
print(f" Mémoire: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================
# 2. VALEURS MANQUANTES
# ============================================
print("\n2. ANALYSE DES VALEURS MANQUANTES")
print("-"*60)

print(f"Valeurs manquantes totales: {df.isnull().sum().sum()}")
nan_cols = df.columns[df.isnull().any()].tolist()
if nan_cols:
    print(f" Colonnes avec NaN: {nan_cols}")
    msno.matrix(df, figsize=(12, 4))
    plt.title("Matrice des valeurs manquantes")
    plt.savefig("resultats/missing_values.png", dpi=150)
    plt.show()
else:
    print(" Aucune valeur manquante détectée")

# ============================================
# 3. STATISTIQUES GLOBALES
# ============================================
print("\n3. STATISTIQUES DESCRIPTIVES GLOBALES")
print("-"*60)

feature_cols = ['position_x', 'position_y', 'position_z', 
                'velocity_x', 'velocity_y', 'velocity_z']
target_cols = ['thrust', 'roll', 'pitch', 'yaw']

print("\n VARIABLES D'ÉTAT (positions et vitesses):")
print(df[feature_cols].describe().round(4))

print("\n VARIABLES DE COMMANDE (PID):")
print(df[target_cols].describe().round(4))

# ============================================
# 4. STATISTIQUES PAR TYPE DE TRAJECTOIRE
# ============================================
print("\n4. STATISTIQUES PAR TYPE DE TRAJECTOIRE")
print("-"*60)

# Note: Les métadonnées des trajectoires sont stockées dans le fichier metadata
# Si vous avez sauvegardé les métadonnées, utilisez-les.
# Sinon, nous faisons une analyse basée sur les distributions

# Créer une colonne de temps fictive pour l'analyse
df['time'] = range(len(df))

# Détection automatique des segments de trajectoire
# (basé sur les changements brusques de position)
df['segment'] = (np.abs(df['velocity_x'].diff()) > 2).cumsum()

# Statistiques par segment (trajectoire)
segment_stats = df.groupby('segment')[target_cols].agg(['mean', 'std']).round(4)
print(f"Nombre de segments détectés: {segment_stats.shape[0]}")
print("\nAperçu des statistiques par segment:")
print(segment_stats.head(10))

# ============================================
# 5. VISUALISATION DES DISTRIBUTIONS
# ============================================
print("\n5. VISUALISATION DES DISTRIBUTIONS")
print("-"*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, feat in enumerate(feature_cols):
    row, col = i // 3, i % 3
    axes[row, col].hist(df[feat], bins=100, alpha=0.7, edgecolor='black', density=True)
    axes[row, col].set_title(f'Distribution de {feat}')
    axes[row, col].set_xlabel(feat)
    axes[row, col].set_ylabel('Densité')
    axes[row, col].axvline(df[feat].mean(), color='red', linestyle='--', label=f'Moyenne: {df[feat].mean():.2f}')
    axes[row, col].axvline(df[feat].median(), color='green', linestyle='--', label=f'Médiane: {df[feat].median():.2f}')
    axes[row, col].legend()

plt.tight_layout()
plt.savefig("resultats/feature_distributions.png", dpi=150)
plt.show()

# Distributions des commandes
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, cmd in enumerate(target_cols):
    row, col = i // 2, i % 2
    axes[row, col].hist(df[cmd], bins=100, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    axes[row, col].set_title(f'Distribution de {cmd}')
    axes[row, col].set_xlabel(cmd)
    axes[row, col].set_ylabel('Densité')
    axes[row, col].axvline(df[cmd].mean(), color='red', linestyle='--', label=f'Moyenne: {df[cmd].mean():.3f}')
    axes[row, col].axvline(df[cmd].median(), color='green', linestyle='--', label=f'Médiane: {df[cmd].median():.3f}')
    axes[row, col].legend()

plt.tight_layout()
plt.savefig("resultats/command_distributions.png", dpi=150)
plt.show()

# ============================================
# 6. DÉTECTION DES OUTLIERS (Méthode IQR)
# ============================================
print("\n6. DÉTECTION DES OUTLIERS (Méthode IQR)")
print("-"*60)

outlier_counts = {}
for col in feature_cols + target_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_counts[col] = outliers
    print(f"  {col:15}: {outliers:6,} outliers ({outliers/len(df)*100:.2f}%)")

# Boxplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, cmd in enumerate(target_cols):
    row, col = i // 2, i % 2
    axes[row, col].boxplot(df[cmd])
    axes[row, col].set_title(cmd)
    axes[row, col].set_ylabel('Valeur')
    axes[row, col].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("resultats/outliers_boxplots.png", dpi=150)
plt.show()

# ============================================
# 7. MATRICE DE CORRÉLATION
# ============================================
print("\n7. MATRICE DE CORRÉLATION")
print("-"*60)

plt.figure(figsize=(10, 8))
corr_matrix = df[feature_cols + target_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5)
plt.title("Matrice de corrélation - Variables d'état et commandes", fontsize=14)
plt.tight_layout()
plt.savefig("resultats/correlation_matrix.png", dpi=150)
plt.show()

# Corrélations les plus fortes avec les commandes
print("\n Corrélations les plus fortes avec les commandes:")
for cmd in target_cols:
    corr = corr_matrix[cmd].drop(cmd).abs().sort_values(ascending=False)
    print(f"\n  {cmd}:")
    for feat in corr.head(3).index:
        print(f"    {feat}: {corr[feat]:.3f}")

# ============================================
# RELATIONS ENTRE COMMANDES
# ============================================
print("\n" + "="*80)
print(" RELATIONS ENTRE COMMANDES")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Thrust vs Altitude
axes[0, 0].scatter(df['position_z'], df['thrust'], alpha=0.1, s=1)
axes[0, 0].set_xlabel('Altitude Z (m)')
axes[0, 0].set_ylabel('Thrust (N)')
axes[0, 0].set_title('Thrust vs Altitude')
axes[0, 0].axhline(y=df['thrust'].mean(), color='r', linestyle='--', label=f'Moyenne: {df["thrust"].mean():.2f} N')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Roll vs Velocity Y
axes[0, 1].scatter(df['velocity_y'], df['roll'], alpha=0.1, s=1)
axes[0, 1].set_xlabel('Vitesse Y (m/s)')
axes[0, 1].set_ylabel('Roll (rad)')
axes[0, 1].set_title('Roll vs Vitesse latérale')
axes[0, 1].grid(True, alpha=0.3)

# Pitch vs Velocity X
axes[1, 0].scatter(df['velocity_x'], df['pitch'], alpha=0.1, s=1)
axes[1, 0].set_xlabel('Vitesse X (m/s)')
axes[1, 0].set_ylabel('Pitch (rad)')
axes[1, 0].set_title('Pitch vs Vitesse longitudinale')
axes[1, 0].grid(True, alpha=0.3)

# Yaw vs Velocity X
axes[1, 1].scatter(df['velocity_x'], df['yaw'], alpha=0.1, s=1)
axes[1, 1].set_xlabel('Vitesse X (m/s)')
axes[1, 1].set_ylabel('Yaw (rad)')
axes[1, 1].set_title('Yaw vs Vitesse X')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultats/commands_relations.png", dpi=150)
plt.show()
print(" Graphique sauvegardé: resultats/commands_relations.png")

# ============================================
# 8. ANALYSE TEMPORELLE
# ============================================
print("\n8. ANALYSE TEMPORELLE (premiers 5000 échantillons)")
print("-"*60)

df_sample = df.iloc[:5000].copy()
df_sample['time'] = range(len(df_sample))

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Altitude
axes[0].plot(df_sample['time'], df_sample['position_z'], label='Altitude (Z)', linewidth=1, color='blue')
axes[0].set_ylabel('Position Z (m)')
axes[0].set_title('Évolution temporelle de l\'altitude')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Thrust
axes[1].plot(df_sample['time'], df_sample['thrust'], label='Thrust', linewidth=1, color='orange')
axes[1].set_ylabel('Thrust (N)')
axes[1].set_title('Évolution temporelle de la poussée')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Angles
axes[2].plot(df_sample['time'], df_sample['roll'], label='Roll', linewidth=1, alpha=0.7)
axes[2].plot(df_sample['time'], df_sample['pitch'], label='Pitch', linewidth=1, alpha=0.7)
axes[2].plot(df_sample['time'], df_sample['yaw'], label='Yaw', linewidth=1, alpha=0.7)
axes[2].set_xlabel('Temps (steps)')
axes[2].set_ylabel('Angle (rad)')
axes[2].set_title('Évolution temporelle des angles')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultats/temporal_evolution.png", dpi=150)
plt.show()

# ============================================
# 9. RAPPORT AUTOMATIQUE (ydata-profiling)
# ============================================
print("\n9. GÉNÉRATION DU RAPPORT AUTOMATIQUE")
print("-"*60)

try:
    from ydata_profiling import ProfileReport
    from IPython.display import display, HTML
    
    profile = ProfileReport(
        df, 
        title="Drone Advanced Dataset Report - 600k samples",
        explorative=True,
        minimal=False
    )
    profile.to_file("resultats/eda_report_advanced.html")
    print( "Rapport sauvegardé: resultats/eda_report_advanced.html")
    
    # Afficher dans Colab (optionnel)
    # display(HTML(profile.to_html()))
    
except Exception as e:
    print(f" Rapport non généré: {e}")

# ============================================
# 10. ANALYSE DES PERCENTILES
# ============================================
print("\n10. ANALYSE DES PERCENTILES")
print("-"*60)

percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

for cmd in target_cols:
    print(f"\n{cmd.upper()}:")
    for p in percentiles:
        val = df[cmd].quantile(p/100)
        print(f"  {p}ème percentile: {val:.4f}")

# ============================================
# 11. STATISTIQUES AVANCÉES
# ============================================
print("\n11. STATISTIQUES AVANCÉES")
print("-"*60)
print("\n Kurtosis (aplatissement des distributions):")
for cmd in target_cols:
    kurt = df[cmd].kurtosis()
    print(f"  {cmd}: {kurt:.3f}")

print("\n Skewness (asymétrie des distributions):")
for cmd in target_cols:
    skew = df[cmd].skew()
    print(f"  {cmd}: {skew:.3f}")

# ============================================
# 12. SAUVEGARDE DU RÉSUMÉ EDA
# ============================================
print("\n12. SAUVEGARDE DU RÉSUMÉ EDA")
print("-"*60)

# Fonction pour convertir les types numpy
def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Créer le dictionnaire avec conversion
eda_summary = {
    'total_samples': int(len(df)),
    'total_features': int(len(df.columns)),
    'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024**2),
    'missing_values': {k: int(v) if not pd.isna(v) else 0 for k, v in df.isnull().sum().to_dict().items()},
    'outliers': {k: int(v) for k, v in outlier_counts.items()},
    'statistics': {
        cmd: {
            'mean': float(df[cmd].mean()),
            'std': float(df[cmd].std()),
            'min': float(df[cmd].min()),
            'max': float(df[cmd].max()),
            'skew': float(df[cmd].skew()),
            'kurtosis': float(df[cmd].kurtosis())
        } for cmd in target_cols
    },
    'percentiles': {
        cmd: {f'{p}%': float(df[cmd].quantile(p/100)) for p in percentiles}
        for cmd in target_cols
    }
}

# Sauvegarde avec conversion
with open('resultats/eda_summary_advanced.json', 'w') as f:
    json.dump(eda_summary, f, indent=2, default=convert_to_serializable)

print(" Résumé sauvegardé: resultats/eda_summary_advanced.json")

# ============================================
# 13. RAPPORT FINAL
# ============================================
print("\n" + "="*80)
print(" RAPPORT FINAL EDA - DATASET AVANCÉ")
print("="*80)
print("\n EDA terminée avec succès!")