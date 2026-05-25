# ============================================
# BENCHMARK ML - COMPARAISON DES MODÈLES CLASSIQUES
# ============================================
print("="*80)
print(" ÉTAPE 5 : BENCHMARK DES MODÈLES MACHINE LEARNING")
print("="*80)

import numpy as np
import pandas as pd
import time
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV

# XGBoost et LightGBM
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Pour la sauvegarde
import joblib

import os
os.makedirs("resultats", exist_ok=True)

# ============================================
# 1. CHARGEMENT DES DONNÉES PRÉPARÉES
# ============================================
print("\n1. CHARGEMENT DES DONNÉES PRÉPARÉES")
print("-"*60)

# Charger les données preprocessing
data = np.load("../data/prepared_data_advanced.npz") if os.path.exists("../data/prepared_data_advanced.npz") else np.load("data/prepared_data_advanced.npz")

# Version enrichie (78 features)
X_train = data['X_train_enh_norm']
X_val = data['X_val_enh_norm']
X_test = data['X_test_enh_norm']

# Targets
y_train = data['y_train_norm']
y_val = data['y_val_norm']
y_test = data['y_test_norm']

# Récupérer les noms des colonnes
feature_cols = data['feature_cols'].tolist() if hasattr(data['feature_cols'], 'tolist') else list(data['feature_cols'])
target_cols = data['target_cols'].tolist() if hasattr(data['target_cols'], 'tolist') else list(data['target_cols'])

print(f" Données chargées avec succès!")
print(f" X_train: {X_train.shape}")
print(f" y_train: {y_train.shape}")
print(f" X_val:   {X_val.shape}")
print(f" X_test:  {X_test.shape}")

# ============================================
# 2. FONCTIONS UTILITAIRES
# ============================================
print("\n2. FONCTIONS UTILITAIRES")
print("-"*60)

def evaluate_model(model, X_test, y_test, model_name="Modèle"):
    """Évalue un modèle et retourne les métriques"""
    y_pred = model.predict(X_test)
    
    # Métriques globales
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Métriques par commande
    metrics_per_target = {}
    for i, name in enumerate(target_cols):
        metrics_per_target[name] = {
            'mse': mean_squared_error(y_test[:, i], y_pred[:, i]),
            'mae': mean_absolute_error(y_test[:, i], y_pred[:, i]),
            'r2': r2_score(y_test[:, i], y_pred[:, i])
        }
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'per_target': metrics_per_target,
        'y_pred': y_pred
    }

def plot_comparison(results_dict, save_path=r"./resultats/benchmark_comparison.png"):
    """Visualise la comparaison des modèles"""
    os.makedirs("resultats", exist_ok=True)
    
    model_names = list(results_dict.keys())
    r2_scores = [results_dict[name]['r2'] for name in model_names]
    mae_scores = [results_dict[name]['mae'] for name in model_names]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphique R²
    colors = ['green' if i == np.argmax(r2_scores) else 'steelblue' for i in range(len(model_names))]
    bars1 = axes[0].bar(range(len(model_names)), r2_scores, color=colors)
    axes[0].set_xticks(range(len(model_names)))
    axes[0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Comparaison des modèles - R² (plus haut = meilleur)')
    axes[0].set_ylim([0, 1])
    axes[0].axhline(y=0.95, color='green', linestyle='--', label='Objectif 0.95')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for bar, val in zip(bars1, r2_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Graphique MAE
    bars2 = axes[1].bar(range(len(model_names)), mae_scores, color=colors)
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Comparaison des modèles - MAE (plus bas = meilleur)')
    axes[1].grid(True, alpha=0.3)
    
    for bar, val in zip(bars2, mae_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f" Graphique sauvegardé: {save_path}")

# ============================================
# 3. DÉFINITION DES MODÈLES
# ============================================
print("\n3. DÉFINITION DES MODÈLES À COMPARER")
print("-"*60)

models = {
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0),
    #'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1),
    'MLP (Sklearn)': MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42, early_stopping=True)
}

print(f" {len(models)} modèles enregistrés pour comparaison")

# ============================================
# 4. ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES
# ============================================
print("\n4. ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES")
print("-"*60)

results = {}
training_times = {}

for name, model in models.items():
    print(f"\n Entraînement de {name}...")
    start_time = time.time()
    
    try:
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        training_times[name] = training_time
        
        # Évaluation sur validation
        y_pred_val = model.predict(X_val)
        val_mse = mean_squared_error(y_val, y_pred_val)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_r2 = r2_score(y_val, y_pred_val)
        
        # Évaluation sur test
        eval_results = evaluate_model(model, X_test, y_test, name)
        results[name] = eval_results
        
        print(f"    Temps: {training_time:.2f}s")
        print(f"    Validation - R²: {val_r2:.4f}, MAE: {val_mae:.4f}")
        print(f"    Test - R²: {eval_results['r2']:.4f}, MAE: {eval_results['mae']:.4f}")
        
        # Détail par commande
        print(f"    Détail par commande (Test):")
        for target, metrics in eval_results['per_target'].items():
            print(f"      {target}: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
            
    except Exception as e:
        print(f"    Erreur: {str(e)[:100]}")
        results[name] = None

# ============================================
# 4.5 MODÈLES SPÉCIALISÉS PAR COMMANDE
# ============================================
print("\n4.5 MODÈLES SPÉCIALISÉS PAR COMMANDE")
print("-"*60)

from xgboost import XGBRegressor

specialized_models = {}
specialized_results = {}

# Hyperparamètres adaptés à chaque commande
params_by_target = {
    'thrust': {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.01},
    'roll': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.02},
    'pitch': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05},
    'yaw': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05}
}

for i, target in enumerate(target_cols):
    print(f"\n Entraînement pour {target}...")
    start_time = time.time()
    
    params = params_by_target[target]
    model = XGBRegressor(**params, random_state=42, verbosity=0)
    model.fit(X_train, y_train[:, i])
    
    training_time = time.time() - start_time
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test[:, i], y_pred)
    mae = mean_absolute_error(y_test[:, i], y_pred)
    
    specialized_models[target] = model
    specialized_results[target] = {'r2': r2, 'mae': mae, 'time': training_time}
    
    print(f"    R² = {r2:.4f}, MAE = {mae:.4f}, Temps = {training_time:.2f}s")

# R² global de l'ensemble
y_pred_ensemble = np.column_stack([specialized_models[t].predict(X_test) for t in target_cols])
r2_ensemble = r2_score(y_test, y_pred_ensemble)
print(f"\n R² global de l'ensemble: {r2_ensemble:.4f}")

# ============================================
# 5. OPTIMISATION DES HYPERPARAMÈTRES (XGBoost pour thrust)
# ============================================
print("\n5. OPTIMISATION DES HYPERPARAMÈTRES")
print("-"*60)

try:
    from sklearn.model_selection import GridSearchCV
    
    print(" Optimisation pour thrust (le plus difficile)...")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    
    xgb = XGBRegressor(random_state=42, verbosity=0)
    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train[:, 0])  # Optimiser sur thrust
    
    print(f" Meilleurs paramètres pour thrust: {grid_search.best_params}")
    print(f" Meilleur R²: {grid_search.best_score_:.4f}")
    
    # Appliquer les meilleurs paramètres
    best_params = grid_search.best_params
    optimized_model = XGBRegressor(**best_params, random_state=42, verbosity=0)
    optimized_model.fit(X_train, y_train[:, 0])
    r2_opt = r2_score(y_test[:, 0], optimized_model.predict(X_test))
    print(f" R² du thrust optimisé: {r2_opt:.4f}")
    
except Exception as e:
    print(f" Optimisation ignorée: {e}")


# ============================================
# 6. ANALYSE DES ERREURS DÉTAILLÉE
# ============================================
print("\n6. ANALYSE DES ERREURS DÉTAILLÉE")
print("-"*60)

# Identifier le meilleur modèle
best_model_name = max(results, key=lambda x: results[x]['r2'] if results[x] is not None else -np.inf)
best_model_results = results[best_model_name]

print(f" Meilleur modèle: {best_model_name}")
print(f"   R²: {best_model_results['r2']:.4f}")
print(f"   MAE: {best_model_results['mae']:.4f}")

# Analyse détaillée des erreurs pour le meilleur modèle
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, target in enumerate(target_cols):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    y_true = y_test[:, i]
    y_pred = best_model_results['y_pred'][:, i]
    errors = np.abs(y_true - y_pred)
    
    ax.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=np.mean(errors), color='red', linestyle='--', label=f'Moyenne: {np.mean(errors):.4f}')
    ax.set_xlabel('Erreur absolue')
    ax.set_ylabel('Fréquence')
    ax.set_title(f'{target} - Distribution des erreurs')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultats/error_analysis.png", dpi=150)
plt.show()
print(" Graphique d'analyse des erreurs sauvegardé: resultats/error_analysis.png")

# ============================================
# 7. TESTS DE ROBUSTESSE
# ============================================
print("\n7. TESTS DE ROBUSTESSE")
print("-"*60)

def test_noise_robustness(model, X_test, y_test, noise_levels=[0, 0.05, 0.1, 0.2, 0.3]):
    """Teste la robustesse au bruit"""
    results = []
    for noise in noise_levels:
        X_noisy = X_test + np.random.normal(0, noise, X_test.shape)
        y_pred = model.predict(X_noisy)  # ← enlever verbose=0
        r2 = r2_score(y_test, y_pred)
        results.append({'noise': noise, 'r2': r2})
        print(f"   Bruit {noise*100:.0f}% → R²: {r2:.4f}")
    return results


def test_perturbation_rejection(model, X_test, y_test, magnitudes=[0, 0.5, 1.0, 2.0]):
    """Teste le rejet de perturbations"""
    results = []
    mid = len(X_test) // 2
    for mag in magnitudes:
        X_pert = X_test.copy()
        X_pert[mid:mid+50] += mag
        y_pred = model.predict(X_pert)
        err_before = mean_squared_error(y_test[:mid], y_pred[:mid])
        err_after = mean_squared_error(y_test[mid:], y_pred[mid:])
        ratio = err_after / err_before if err_before > 0 else 1
        results.append({'magnitude': mag, 'ratio': ratio})
        print(f"   Magnitude {mag} → Ratio: {ratio:.3f}")
    return results


# Appliquer les tests au meilleur modèle
print("\n Test de robustesse au bruit:")
best_model = models[best_model_name]
noise_results = test_noise_robustness(best_model, X_test, y_test)


print("\n Test de rejet de perturbation:")
pert_results = test_perturbation_rejection(best_model, X_test, y_test)

# ============================================
# 8. VISUALISATION DE LA ROBUSTESSE
# ============================================
print("\n8. VISUALISATION DE LA ROBUSTESSE")
print("-"*60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Graphique robustesse au bruit
noise_levels = [r['noise'] for r in noise_results] 
r2_scores = [r['r2'] for r in noise_results]
axes[0].plot(noise_levels, r2_scores, 'o-', linewidth=2, markersize=8, color='blue')
axes[0].set_xlabel('Niveau de bruit (std)')
axes[0].set_ylabel('R² Score')
axes[0].set_title(f'Robustesse au bruit - {best_model_name}')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0.95, color='green', linestyle='--', label='Référence PID')
axes[0].legend()

# Graphique rejet de perturbation
magnitudes = [p['magnitude'] for p in pert_results]  
degradation = [p['ratio'] for p in pert_results]
axes[1].plot(magnitudes, degradation, 's-', linewidth=2, markersize=8, color='red')
axes[1].set_xlabel('Magnitude de perturbation')
axes[1].set_ylabel('Ratio de dégradation (après/avant)')
axes[1].set_title(f'Rejet de perturbations - {best_model_name}')
axes[1].axhline(y=1.0, color='green', linestyle='--', label='Performance nominale')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./resultats/robustness_analysis.png", dpi=150)
plt.show()
print(" Graphique de robustesse sauvegardé: resultats/robustness_analysis.png")

# ============================================
# 9. COMPARAISON AVEC PID (SIMULATION)
# ============================================
print("\n9. COMPARAISON AVEC CONTRÔLEUR PID IDÉAL")
print("-"*60)

# Le PID parfait aurait R² = 1.0 et MAE = 0 c'est un PID théorique
print(f"""
PID théorique (optimal):
   - R² = 1.0000
   - MAE = 0.0000

Meilleur modèle ML ({best_model_name}):
   - R² = {best_model_results['r2']:.4f}
   - MAE = {best_model_results['mae']:.4f}

Écart par rapport au PID:
   - Différence R²: {1.0 - best_model_results['r2']:.4f}
   - MAE supplémentaire: {best_model_results['mae']:.4f}
""")
# ============================================
# 10. SAUVEGARDE DES RÉSULTATS
# ============================================
print("\n10. SAUVEGARDE DES RÉSULTATS")
print("-"*60)

os.makedirs("models", exist_ok=True)
os.makedirs("resultats", exist_ok=True)

# Sauvegarde du meilleur modèle
best_model = models[best_model_name]
joblib.dump(best_model, "models/best_ml_model.pkl")
print(f" Meilleur modèle sauvegardé: models/best_ml_model.pkl")

# Sauvegarde des résultats
results_summary = {
    'best_model': best_model_name,
    'best_r2': float(best_model_results['r2']),
    'best_mae': float(best_model_results['mae']),
    'all_models': {
        name: {
            'r2': float(results[name]['r2']),
            'mae': float(results[name]['mae']),
            'mse': float(results[name]['mse'])
        } for name in results if results[name] is not None
    },
    'robustness': {
        'noise': noise_results,
        'perturbation': pert_results
    }
}

with open('./resultats/benchmark_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print(" Résultats sauvegardés: resultats/benchmark_results.json")

# ============================================
# 11. RAPPORT FINAL
# ============================================
print("\n" + "="*80)
print(" RAPPORT FINAL - BENCHMARK ML")
print("="*80)
print("\n BENCHMARK ML TERMINÉ AVEC SUCCÈS !")
print(" Passez à l'étape suivante : Deep Learning avec LSTM")