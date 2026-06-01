# ============================================
# COMPARAISON PID RÉEL vs MODÈLES ML
# ============================================
print("="*80)
print(" COMPARAISON PID RÉEL vs MODÈLES MACHINE LEARNING")
print("="*80)

import numpy as np
import json
import os
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid

# ============================================
# 1. CHARGEMENT DES DONNÉES PRÉPARÉES
# ============================================
print("\n1. CHARGEMENT DES DONNÉES")
print("-"*60)

# Charger les données preprocessing
if os.path.exists("../data/prepared_data_advanced.npz"):
    file_path = "../data/prepared_data_advanced.npz"
elif os.path.exists("data/prepared_data_advanced.npz"):
    file_path = "data/prepared_data_advanced.npz"
else:
    print(" Fichier non trouvé!")
    exit(1)

data = np.load(file_path)

# Version originale (6 features) pour le PID
X_test_orig = data['X_test_orig_norm']
y_test = data['y_test_norm']

print(f" Données chargées:")
print(f"   X_test (6 features): {X_test_orig.shape}")
print(f"   y_test: {y_test.shape}")

# Récupérer les noms des colonnes
target_cols = data['target_cols'].tolist() if hasattr(data['target_cols'], 'tolist') else list(data['target_cols'])
print(f"   Commandes: {target_cols}")

# ============================================
# 2. CHARGEMENT DES RÉSULTATS DU MODÈLE ML (XGBoost)
# ============================================
print("\n2. CHARGEMENT DES RÉSULTATS ML")
print("-"*60)

# Charger les résultats du benchmark
if os.path.exists("./resultats/benchmark_results.json"):
    with open("./resultats/benchmark_results.json", 'r') as f:
        benchmark_results = json.load(f)
    
    best_model_name = benchmark_results['best_model']
    best_model_r2 = benchmark_results['best_r2']
    best_model_mae = benchmark_results['best_mae']
    
    print(f" Résultats ML chargés:")
    print(f"   Meilleur modèle: {best_model_name}")
    print(f"   R² = {best_model_r2:.4f}")
    print(f"   MAE = {best_model_mae:.4f}")
else:
    print(" Fichier benchmark_results.json non trouvé")
    best_model_name = "XGBoost"
    best_model_r2 = 0.8794
    best_model_mae = 0.2130
    print(f"   Valeurs par défaut: R²={best_model_r2}, MAE={best_model_mae}")

# ============================================
# 3. CHARGEMENT DES SCALERS
# ============================================
print("\n3. CHARGEMENT DES SCALERS")
print("-"*60)

scaler_X = joblib.load('models/scaler_X_original.pkl')
scaler_y = joblib.load('models/scaler_y.pkl')

print(f" Scalers chargés: scaler_X (6 features), scaler_y")

# ============================================
# 4. DÉNORMALISATION DES DONNÉES
# ============================================
print("\n4. DÉNORMALISATION DES DONNÉES")
print("-"*60)

# Dénormaliser
X_test_physical = scaler_X.inverse_transform(X_test_orig)
y_test_physical = scaler_y.inverse_transform(y_test)

print(f"   X_test normalisé - min: {X_test_orig.min():.4f}, max: {X_test_orig.max():.4f}")
print(f"   Positions X - min: {X_test_physical[:,0].min():.2f}, max: {X_test_physical[:,0].max():.2f}")
print(f"   Vitesses X - min: {X_test_physical[:,3].min():.2f}, max: {X_test_physical[:,3].max():.2f}")

# Clipping (sécurité)
X_test_physical = np.clip(X_test_physical, -15, 15)
print(f"   Après clipping - min: {X_test_physical.min():.2f}, max: {X_test_physical.max():.2f}")

# ============================================
# 5. FONCTION PID UNIQUE
# ============================================
def simulate_pid(X_test_physical, y_test_physical, Kp=0.8, Ki=0.02, Kd=0.5, dt=0.02):
    """
    Simule le PID avec gains personnalisables
    """
    n_samples = len(X_test_physical)
    pid_commands_physical = np.zeros((n_samples, 4))
    
    Kp_pos, Ki_pos, Kd_pos = Kp, Ki, Kd
    Kp_vel, Ki_vel, Kd_vel = Kp*0.8, Ki*0.5, Kd*0.6
    masse = 1.38
    g = 9.81
    
    positions = X_test_physical[:, :3]
    velocities = X_test_physical[:, 3:6]
    targets = y_test_physical[:, :3]
    
    integral_pos = np.zeros(3)
    prev_error_pos = np.zeros(3)
    integral_vel = np.zeros(3)
    prev_error_vel = np.zeros(3)
    
    for t in range(n_samples):
        error_pos = targets[t] - positions[t]
        integral_pos += error_pos * dt
        integral_pos = np.clip(integral_pos, -2.0, 2.0)
        derivative_pos = (error_pos - prev_error_pos) / dt if dt > 0 else 0
        pos_command = Kp_pos * error_pos + Ki_pos * integral_pos + Kd_pos * derivative_pos
        prev_error_pos = error_pos.copy()
        
        target_vel = np.zeros(3)
        error_vel = target_vel - velocities[t]
        integral_vel += error_vel * dt
        integral_vel = np.clip(integral_vel, -2.0, 2.0)
        derivative_vel = (error_vel - prev_error_vel) / dt if dt > 0 else 0
        vel_command = Kp_vel * error_vel + Ki_vel * integral_vel + Kd_vel * derivative_vel
        prev_error_vel = error_vel.copy()
        
        thrust = np.clip(masse * g + pos_command[2] * masse, 8.0, 20.0)
        roll = np.clip(pos_command[1] * 0.08, -0.4, 0.4)
        pitch = np.clip(pos_command[0] * 0.08, -0.4, 0.4)
        yaw = np.clip(vel_command[0] * 0.5, -0.8, 0.8)
        
        pid_commands_physical[t] = [thrust, roll, pitch, yaw]
    
    return scaler_y.transform(pid_commands_physical)

# ============================================
# 6. PID ORIGINAL (gains par défaut)
# ============================================
print("\n5. SIMULATION DU PID ORIGINAL")
print("-"*60)

pid_predictions = simulate_pid(X_test_physical, y_test_physical, Kp=0.8, Ki=0.02, Kd=0.5)

pid_r2 = r2_score(y_test, pid_predictions)
pid_mae = mean_absolute_error(y_test, pid_predictions)
pid_mse = mean_squared_error(y_test, pid_predictions)

print(f"\n PERFORMANCES DU PID ORIGINAL:")
print(f"   R² = {pid_r2:.4f}")
print(f"   MSE = {pid_mse:.4f}")
print(f"   MAE = {pid_mae:.4f}")

# ============================================
# 7. OPTIMISATION DES GAINS PID
# ============================================
print("\n6. OPTIMISATION DES GAINS PID")
print("-"*60)

print(" Recherche des meilleurs gains PID...")

param_grid = {
    'Kp': [0.5, 1.0, 2.0, 5.0],
    'Ki': [0.01, 0.05, 0.1, 0.5],
    'Kd': [0.3, 0.8, 1.5, 3.0]
}

best_r2 = -np.inf
best_params = None

n_samples_test = min(5000, len(X_test_physical))
X_sample = X_test_physical[:n_samples_test]
y_sample = y_test_physical[:n_samples_test]
y_norm_sample = y_test[:n_samples_test]

for params in ParameterGrid(param_grid):
    try:
        pid_pred = simulate_pid(X_sample, y_sample, params['Kp'], params['Ki'], params['Kd'])
        r2 = r2_score(y_norm_sample, pid_pred)
        print(f"   Kp={params['Kp']}, Ki={params['Ki']}, Kd={params['Kd']} → R²={r2:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_params = params
    except:
        pass

print(f"\n MEILLEURS GAINS TROUVÉS:")
print(f"   Kp = {best_params['Kp']}, Ki = {best_params['Ki']}, Kd = {best_params['Kd']}")
print(f"   R² = {best_r2:.4f}")

# ============================================
# 8. PID OPTIMISÉ
# ============================================
print("\n7. SIMULATION DU PID OPTIMISÉ")
print("-"*60)

pid_optimized = simulate_pid(X_test_physical, y_test_physical, 
                              best_params['Kp'], best_params['Ki'], best_params['Kd'])

pid_opt_r2 = r2_score(y_test, pid_optimized)
pid_opt_mae = mean_absolute_error(y_test, pid_optimized)
pid_opt_mse = mean_squared_error(y_test, pid_optimized)

print(f"\n PERFORMANCES DU PID OPTIMISÉ:")
print(f"   R² = {pid_opt_r2:.4f}")
print(f"   MSE = {pid_opt_mse:.4f}")
print(f"   MAE = {pid_opt_mae:.4f}")

# ============================================
# 9. COMPARAISON FINALE
# ============================================
print("\n" + "="*60)
print(" COMPARAISON FINALE")
print("="*60)

print(f"""
┌────────────┬──────────┬──────────┐
│ Modèle     │ R²       │ MAE      │
├────────────┼──────────┼──────────┤
│ XGBoost    │ {best_model_r2:.4f} │ {best_model_mae:.4f} │
│ PID orig   │ {pid_r2:.4f} │ {pid_mae:.4f} │
│ PID optim  │ {pid_opt_r2:.4f} │ {pid_opt_mae:.4f} │
└────────────┴──────────┴──────────┘
""")

if best_model_r2 > pid_opt_r2:
    print(f" XGBoost surpasse le PID optimisé (+{best_model_r2 - pid_opt_r2:.4f} en R²)")
else:
    print(f" PID optimisé surpasse XGBoost (+{pid_opt_r2 - best_model_r2:.4f} en R²)")

# ============================================
# 10. SAUVEGARDE
# ============================================
print("\n8. SAUVEGARDE DES RÉSULTATS")
print("-"*60)

results = {
    'pid_original': {'r2': float(pid_r2), 'mae': float(pid_mae), 'mse': float(pid_mse)},
    'pid_optimized': {'r2': float(pid_opt_r2), 'mae': float(pid_opt_mae), 'mse': float(pid_opt_mse)},
    'ml': {'model': best_model_name, 'r2': float(best_model_r2), 'mae': float(best_model_mae)},
    'best_gains': best_params
}

with open('./resultats/comparison_pid_vs_ml.json', 'w') as f:
    json.dump(results, f, indent=2)

print(" Résultats sauvegardés: resultats/comparison_pid_vs_ml.json")
print("\n COMPARAISON TERMINÉE AVEC SUCCÈS !")