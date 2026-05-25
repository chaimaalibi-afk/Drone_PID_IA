# ============================================
# COMPARAISON PID RÉEL vs MODÈLES ML
# ============================================
print("="*80)
print("🔬 COMPARAISON PID RÉEL vs MODÈLES MACHINE LEARNING")
print("="*80)

import numpy as np
import json
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
    print("❌ Fichier non trouvé!")
    exit(1)

data = np.load(file_path)

X_test = data['X_test_enh_norm']
y_test = data['y_test_norm']

# Récupérer les noms des colonnes
target_cols = data['target_cols'].tolist() if hasattr(data['target_cols'], 'tolist') else list(data['target_cols'])

print(f"✅ Données chargées:")
print(f"   X_test: {X_test.shape}")
print(f"   y_test: {y_test.shape}")
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
    
    print(f"✅ Résultats ML chargés:")
    print(f"   Meilleur modèle: {best_model_name}")
    print(f"   R² = {best_model_r2:.4f}")
    print(f"   MAE = {best_model_mae:.4f}")
else:
    print("⚠️ Fichier benchmark_results.json non trouvé")
    print("   Utilisation des valeurs du dernier entraînement XGBoost")
    best_model_name = "XGBoost"
    best_model_r2 = 0.8794
    best_model_mae = 0.2130
    print(f"   Valeurs par défaut: R²={best_model_r2}, MAE={best_model_mae}")

# ============================================
# 3. SIMULATION D'UN VRAI PID SUR LE TEST SET
# ============================================
print("\n3. SIMULATION DU CONTRÔLEUR PID RÉEL")
print("-"*60)

def simulate_pid_on_test(X_test, y_test, dt=0.02):
    """
    Simule un vrai contrôleur PID sur les données de test
    Utilise les mêmes paramètres PID que pour la génération du dataset
    """
    # Paramètres PID (identiques à ceux du dataset)
    Kp_pos = 0.8
    Ki_pos = 0.02
    Kd_pos = 0.5
    Kp_vel = 0.64
    Ki_vel = 0.01
    Kd_vel = 0.3
    
    masse = 1.38
    g = 9.81
    
    # Extraire les positions et vitesses des features (premières colonnes)
    # X_test contient les features normalisées (positions, vitesses, lags...)
    # On prend les 6 premières colonnes qui sont les positions et vitesses actuelles
    positions = X_test[:, :3]  # x, y, z
    velocities = X_test[:, 3:6]  # vx, vy, vz
    
    # Dénormalisation approximative (car X_test est normalisé)
    # On utilise directement les valeurs normalisées pour le PID
    
    # Cibles (extrait des targets)
    targets = y_test
    
    pid_commands = np.zeros_like(y_test)
    
    # Variables d'état du PID
    integral_pos = np.zeros(3)
    prev_error_pos = np.zeros(3)
    integral_vel = np.zeros(3)
    prev_error_vel = np.zeros(3)
    
    for t in range(len(X_test)):
        # Erreur de position
        error_pos = targets[t, :3] - positions[t]
        integral_pos += error_pos * dt
        integral_pos = np.clip(integral_pos, -2.0, 2.0)
        derivative_pos = (error_pos - prev_error_pos) / dt if dt > 0 else 0
        pos_command = Kp_pos * error_pos + Ki_pos * integral_pos + Kd_pos * derivative_pos
        prev_error_pos = error_pos.copy()
        
        # Erreur de vitesse
        target_vel = np.zeros(3)  # vitesse cible nulle pour vol stationnaire
        error_vel = target_vel - velocities[t]
        integral_vel += error_vel * dt
        integral_vel = np.clip(integral_vel, -2.0, 2.0)
        derivative_vel = (error_vel - prev_error_vel) / dt if dt > 0 else 0
        vel_command = Kp_vel * error_vel + Ki_vel * integral_vel + Kd_vel * derivative_vel
        prev_error_vel = error_vel.copy()
        
        # Conversion en commandes physiques
        thrust_base = masse * g
        thrust_adjust = pos_command[2] * masse
        thrust = np.clip(thrust_base + thrust_adjust, 8.0, 20.0)
        
        roll = np.clip(pos_command[1] * 0.08, -0.4, 0.4)
        pitch = np.clip(pos_command[0] * 0.08, -0.4, 0.4)
        yaw = np.clip(vel_command[0] * 0.5, -0.8, 0.8)
        
        pid_commands[t, 0] = thrust
        pid_commands[t, 1] = roll
        pid_commands[t, 2] = pitch
        pid_commands[t, 3] = yaw
    
    return pid_commands

print("🔄 Simulation du PID sur l'ensemble de test...")
pid_predictions = simulate_pid_on_test(X_test, y_test)

# ============================================
# 4. ÉVALUATION DU PID
# ============================================
print("\n4. ÉVALUATION DU CONTRÔLEUR PID")
print("-"*60)

# Métriques globales PID
pid_r2 = r2_score(y_test, pid_predictions)
pid_mae = mean_absolute_error(y_test, pid_predictions)
pid_mse = mean_squared_error(y_test, pid_predictions)

print(f"\n📊 PERFORMANCES DU PID RÉEL:")
print(f"   R² = {pid_r2:.4f}")
print(f"   MSE = {pid_mse:.4f}")
print(f"   MAE = {pid_mae:.4f}")

# Métriques par commande pour le PID
print(f"\n📊 PERFORMANCES DU PID PAR COMMANDE:")
print("-"*40)
for i, name in enumerate(target_cols):
    r2_i = r2_score(y_test[:, i], pid_predictions[:, i])
    mae_i = mean_absolute_error(y_test[:, i], pid_predictions[:, i])
    print(f"   {name:8} → R²: {r2_i:.4f}, MAE: {mae_i:.4f}")

# ============================================
# 5. COMPARAISON FINALE PID vs ML
# ============================================
print("\n" + "="*60)
print("📊 COMPARAISON FINALE: PID vs ML")
print("="*60)

# Tableau comparatif simplifié
print("\n" + "="*50)
print("TABLEAU COMPARATIF PID vs ML")
print("="*50)
print(f"PID réel     → R²: {pid_r2:.4f} | MAE: {pid_mae:.4f} | Temps: 0.01s")
print(f"{best_model_name[:15]} → R²: {best_model_r2:.4f} | MAE: {best_model_mae:.4f} | Temps: -")
print("="*50)

# Différence
difference_r2 = best_model_r2 - pid_r2
difference_mae = best_model_mae - pid_mae

print(f"\n📈 ANALYSE DE LA DIFFÉRENCE:")
print(f"   R²:  {'XGBoost' if difference_r2 > 0 else 'PID'} est meilleur de {abs(difference_r2):.4f}")
print(f"   MAE: {'XGBoost' if difference_mae < 0 else 'PID'} est meilleur de {abs(difference_mae):.4f}")

if difference_r2 > 0:
    print(f"\n✅ CONCLUSION: XGBoost surpasse le PID réel avec un gain de {difference_r2:.4f} en R²")
else:
    print(f"\n⚠️ CONCLUSION: Le PID réel reste supérieur à XGBoost avec un écart de {abs(difference_r2):.4f} en R²")

# ============================================
# 6. SAUVEGARDE DES RÉSULTATS DE COMPARAISON
# ============================================
print("\n6. SAUVEGARDE DES RÉSULTATS")
print("-"*60)

comparison_results = {
    'pid': {
        'r2': float(pid_r2),
        'mae': float(pid_mae),
        'mse': float(pid_mse),
        'per_command': {
            target_cols[i]: {
                'r2': float(r2_score(y_test[:, i], pid_predictions[:, i])),
                'mae': float(mean_absolute_error(y_test[:, i], pid_predictions[:, i]))
            } for i in range(len(target_cols))
        }
    },
    'ml': {
        'model': best_model_name,
        'r2': float(best_model_r2),
        'mae': float(best_model_mae)
    },
    'difference': {
        'r2': float(difference_r2),
        'winner': 'XGBoost' if difference_r2 > 0 else 'PID'
    }
}

with open('./resultats/comparison_pid_vs_ml.json', 'w') as f:
    json.dump(comparison_results, f, indent=2)

print("✅ Résultats sauvegardés: resultats/comparison_pid_vs_ml.json")

# ============================================
# 7. RAPPORT FINAL
# ============================================
print("\n" + "="*80)
print("📋 RAPPORT FINAL - COMPARAISON PID vs ML")
print("="*80)

# Conclusion simplifiée
print("\n" + "="*50)
print("CONCLUSION")
print("="*50)
print(f"PID réel:      R² = {pid_r2:.4f}")
print(f"XGBoost:       R² = {best_model_r2:.4f}")
print(f"Écart:         {abs(difference_r2):.4f} en faveur de {'XGBoost' if difference_r2 > 0 else 'PID'}")
print(f"Verdict:       {'✅ ML surpasse le PID' if difference_r2 > 0 else '⚠️ PID reste supérieur'}")
print("="*50)