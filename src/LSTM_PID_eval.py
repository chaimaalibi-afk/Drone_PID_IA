# ============================================
# COMPARAISON LSTM vs PID RÉEL (AVEC ANALYSE COMPLÈTE)
# ============================================
print("="*80)
print(" COMPARAISON LSTM vs PID RÉEL")
print("="*80)

import numpy as np
import os
import json
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("resultats", exist_ok=True)

# ============================================
# 1. CHARGEMENT DES DONNÉES
# ============================================
print("\n1. CHARGEMENT DES DONNÉES")
print("-"*60)

if os.path.exists("../data/prepared_data_advanced.npz"):
    file_path = "../data/prepared_data_advanced.npz"
elif os.path.exists("data/prepared_data_advanced.npz"):
    file_path = "data/prepared_data_advanced.npz"
else:
    print(" Fichier non trouvé!")
    exit(1)

data = np.load(file_path)

X_test_enh = data['X_test_enh_norm']
y_test = data['y_test_norm']
target_cols = data['target_cols'].tolist()

print(f" Données chargées:")
print(f"   X_test: {X_test_enh.shape}")
print(f"   y_test: {y_test.shape}")
print(f"   Commandes: {target_cols}")

# ============================================
# 2. CRÉATION DES SÉQUENCES
# ============================================
print("\n2. CRÉATION DES SÉQUENCES")
print("-"*60)

def create_sequences(X, y, seq_length=50):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

SEQ_LENGTH = 50
X_test_seq, y_test_seq = create_sequences(X_test_enh, y_test, SEQ_LENGTH)
print(f" X_test_seq: {X_test_seq.shape}")

# ============================================
# 3. CHARGEMENT DU MODÈLE LSTM
# ============================================
print("\n3. CHARGEMENT DU MODÈLE LSTM")
print("-"*60)

def create_lstm_model(input_shape):
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LSTM(32, return_sequences=False, dropout=0.2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    outputs = keras.layers.Dense(4)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

input_shape = (SEQ_LENGTH, X_test_enh.shape[1])
model = create_lstm_model(input_shape)

if os.path.exists('models/best_lstm_model.h5'):
    model.load_weights('models/best_lstm_model.h5')
    print(" Modèle LSTM chargé")
else:
    print(" Modèle non trouvé!")
    exit(1)

# ============================================
# 4. PRÉDICTIONS LSTM
# ============================================
print("\n4. PRÉDICTIONS LSTM")
print("-"*60)

y_pred_lstm = model.predict(X_test_seq)

lstm_mse = mean_squared_error(y_test_seq, y_pred_lstm)
lstm_mae = mean_absolute_error(y_test_seq, y_pred_lstm)
lstm_r2 = r2_score(y_test_seq, y_pred_lstm)

print(f"\n LSTM - R²: {lstm_r2:.4f}, MAE: {lstm_mae:.4f}")

# ============================================
# 5. SIMULATION PID RÉEL
# ============================================
print("\n5. SIMULATION PID RÉEL")
print("-"*60)

scaler_X = joblib.load('models/scaler_X_enhanced.pkl')
scaler_y = joblib.load('models/scaler_y.pkl')

def simulate_pid(X_test_seq, y_test_seq, scaler_X, scaler_y, dt=0.02):
    n = len(X_test_seq)
    pid_cmd = np.zeros((n, 4))
    
    X_last = X_test_seq[:, -1, :]
    X_phys = scaler_X.inverse_transform(X_last)
    y_phys = scaler_y.inverse_transform(y_test_seq)
    
    Kp, Ki, Kd = 3.0, 0.05, 1.0
    m, g = 1.38, 9.81
    
    integral = np.zeros(3)
    prev_error = np.zeros(3)
    
    for t in range(n):
        error = y_phys[t, :3] - X_phys[t, :3]
        integral += error * dt
        derivative = (error - prev_error) / dt
        cmd = Kp * error + Ki * integral + Kd * derivative
        prev_error = error.copy()
        
        thrust = np.clip(m*g + cmd[2]*m, 8, 20)
        roll = np.clip(cmd[1] * 0.08, -0.4, 0.4)
        pitch = np.clip(cmd[0] * 0.08, -0.4, 0.4)
        yaw = 0.0
        
        pid_cmd[t] = scaler_y.transform([[thrust, roll, pitch, yaw]])[0]
    
    return pid_cmd

print(" Simulation du PID réel...")
pid_pred = simulate_pid(X_test_seq, y_test_seq, scaler_X, scaler_y)

pid_r2 = r2_score(y_test_seq, pid_pred)
pid_mae = mean_absolute_error(y_test_seq, pid_pred)
pid_mse = mean_squared_error(y_test_seq, pid_pred)

print(f" PID réel - R²: {pid_r2:.4f}, MAE: {pid_mae:.4f}")

# ============================================
# 6. COMPARAISON FINALE
# ============================================
print("\n" + "="*60)
print("COMPARAISON LSTM vs PID RÉEL")
print("="*60)

print(f"""
LSTM:     R² = {lstm_r2:.4f} | MAE = {lstm_mae:.4f}
PID réel: R² = {pid_r2:.4f} | MAE = {pid_mae:.4f}
""")

difference_r2 = lstm_r2 - pid_r2
if difference_r2 > 0:
    print(f" LSTM est meilleur de {difference_r2:.4f} en R²")
else:
    print(f" PID est meilleur de {abs(difference_r2):.4f} en R²")

# ============================================
# 7. COURBES D'APPRENTISSAGE (à partir de l'historique sauvegardé)
# ============================================
print("\n7. COURBES D'APPRENTISSAGE")
print("-"*60)

# Note: Pour avoir les courbes d'apprentissage, il faut les sauvegarder pendant l'entraînement.
# Si vous avez un fichier history.json, chargez-le. Sinon, ce graphique sera ignoré.

if os.path.exists('resultats/lstm_training_history.json'):
    import json
    with open('resultats/lstm_training_history.json', 'r') as f:
        history_data = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history_data['loss'], label='Train Loss', linewidth=1.5)
    axes[0].plot(history_data['val_loss'], label='Validation Loss', linewidth=1.5)
    axes[0].set_title('Courbe d\'apprentissage - Loss (MSE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    axes[1].plot(history_data['mae'], label='Train MAE', linewidth=1.5)
    axes[1].plot(history_data['val_mae'], label='Validation MAE', linewidth=1.5)
    axes[1].set_title('Courbe d\'apprentissage - MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("resultats/lstm_training_curves.png", dpi=150)
    plt.show()
    print(" Graphique sauvegardé: resultats/lstm_training_curves.png")
else:
    print(" Fichier d'historique non trouvé (les courbes d'apprentissage ne seront pas affichées)")

# ============================================
# 8. PRÉDICTIONS LSTM vs RÉELLES
# ============================================
print("\n8. PRÉDICTIONS LSTM vs RÉELLES")
print("-"*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, name in enumerate(target_cols):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    ax.scatter(y_test_seq[:500, i], y_pred_lstm[:500, i], alpha=0.3, s=10)
    ax.plot([y_test_seq[:, i].min(), y_test_seq[:, i].max()], 
            [y_test_seq[:, i].min(), y_test_seq[:, i].max()], 
            'r--', linewidth=2, label='Prédiction parfaite')
    ax.set_xlabel(f'{name} (réel)')
    ax.set_ylabel(f'{name} (prédit)')
    ax.set_title(f'{name} - LSTM (R²={r2_score(y_test_seq[:, i], y_pred_lstm[:, i]):.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultats/lstm_predictions.png", dpi=150)
plt.show()
print(" Graphique sauvegardé: resultats/lstm_predictions.png")

# ============================================
# 9. ANALYSE DES RÉSIDUS LSTM
# ============================================
print("\n9. ANALYSE DES RÉSIDUS LSTM")
print("-"*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, name in enumerate(target_cols):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    residuals = y_test_seq[:, i] - y_pred_lstm[:, i]
    ax.hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erreur nulle')
    ax.axvline(x=np.mean(residuals), color='green', linestyle='--', 
               label=f'Moyenne: {np.mean(residuals):.4f}')
    ax.set_xlabel('Résidu')
    ax.set_ylabel('Fréquence')
    ax.set_title(f'{name} - Distribution des résidus LSTM')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultats/lstm_residuals.png", dpi=150)
plt.show()
print(" Graphique sauvegardé: resultats/lstm_residuals.png")

# ============================================
# 10. TESTS DE ROBUSTESSE LSTM
# ============================================
print("\n10. TESTS DE ROBUSTESSE LSTM")
print("-"*60)

def test_noise_robustness(model, X_test, y_test, noise_levels=[0, 0.05, 0.1, 0.2, 0.3]):
    results = []
    for noise in noise_levels:
        X_noisy = X_test + np.random.normal(0, noise, X_test.shape)
        y_pred = model.predict(X_noisy, verbose=0)
        r2 = r2_score(y_test, y_pred)
        results.append({'noise': noise, 'r2': r2})
        print(f"   Bruit {noise*100:.0f}% → R²: {r2:.4f}")
    return results

def test_perturbation_rejection(model, X_test, y_test, magnitudes=[0, 0.5, 1.0, 2.0]):
    results = []
    mid = len(X_test) // 2
    for mag in magnitudes:
        X_pert = X_test.copy()
        X_pert[mid:mid+50] += mag
        y_pred = model.predict(X_pert, verbose=0)
        err_before = mean_squared_error(y_test[:mid], y_pred[:mid])
        err_after = mean_squared_error(y_test[mid:], y_pred[mid:])
        ratio = err_after / err_before if err_before > 0 else 1
        results.append({'magnitude': mag, 'ratio': ratio})
        print(f"   Magnitude {mag} → Ratio: {ratio:.3f}")
    return results

print("\n Test robustesse au bruit LSTM:")
noise_results = test_noise_robustness(model, X_test_seq, y_test_seq)

print("\n Test rejet de perturbation LSTM:")
pert_results = test_perturbation_rejection(model, X_test_seq, y_test_seq)

# Visualisation robustesse
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

noise_vals = [r['noise'] for r in noise_results]
r2_vals = [r['r2'] for r in noise_results]
axes[0].plot(noise_vals, r2_vals, 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Niveau de bruit')
axes[0].set_ylabel('R²')
axes[0].set_title('Robustesse au bruit - LSTM')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=pid_r2, color='red', linestyle='--', label=f'PID réel (R²={pid_r2:.3f})')
axes[0].legend()

mag_vals = [r['magnitude'] for r in pert_results]
ratio_vals = [r['ratio'] for r in pert_results]
axes[1].plot(mag_vals, ratio_vals, 's-', linewidth=2, markersize=8, color='red')
axes[1].axhline(y=1.0, color='green', linestyle='--', label='Performance nominale')
axes[1].set_xlabel('Magnitude perturbation')
axes[1].set_ylabel('Ratio dégradation')
axes[1].set_title('Rejet de perturbations - LSTM')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultats/lstm_robustness.png", dpi=150)
plt.show()
print(" Graphique sauvegardé: resultats/lstm_robustness.png")

# ============================================
# 11. COMPARAISON VISUELLE LSTM vs PID
# ============================================
print("\n11. COMPARAISON VISUELLE LSTM vs PID")
print("-"*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, name in enumerate(target_cols):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    sample_size = min(500, len(y_test_seq))
    ax.plot(y_test_seq[:sample_size, i], label='Réel', linewidth=1.5, alpha=0.7)
    ax.plot(y_pred_lstm[:sample_size, i], label='LSTM', linewidth=1.5, alpha=0.7)
    ax.plot(pid_pred[:sample_size, i], label='PID réel', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Pas de temps')
    ax.set_ylabel(f'{name}')
    ax.set_title(f'{name} - Comparaison LSTM vs PID réel')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultats/lstm_vs_pid_comparison.png", dpi=150)
plt.show()
print(" Graphique sauvegardé: resultats/lstm_vs_pid_comparison.png")

# ============================================
# 12. SAUVEGARDE DES RÉSULTATS
# ============================================
print("\n12. SAUVEGARDE DES RÉSULTATS")
print("-"*60)

comparison_results = {
    'lstm': {
        'r2': float(lstm_r2),
        'mae': float(lstm_mae),
        'mse': float(lstm_mse),
        'per_command': {
            target_cols[i]: {
                'r2': float(r2_score(y_test_seq[:, i], y_pred_lstm[:, i])),
                'mae': float(mean_absolute_error(y_test_seq[:, i], y_pred_lstm[:, i]))
            } for i in range(len(target_cols))
        },
        'robustness': {
            'noise': noise_results,
            'perturbation': pert_results
        }
    },
    'pid': {
        'r2': float(pid_r2),
        'mae': float(pid_mae),
        'mse': float(pid_mse),
        'per_command': {
            target_cols[i]: {
                'r2': float(r2_score(y_test_seq[:, i], pid_pred[:, i])),
                'mae': float(mean_absolute_error(y_test_seq[:, i], pid_pred[:, i]))
            } for i in range(len(target_cols))
        }
    },
    'difference': {
        'r2': float(difference_r2),
        'winner': 'LSTM' if difference_r2 > 0 else 'PID'
    }
}

with open('resultats/comparison_lstm_vs_pid.json', 'w') as f:
    json.dump(comparison_results, f, indent=2)

print(" Résultats sauvegardés: resultats/comparison_lstm_vs_pid.json")

# ============================================
# 13. RAPPORT FINAL
# ============================================
print("\n" + "="*80)
print(" RAPPORT FINAL - COMPARAISON LSTM vs PID")
print("="*80)

print(f"""
LSTM:     R² = {lstm_r2:.4f} | MAE = {lstm_mae:.4f}
PID réel: R² = {pid_r2:.4f} | MAE = {pid_mae:.4f}
Différence: {abs(difference_r2):.4f} en faveur de {'LSTM' if difference_r2 > 0 else 'PID'}
Verdict: {' LSTM surpasse le PID réel' if difference_r2 > 0 else '⚠️ PID réel reste supérieur'}
""")

print("\n COMPARAISON LSTM vs PID RÉEL TERMINÉE AVEC SUCCÈS !")