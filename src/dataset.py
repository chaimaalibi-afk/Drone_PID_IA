# generate_advanced_dataset.py
import numpy as np
import pandas as pd
import os
from scipy import signal

class AdvancedDroneDatasetGenerator:
    def __init__(self, dt=0.02):
        self.dt = dt
        self.gravity = 9.81
        
        # Plages de variations pour la robustesse
        self.mass_range = (1.2, 1.6)  # kg (variation de masse)
        self.wind_speed_range = (0, 5)  # m/s (vent)
        self.turbulence_range = (0, 0.5)  # intensité turbulences
        
        # Gains PID (variations pour diversité)
        self.kp_range = (0.5, 1.5)
        self.ki_range = (0.01, 0.05)
        self.kd_range = (0.3, 0.8)
        
        # Bruit des capteurs (réaliste)
        self.gps_noise_std = 0.05  # m
        self.accel_noise_std = 0.03  # m/s²
        self.gyro_noise_std = 0.005  # rad/s
        
        self.reset()
        self.states = []
        self.actions = []
        self.metadata = []
        
    def reset(self, mass=None, wind_speed=None, kp=None, ki=None, kd=None):
        """Réinitialise avec paramètres aléatoires"""
        
        # Paramètres variables pour diversité
        self.mass = mass if mass else np.random.uniform(*self.mass_range)
        self.wind_speed = wind_speed if wind_speed else np.random.uniform(*self.wind_speed_range)
        self.turbulence = np.random.uniform(*self.turbulence_range)
        
        # Gains PID variables
        self.Kp_pos = kp if kp else np.random.uniform(*self.kp_range)
        self.Ki_pos = ki if ki else np.random.uniform(*self.ki_range)
        self.Kd_pos = kd if kd else np.random.uniform(*self.kd_range)
        
        self.Kp_vel = self.Kp_pos * 0.8
        self.Ki_vel = self.Ki_pos * 0.5
        self.Kd_vel = self.Kd_pos * 0.6
        
        # Position initiale aléatoire
        self.position = np.array([
            np.random.uniform(-8, 8),
            np.random.uniform(-8, 8),
            np.random.uniform(1.5, 4)
        ])
        self.velocity = np.array([0.0, 0.0, 0.0])
        
        # Accumulateurs PID
        self.integral_pos = np.zeros(3)
        self.prev_error_pos = np.zeros(3)
        self.integral_vel = np.zeros(3)
        self.prev_error_vel = np.zeros(3)
        
        # Compteur de temps pour le vent
        self.time = 0
        
    def get_state(self):
        return {
            'position_x': self.position[0],
            'position_y': self.position[1],
            'position_z': self.position[2],
            'velocity_x': self.velocity[0],
            'velocity_y': self.velocity[1],
            'velocity_z': self.velocity[2],
            'mass': self.mass,
            'wind_speed': self.wind_speed
        }
    
    def generate_trajectory(self, traj_type, duration=60):
        """
        Génère une trajectoire de référence
        traj_type: 'hover', 'line', 'circle', 'figure_eight', 
                   'sine', 'spiral', 'clover', 'random'
        """
        t = np.arange(0, duration, self.dt)
        
        if traj_type == 'hover':
            # Vol stationnaire
            x_ref = np.zeros_like(t)
            y_ref = np.zeros_like(t)
            z_ref = 2.5 * np.ones_like(t)
            
        elif traj_type == 'line':
            # Ligne droite
            x_ref = np.linspace(-5, 5, len(t))
            y_ref = np.zeros_like(t)
            z_ref = 2.5 * np.ones_like(t)
            
        elif traj_type == 'circle':
            # Cercle
            radius = 4.0
            omega = 0.4
            x_ref = radius * np.cos(omega * t)
            y_ref = radius * np.sin(omega * t)
            z_ref = 2.5 * np.ones_like(t)
            
        elif traj_type == 'figure_eight':
            # Figure en 8
            a, b = 4.0, 3.0
            omega = 0.3
            x_ref = a * np.sin(omega * t)
            y_ref = b * np.sin(2 * omega * t)
            z_ref = 2.5 * np.ones_like(t)
            
        elif traj_type == 'sine':
            # Trajectoire sinusoïdale
            x_ref = t * 0.2
            y_ref = 3 * np.sin(0.8 * t)
            z_ref = 2.5 + 0.5 * np.sin(0.5 * t)
            
        elif traj_type == 'spiral':
            # Spirale montante
            radius = 3 * (1 - t / duration)
            omega = 1.5
            x_ref = radius * np.cos(omega * t)
            y_ref = radius * np.sin(omega * t)
            z_ref = 2.5 + 3 * t / duration
            
        elif traj_type == 'clover':
            # Trèfle à 4 feuilles
            omega = 0.5
            x_ref = 3 * np.sin(omega * t) * np.cos(omega * t)
            y_ref = 3 * np.sin(omega * t) * np.sin(2 * omega * t)
            z_ref = 2.5 * np.ones_like(t)
            
        else:  # random
            # Waypoints aléatoires lissés
            n_waypoints = 10
            waypoints_x = np.random.uniform(-6, 6, n_waypoints)
            waypoints_y = np.random.uniform(-6, 6, n_waypoints)
            waypoints_z = np.random.uniform(2, 4, n_waypoints)
            
            t_waypoints = np.linspace(0, duration, n_waypoints)
            x_ref = np.interp(t, t_waypoints, waypoints_x)
            y_ref = np.interp(t, t_waypoints, waypoints_y)
            z_ref = np.interp(t, t_waypoints, waypoints_z)
        
        return np.column_stack([x_ref, y_ref, z_ref]), t
    
    def wind_force(self, t):
        """Génère une force de vent réaliste"""
        # Vent constant + rafales + turbulence
        wind_dir = np.random.uniform(0, 2*np.pi)
        wind_const = self.wind_speed * np.array([np.cos(wind_dir), np.sin(wind_dir), 0])
        
        # Rafales périodiques
        gust = 2.0 * np.exp(-((t % 30 - 15)**2) / 100) * np.random.normal(0, 1, 3)
        
        # Turbulence (bruit coloré)
        turbulence = self.turbulence * np.random.normal(0, 1, 3)
        
        return wind_const + gust + turbulence
    
    def pid_controller(self, current, target, dt, Kp, Ki, Kd, integral, prev_error):
        error = target - current
        integral += error * dt
        integral = np.clip(integral, -2.0, 2.0)
        derivative = (error - prev_error) / dt if dt > 0 else 0
        control = Kp * error + Ki * integral + Kd * derivative
        return control, integral, error
    
    def compute_command(self, target_pos, target_vel):
        """Calcule les commandes PID avec paramètres actuels"""
        
        # Contrôle en position
        pos_command, self.integral_pos, self.prev_error_pos = self.pid_controller(
            self.position, target_pos, self.dt,
            self.Kp_pos, self.Ki_pos, self.Kd_pos,
            self.integral_pos, self.prev_error_pos
        )
        
        # Contrôle en vitesse
        vel_command, self.integral_vel, self.prev_error_vel = self.pid_controller(
            self.velocity, target_vel, self.dt,
            self.Kp_vel, self.Ki_vel, self.Kd_vel,
            self.integral_vel, self.prev_error_vel
        )
        
        # THRUST (avec compensation de masse variable)
        thrust_base = self.mass * self.gravity
        thrust_adjust = pos_command[2] * self.mass
        thrust = np.clip(thrust_base + thrust_adjust, 6.0, 22.0)
        
        # ROLL et PITCH
        roll = np.clip(pos_command[1] * 0.08, -0.5, 0.5)
        pitch = np.clip(pos_command[0] * 0.08, -0.5, 0.5)
        yaw = np.clip(vel_command[0] * 0.5, -0.8, 0.8)
        
        return {'thrust': thrust, 'roll': roll, 'pitch': pitch, 'yaw': yaw}
    
    def update_dynamics(self, command, t):
        """Met à jour la dynamique avec vent et bruit"""
        roll = command['roll']
        pitch = command['pitch']
        thrust_total = command['thrust']
        
        # Forces du vent
        wind = self.wind_force(t)
        
        # Accélérations avec vent
        ax = -np.sin(pitch) * (thrust_total / self.mass) + wind[0] / self.mass
        ay = np.sin(roll) * (thrust_total / self.mass) + wind[1] / self.mass
        az = (thrust_total / self.mass) - self.gravity + wind[2] / self.mass
        
        # Mise à jour Euler
        self.velocity += np.array([ax, ay, az]) * self.dt
        self.position += self.velocity * self.dt
        
        # Limites
        self.position = np.clip(self.position, [-12, -12, 0.5], [12, 12, 8])
        
        # Traînée aérodynamique
        self.velocity *= 0.998
        
        # Bruit de capteurs (mesure)
        self.velocity += np.random.normal(0, self.accel_noise_std, 3)
        self.position += np.random.normal(0, self.gps_noise_std, 3)
        
        self.time += self.dt
    
    def generate_trajectory_with_variations(self, traj_type, duration=60):
        """Génère une trajectoire complète avec variations de paramètres"""
        
        # Réinitialiser avec paramètres aléatoires
        self.reset()
        
        # Cible à poursuivre
        target_traj, t = self.generate_trajectory(traj_type, duration)
        
        # Vitesse cible (dérivée numérique)
        target_vel = np.gradient(target_traj, self.dt, axis=0)
        
        # Initialiser les cibles
        current_idx = 0
        self.target_pos = target_traj[current_idx]
        self.target_vel = target_vel[current_idx]
        
        # Stockage
        states = []
        actions = []
        
        for step in range(len(target_traj)):
            # Mettre à jour la cible
            self.target_pos = target_traj[step]
            self.target_vel = target_vel[step]
            
            # Calculer commande
            command = self.compute_command(self.target_pos, self.target_vel)
            
            # Sauvegarder
            state = self.get_state()
            states.append([
                state['position_x'], state['position_y'], state['position_z'],
                state['velocity_x'], state['velocity_y'], state['velocity_z']
            ])
            actions.append([
                command['thrust'], command['roll'], command['pitch'], command['yaw']
            ])
            
            # Mettre à jour dynamique
            self.update_dynamics(command, step * self.dt)
        
        return np.array(states), np.array(actions), traj_type
    
    def generate_big_dataset(self, n_trajectories=200, duration=60):
        """
        Génère un grand dataset avec tous les types de trajectoires
        """
        traj_types = ['hover', 'line', 'circle', 'figure_eight', 
                      'sine', 'spiral', 'clover', 'random']
        
        all_states = []
        all_actions = []
        all_metadata = []
        
        total_samples = 0
        
        for i in range(n_trajectories):
            traj_type = np.random.choice(traj_types)
            print(f"Trajectoire {i+1}/{n_trajectories}: {traj_type}")
            
            states, actions, traj = self.generate_trajectory_with_variations(traj_type, duration)
            
            all_states.append(states)
            all_actions.append(actions)
            all_metadata.append({
                'trajectory': traj,
                'mass': self.mass,
                'wind_speed': self.wind_speed,
                'kp': self.Kp_pos
            })
            
            total_samples += len(states)
        
        # Concaténer
        states = np.vstack(all_states)
        actions = np.vstack(all_actions)
        
        print(f"\n Dataset généré: {total_samples} échantillons")
        print(f" {n_trajectories} trajectoires de {duration}s chacune")
        
        return states, actions, all_metadata
    
    def save_dataset(self, filename="data/drone_dataset_advanced.csv"):
        os.makedirs("data", exist_ok=True)
        states, actions, metadata = self.generate_big_dataset(n_trajectories=200, duration=60)
        
        df = pd.DataFrame(
            np.column_stack([states, actions]),
            columns=['position_x', 'position_y', 'position_z',
                    'velocity_x', 'velocity_y', 'velocity_z',
                    'thrust', 'roll', 'pitch', 'yaw']
        )
        
        df.to_csv(filename, index=False)
        
        print(f"\n Dataset sauvegardé: {filename}")
        print(f" {len(df)} échantillons")
        print(f" Taille: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
        
        # Statistiques
        print(f"\n STATISTIQUES:")
        print(f"   Thrust: min={df['thrust'].min():.2f}, max={df['thrust'].max():.2f}, std={df['thrust'].std():.3f}")
        print(f"   Roll:   min={df['roll'].min():.3f}, max={df['roll'].max():.3f}, std={df['roll'].std():.3f}")
        print(f"   Pitch:  min={df['pitch'].min():.3f}, max={df['pitch'].max():.3f}, std={df['pitch'].std():.3f}")
        print(f"   Yaw:    min={df['yaw'].min():.3f}, max={df['yaw'].max():.3f}, std={df['yaw'].std():.3f}")
        
        return df, metadata


if __name__ == "__main__":
    print("="*80)
    print(" GÉNÉRATION DE DATASET AVANCÉ POUR LSTM")
    print("="*80)
    print()
    print(" Caractéristiques du dataset:")
    print("   - 8 types de trajectoires (hover, line, circle, figure_eight, sine, spiral, clover, random)")
    print("   - Variations de masse (1.2-1.6 kg)")
    print("   - Vent et turbulences (0-5 m/s)")
    print("   - Variations des gains PID")
    print("   - Bruit de capteurs réaliste")
    print("   - 200 trajectoires × 60s = ~600 000 échantillons")
    print()
    
    generator = AdvancedDroneDatasetGenerator(dt=0.02)
    df, metadata = generator.save_dataset("data/drone_dataset_advanced.csv")
    
    print("\n GÉNÉRATION TERMINÉE!")
    print("\n Ce dataset est optimisé pour LSTM:")
    print("   ✓ Grand nombre d'échantillons (~600k)")
    print("   ✓ Trajectoires variées (8 types)")
    print("   ✓ Paramètres variables (robustesse)")
    print("   ✓ Bruit et perturbations réalistes")