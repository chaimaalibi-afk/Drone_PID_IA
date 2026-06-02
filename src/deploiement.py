# ============================================
# DASHBOARD STREAMLIT - DEPLOIEMENT DRONE
# PID vs LSTM avec visualisation 3D professionnelle
# Execution : streamlit run deploiement.py
# ============================================

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:  # TensorFlow peut etre indisponible sur certaines machines.
    tf = None
    keras = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_lstm_model.h5"
SCALER_X_PATH = PROJECT_DIR / "models" / "scaler_X_enhanced.pkl"
SCALER_Y_PATH = PROJECT_DIR / "models" / "scaler_y.pkl"

DT = 0.02
MASS = 1.38
G = 9.81
SEQ_LENGTH = 50
WINDOW_SIZE = 10

BASE_FEATURES = [
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
]
TARGET_COLUMNS = ["thrust", "roll", "pitch", "yaw"]
TRAJECTORY_MODES = [
    "Point fixe",
    "Ligne droite",
    "Cercle puis cible",
    "Arc",
    "Sinusoide",
    "Spirale",
    "Figure huit",
]


st.set_page_config(
    page_title="Drone PID vs LSTM",
    page_icon=":material/flight:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Matrice de rotation ZYX pour roll, pitch, yaw en radians."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def transform_points(points: np.ndarray, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    rot = rotation_matrix(float(orientation[0]), float(orientation[1]), float(orientation[2]))
    return points @ rot.T + position


def box_mesh(center: np.ndarray, size: tuple[float, float, float]) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    lx, ly, lz = size
    x, y, z = lx / 2, ly / 2, lz / 2
    vertices = np.array(
        [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ],
        dtype=float,
    )
    vertices += center
    i = [0, 0, 0, 1, 1, 2, 4, 4, 5, 5, 6, 7]
    j = [1, 2, 4, 2, 5, 3, 5, 6, 6, 1, 7, 4]
    k = [2, 3, 5, 5, 6, 7, 6, 7, 1, 0, 4, 3]
    return vertices, i, j, k


def ellipse_points(center: np.ndarray, radius_x: float, radius_y: float, n: int = 48) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n)
    return np.column_stack(
        [
            center[0] + radius_x * np.cos(theta),
            center[1] + radius_y * np.sin(theta),
            center[2] + np.zeros_like(theta),
        ]
    )


def blade_mesh(center: np.ndarray, angle: float, length: float = 0.55, width: float = 0.08) -> np.ndarray:
    direction = np.array([np.cos(angle), np.sin(angle), 0.0])
    normal = np.array([-np.sin(angle), np.cos(angle), 0.0])
    return np.array(
        [
            center - direction * length - normal * width,
            center + direction * length - normal * width,
            center + direction * length + normal * width,
            center - direction * length + normal * width,
        ]
    )


def create_drone_mesh(position, orientation):
    """
    position: [x, y, z] centre du drone
    orientation: [roll, pitch, yaw] en radians
    Retourne les traces Plotly du corps, des bras et des helices.
    """
    position = np.asarray(position, dtype=float)
    orientation = np.asarray(orientation, dtype=float)
    traces = []

    body_vertices, i, j, k = box_mesh(np.array([0.0, 0.0, 0.0]), (0.7, 0.42, 0.22))
    body_world = transform_points(body_vertices, position, orientation)
    traces.append(
        go.Mesh3d(
            x=body_world[:, 0],
            y=body_world[:, 1],
            z=body_world[:, 2],
            i=i,
            j=j,
            k=k,
            color="#2b3440",
            opacity=1,
            flatshading=True,
            name="Corps drone",
            showscale=False,
        )
    )

    nose_local = np.array([[0.32, -0.22, 0.13], [0.32, 0.22, 0.13], [0.58, 0.0, 0.05]])
    nose_world = transform_points(nose_local, position, orientation)
    traces.append(
        go.Mesh3d(
            x=nose_world[:, 0],
            y=nose_world[:, 1],
            z=nose_world[:, 2],
            i=[0],
            j=[1],
            k=[2],
            color="#f97316",
            opacity=1,
            name="Avant",
            showscale=False,
        )
    )

    motor_centers = np.array(
        [
            [0.95, 0.75, 0.03],
            [0.95, -0.75, 0.03],
            [-0.95, 0.75, 0.03],
            [-0.95, -0.75, 0.03],
        ]
    )

    arm_pairs = [([0.0, 0.0, 0.0], motor_centers[idx]) for idx in range(4)]
    for start, end in arm_pairs:
        arm = transform_points(np.array([start, end], dtype=float), position, orientation)
        traces.append(
            go.Scatter3d(
                x=arm[:, 0],
                y=arm[:, 1],
                z=arm[:, 2],
                mode="lines",
                line=dict(color="#374151", width=8),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    spin = time.time() * 16
    for idx, center in enumerate(motor_centers):
        motor_vertices, mi, mj, mk = box_mesh(center + np.array([0.0, 0.0, -0.04]), (0.18, 0.18, 0.12))
        motor_world = transform_points(motor_vertices, position, orientation)
        traces.append(
            go.Mesh3d(
                x=motor_world[:, 0],
                y=motor_world[:, 1],
                z=motor_world[:, 2],
                i=mi,
                j=mj,
                k=mk,
                color="#111827",
                opacity=1,
                name=f"Moteur {idx + 1}",
                showscale=False,
            )
        )

        ring = transform_points(ellipse_points(center, 0.52, 0.52), position, orientation)
        traces.append(
            go.Scatter3d(
                x=ring[:, 0],
                y=ring[:, 1],
                z=ring[:, 2],
                mode="lines",
                line=dict(color="#94a3b8", width=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        for angle in (spin + idx * 0.4, spin + np.pi / 2 + idx * 0.4):
            blade = transform_points(blade_mesh(center, angle), position, orientation)
            traces.append(
                go.Mesh3d(
                    x=blade[:, 0],
                    y=blade[:, 1],
                    z=blade[:, 2],
                    i=[0, 0],
                    j=[1, 2],
                    k=[2, 3],
                    color="#38bdf8",
                    opacity=0.82,
                    name="Helice",
                    showscale=False,
                    showlegend=False,
                )
            )

    return traces


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = None
    scaler_X = None
    scaler_y = None
    errors = []

    if tf is None or keras is None:
        errors.append("TensorFlow n'est pas disponible.")
    elif MODEL_PATH.exists():
        try:
            model = keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as exc:
            try:
                model = build_lstm_model((SEQ_LENGTH, 78))
                model.load_weights(MODEL_PATH)
            except Exception as weight_exc:
                errors.append(f"Impossible de charger le LSTM: {exc} | {weight_exc}")
    else:
        errors.append(f"Modele absent: {MODEL_PATH}")

    try:
        scaler_X = joblib.load(SCALER_X_PATH)
    except Exception as exc:
        errors.append(f"Impossible de charger scaler_X_enhanced.pkl: {exc}")

    try:
        scaler_y = joblib.load(SCALER_Y_PATH)
    except Exception as exc:
        errors.append(f"Impossible de charger scaler_y.pkl: {exc}")

    return model, scaler_X, scaler_y, errors


def build_lstm_model(input_shape):
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LSTM(32, return_sequences=False, dropout=0.2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(4)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


@dataclass
class Command:
    thrust: float
    roll: float
    pitch: float
    yaw: float

    def as_array(self) -> np.ndarray:
        return np.array([self.thrust, self.roll, self.pitch, self.yaw], dtype=float)


class DroneSimulator:
    def __init__(self, dt: float = DT, mass: float = MASS, gravity: float = G):
        self.dt = dt
        self.mass = mass
        self.g = gravity
        self.reset()

    def reset(self, initial_pos=(0.0, 0.0, 2.0)):
        self.position = np.array(initial_pos, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.orientation = np.zeros(3, dtype=float)
        self.time = 0.0
        self.trajectory = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.orientation_history = [self.orientation.copy()]
        self.command_history = [Command(self.mass * self.g, 0.0, 0.0, 0.0)]
        self.target_history = []

    def state_vector(self) -> np.ndarray:
        return np.r_[self.position, self.velocity].astype(float)

    def apply(self, command: Command):
        thrust = float(np.clip(command.thrust, 6.0, 22.0))
        roll = float(np.clip(command.roll, -0.5, 0.5))
        pitch = float(np.clip(command.pitch, -0.5, 0.5))
        yaw = float(np.clip(command.yaw, -0.8, 0.8))

        ax = -np.sin(pitch) * (thrust / self.mass)
        ay = np.sin(roll) * (thrust / self.mass)
        az = (thrust / self.mass) - self.g

        self.velocity += np.array([ax, ay, az]) * self.dt
        self.velocity = np.clip(self.velocity, -5.0, 5.0)
        self.position += self.velocity * self.dt
        self.position = np.clip(self.position, [-12.0, -12.0, 0.5], [12.0, 12.0, 8.0])
        self.velocity *= 0.998
        self.orientation = np.array([roll, pitch, yaw], dtype=float)
        self.time += self.dt

        self.trajectory.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        self.orientation_history.append(self.orientation.copy())
        self.command_history.append(Command(thrust, roll, pitch, yaw))

        if len(self.trajectory) > 2500:
            self.trajectory = self.trajectory[-2500:]
            self.velocity_history = self.velocity_history[-2500:]
            self.orientation_history = self.orientation_history[-2500:]
            self.command_history = self.command_history[-2500:]


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float = DT):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)

    def update_gains(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def compute(self, position: np.ndarray, target: np.ndarray) -> Command:
        error = target - position
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -2.0, 2.0)
        derivative = (error - self.prev_error) / self.dt
        pid = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error.copy()

        thrust = np.clip(MASS * G + pid[2] * MASS, 6.0, 22.0)
        roll = np.clip(pid[1] * 0.08, -0.5, 0.5)
        pitch = np.clip(-pid[0] * 0.08, -0.5, 0.5)
        yaw = np.clip(0.08 * np.arctan2(error[1], error[0]), -0.8, 0.8)
        return Command(float(thrust), float(roll), float(pitch), float(yaw))


class LSTMController:
    def __init__(self, model, scaler_X, scaler_y, seq_length: int = SEQ_LENGTH):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.seq_length = seq_length
        self.raw_history: list[np.ndarray] = []

    def reset(self):
        self.raw_history = []

    def _enriched_features(self) -> np.ndarray:
        data = np.array(self.raw_history, dtype=float)
        current = data[-1]
        features = list(current)

        for feature_idx in range(len(BASE_FEATURES)):
            series = data[:, feature_idx]
            for lag in range(1, WINDOW_SIZE + 1):
                features.append(series[-lag])
            features.append(np.mean(series[-WINDOW_SIZE:]))
            features.append((series[-1] - series[-2]) / DT)

        enriched = np.array(features, dtype=float)
        if enriched.shape[0] != 78:
            raise ValueError(f"Le LSTM attend 78 features, recu {enriched.shape[0]}.")
        return enriched

    def compute(self, drone: DroneSimulator, fallback_pid: PIDController, target: np.ndarray) -> Command:
        self.raw_history.append(drone.state_vector())
        guide = fallback_pid.compute(drone.position, target)
        if len(self.raw_history) > self.seq_length + WINDOW_SIZE + 2:
            self.raw_history = self.raw_history[-(self.seq_length + WINDOW_SIZE + 2) :]

        if len(self.raw_history) < self.seq_length + WINDOW_SIZE:
            return guide

        sequence = []
        full_history = list(self.raw_history)
        for offset in range(self.seq_length, 0, -1):
            self.raw_history = full_history[: len(full_history) - offset + 1]
            sequence.append(self._enriched_features())
        self.raw_history = full_history

        X = np.array(sequence, dtype=float)
        X_scaled = self.scaler_X.transform(X).reshape(1, self.seq_length, 78)
        pred_scaled = self.model.predict(X_scaled, verbose=0)
        pred = self.scaler_y.inverse_transform(pred_scaled)[0]

        lstm = Command(
            thrust=float(np.clip(pred[0], 6.0, 22.0)),
            roll=float(np.clip(pred[1], -0.5, 0.5)),
            pitch=float(np.clip(pred[2], -0.5, 0.5)),
            yaw=float(np.clip(pred[3], -0.8, 0.8)),
        )

        # Le LSTM entraine ici ne recoit pas la cible en entree. Cette correction
        # conserve sa prediction comme feed-forward, puis ajoute la poursuite cible.
        guide_weight = 0.94
        lstm_weight = 1.0 - guide_weight
        return Command(
            thrust=float(np.clip(guide_weight * guide.thrust + lstm_weight * lstm.thrust, 6.0, 22.0)),
            roll=float(np.clip(guide_weight * guide.roll + lstm_weight * lstm.roll, -0.5, 0.5)),
            pitch=float(np.clip(guide_weight * guide.pitch + lstm_weight * lstm.pitch, -0.5, 0.5)),
            yaw=float(np.clip(guide_weight * guide.yaw + lstm_weight * lstm.yaw, -0.8, 0.8)),
        )


def initialize_session():
    if "running" not in st.session_state:
        st.session_state.running = False
    if "drone_pid" not in st.session_state:
        st.session_state.drone_pid = DroneSimulator()
    if "drone_lstm" not in st.session_state:
        st.session_state.drone_lstm = DroneSimulator()
    if "pid_controller" not in st.session_state:
        st.session_state.pid_controller = PIDController(0.8, 0.02, 0.5)
    if "lstm_fallback_pid" not in st.session_state:
        st.session_state.lstm_fallback_pid = PIDController(0.8, 0.02, 0.5)
    if "lstm_controller" not in st.session_state:
        st.session_state.lstm_controller = None
    if "reference_start" not in st.session_state:
        st.session_state.reference_start = np.array([0.0, 0.0, 2.0], dtype=float)


def reset_simulation():
    st.session_state.running = False
    st.session_state.drone_pid = DroneSimulator()
    st.session_state.drone_lstm = DroneSimulator()
    st.session_state.pid_controller = PIDController(
        st.session_state.kp,
        st.session_state.ki,
        st.session_state.kd,
    )
    st.session_state.lstm_fallback_pid = PIDController(
        st.session_state.kp,
        st.session_state.ki,
        st.session_state.kd,
    )
    if st.session_state.lstm_controller is not None:
        st.session_state.lstm_controller.reset()
    st.session_state.reference_start = np.array([0.0, 0.0, 2.0], dtype=float)


def smoothstep(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress * progress * (3.0 - 2.0 * progress)


def perpendicular_xy(vector: np.ndarray) -> np.ndarray:
    horizontal = np.array([vector[0], vector[1], 0.0], dtype=float)
    norm = np.linalg.norm(horizontal[:2])
    if norm < 1e-6:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return np.array([-horizontal[1], horizontal[0], 0.0], dtype=float) / norm


def reference_target(
    final_target: np.ndarray,
    elapsed_time: float,
    mode: str,
    duration: float,
    start: np.ndarray,
) -> np.ndarray:
    """Cible mobile suivie par les controleurs pendant la simulation."""
    if mode == "Point fixe":
        return final_target.copy()

    progress = smoothstep(elapsed_time / max(duration, DT))
    delta = final_target - start
    base = start + progress * delta
    perp = perpendicular_xy(delta)
    distance = max(float(np.linalg.norm(delta[:2])), 1.0)
    amplitude = min(3.0, 0.35 * distance)

    if mode == "Ligne droite":
        return base

    if mode == "Cercle puis cible":
        circle_part = 0.68
        center = (start + final_target) / 2.0
        radius_vec = start - center
        radius = max(float(np.linalg.norm(radius_vec[:2])), 1.2)
        if np.linalg.norm(radius_vec[:2]) < 1e-6:
            radius_vec = np.array([radius, 0.0, 0.0])
        else:
            radius_vec = np.array([radius_vec[0], radius_vec[1], 0.0])
            radius_vec = radius_vec / np.linalg.norm(radius_vec[:2]) * radius
        perp = perpendicular_xy(radius_vec)

        if progress <= circle_part:
            p = progress / circle_part
            angle = 2.0 * np.pi * p
            circle = center + radius_vec * np.cos(angle) + perp * radius * np.sin(angle)
            circle[2] = start[2] + 0.45 * np.sin(2.0 * np.pi * p)
            return np.clip(circle, [-12.0, -12.0, 0.5], [12.0, 12.0, 8.0])

        p = smoothstep((progress - circle_part) / (1.0 - circle_part))
        return np.clip(start + p * (final_target - start), [-12.0, -12.0, 0.5], [12.0, 12.0, 8.0])

    if mode == "Arc":
        arc = perp * amplitude * np.sin(np.pi * progress)
        arc[2] = 0.8 * np.sin(np.pi * progress)
        return np.clip(base + arc, [-12.0, -12.0, 0.5], [12.0, 12.0, 8.0])

    if mode == "Sinusoide":
        wave = perp * amplitude * np.sin(4.0 * np.pi * progress)
        wave[2] = 0.45 * np.sin(2.0 * np.pi * progress)
        return np.clip(base + wave, [-12.0, -12.0, 0.5], [12.0, 12.0, 8.0])

    if mode == "Spirale":
        radius = amplitude * np.sin(np.pi * progress)
        angle = 5.0 * np.pi * progress
        swirl = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.6 * np.sin(np.pi * progress)])
        return np.clip(base + swirl, [-12.0, -12.0, 0.5], [12.0, 12.0, 8.0])

    if mode == "Figure huit":
        eight = perp * amplitude * np.sin(2.0 * np.pi * progress)
        eight += np.array([delta[0], delta[1], 0.0]) * 0.12 * np.sin(4.0 * np.pi * progress)
        eight[2] = 0.45 * np.sin(2.0 * np.pi * progress)
        return np.clip(base + eight, [-12.0, -12.0, 0.5], [12.0, 12.0, 8.0])

    return final_target.copy()


def reference_path(final_target: np.ndarray, mode: str, duration: float, start: np.ndarray, samples: int = 140) -> np.ndarray:
    times = np.linspace(0.0, duration, samples)
    return np.array([reference_target(final_target, t, mode, duration, start) for t in times])


def make_scene(
    drone: DroneSimulator,
    target: np.ndarray,
    title: str,
    upto: int | None = None,
    final_target: np.ndarray | None = None,
    reference_curve: np.ndarray | None = None,
    camera=None,
) -> go.Figure:
    safe_len = max(1, len(drone.trajectory))
    safe_upto = safe_len if upto is None else int(np.clip(upto, 1, safe_len))
    current_idx = safe_upto - 1
    trajectory = np.array(drone.trajectory[:safe_upto], dtype=float)
    pos = drone.position if upto is None else drone.trajectory[current_idx]
    orient = drone.orientation if upto is None else drone.orientation_history[min(current_idx, len(drone.orientation_history) - 1)]

    fig = go.Figure()
    if reference_curve is not None and len(reference_curve) > 1:
        fig.add_trace(
            go.Scatter3d(
                x=reference_curve[:, 0],
                y=reference_curve[:, 1],
                z=reference_curve[:, 2],
                mode="lines",
                line=dict(color="#16a34a", width=4, dash="dash"),
                name="Reference",
            )
        )

    if len(trajectory) > 1:
        fig.add_trace(
            go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode="lines",
                line=dict(color="#2563eb", width=5),
                name="Trajectoire",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[target[0]],
            y=[target[1]],
            z=[target[2]],
            mode="markers",
            marker=dict(size=10, color="#22c55e", symbol="circle"),
            name="Cible courante",
        )
    )

    if final_target is not None and np.linalg.norm(final_target - target) > 1e-6:
        fig.add_trace(
            go.Scatter3d(
                x=[final_target[0]],
                y=[final_target[1]],
                z=[final_target[2]],
                mode="markers",
                marker=dict(size=8, color="#15803d", symbol="diamond"),
                name="Destination finale",
            )
        )

    for trace in create_drone_mesh(pos, orient):
        fig.add_trace(trace)

    fig.update_layout(
        title=title,
        height=560,
        margin=dict(l=0, r=0, t=42, b=0),
        showlegend=False,
        scene=dict(
            xaxis=dict(title="X (m)", range=[-12, 12], backgroundcolor="#f8fafc"),
            yaxis=dict(title="Y (m)", range=[-12, 12], backgroundcolor="#f8fafc"),
            zaxis=dict(title="Z (m)", range=[0, 8], backgroundcolor="#f8fafc"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.55),
            camera=camera or dict(eye=dict(x=1.45, y=-1.55, z=1.05)),
        ),
    )
    return fig


def make_angles_chart(drone_pid: DroneSimulator, drone_lstm: DroneSimulator | None, compare: bool) -> go.Figure:
    fig = go.Figure()
    drones = [("PID", drone_pid, "#2563eb")]
    if compare and drone_lstm is not None:
        drones.append(("LSTM", drone_lstm, "#f97316"))

    for label, drone, color in drones:
        angles = np.rad2deg(np.array(drone.orientation_history, dtype=float))
        t = np.arange(len(angles)) * DT
        fig.add_trace(go.Scatter(x=t, y=angles[:, 0], name=f"Roll {label}", line=dict(color=color, width=2)))
        fig.add_trace(
            go.Scatter(
                x=t,
                y=angles[:, 1],
                name=f"Pitch {label}",
                line=dict(color=color, width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=angles[:, 2],
                name=f"Yaw {label}",
                line=dict(color=color, width=2, dash="dot"),
            )
        )

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=28, b=20),
        xaxis_title="Temps (s)",
        yaxis_title="Angle (degres)",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def metric_panel(label: str, drone: DroneSimulator, target: np.ndarray):
    cmd = drone.command_history[-1]
    angles = np.rad2deg(drone.orientation)
    error = np.linalg.norm(target - drone.position)

    st.subheader(label)
    c1, c2, c3 = st.columns(3)
    c1.metric("Erreur position", f"{error:.3f} m")
    c2.metric("Thrust", f"{cmd.thrust:.2f} N")
    c3.metric("Yaw", f"{cmd.yaw:.2f} rad")
    c1.metric("Roll", f"{cmd.roll:.2f} rad", f"{angles[0]:.1f} deg")
    c2.metric("Pitch", f"{cmd.pitch:.2f} rad", f"{angles[1]:.1f} deg")
    c3.metric("Position", f"{drone.position[0]:.1f}, {drone.position[1]:.1f}, {drone.position[2]:.1f}")


def run_steps(
    final_target: np.ndarray,
    control_mode: str,
    trajectory_mode: str,
    trajectory_duration: float,
    steps: int,
    lstm_ready: bool,
) -> np.ndarray:
    pid = st.session_state.pid_controller
    pid.update_gains(st.session_state.kp, st.session_state.ki, st.session_state.kd)

    fallback = st.session_state.lstm_fallback_pid
    fallback.update_gains(st.session_state.kp, st.session_state.ki, st.session_state.kd)

    current_target = final_target.copy()
    for _ in range(steps):
        elapsed_time = max(st.session_state.drone_pid.time, st.session_state.drone_lstm.time)
        current_target = reference_target(
            final_target,
            elapsed_time,
            trajectory_mode,
            trajectory_duration,
            st.session_state.reference_start,
        )

        if control_mode in ("PID", "Comparaison PID vs LSTM"):
            cmd_pid = pid.compute(st.session_state.drone_pid.position, current_target)
            st.session_state.drone_pid.apply(cmd_pid)

        if control_mode in ("IA LSTM", "Comparaison PID vs LSTM") and lstm_ready:
            cmd_lstm = st.session_state.lstm_controller.compute(
                st.session_state.drone_lstm,
                fallback,
                current_target,
            )
            st.session_state.drone_lstm.apply(cmd_lstm)

    return current_target


def main():
    st.title("Dashboard professionnel de controle drone")

    initialize_session()
    model, scaler_X, scaler_y, load_errors = load_artifacts()
    lstm_ready = model is not None and scaler_X is not None and scaler_y is not None

    if lstm_ready and st.session_state.lstm_controller is None:
        st.session_state.lstm_controller = LSTMController(model, scaler_X, scaler_y)

    with st.sidebar:
        st.header("Controle")
        available_modes = ["PID"]
        if lstm_ready:
            available_modes.extend(["IA LSTM", "Comparaison PID vs LSTM"])
        mode = st.radio("Mode", available_modes, horizontal=False)

        st.divider()
        st.subheader("Position cible")
        final_target = np.array(
            [
                st.slider("X cible", -10.0, 10.0, 3.0, 0.1),
                st.slider("Y cible", -10.0, 10.0, 4.0, 0.1),
                st.slider("Z cible", 0.8, 7.5, 3.0, 0.1),
            ],
            dtype=float,
        )

        st.divider()
        st.subheader("Trajectoire")
        trajectory_mode = st.selectbox("Type", TRAJECTORY_MODES, index=1)
        trajectory_duration = st.slider("Duree vers la cible (s)", 2.0, 30.0, 10.0, 0.5)

        st.divider()
        st.subheader("Gains PID")
        st.session_state.kp = st.slider("Kp", 0.05, 2.5, float(st.session_state.get("kp", 0.8)), 0.01)
        st.session_state.ki = st.slider("Ki", 0.0, 0.3, float(st.session_state.get("ki", 0.02)), 0.005)
        st.session_state.kd = st.slider("Kd", 0.0, 2.5, float(st.session_state.get("kd", 0.5)), 0.01)

        st.divider()
        speed = st.slider("Vitesse simulation", 1, 10, 4, 1)
        c1, c2 = st.columns(2)
        if c1.button("Demarrer", type="primary", use_container_width=True):
            active_drone = st.session_state.drone_lstm if mode == "IA LSTM" and lstm_ready else st.session_state.drone_pid
            st.session_state.reference_start = active_drone.position.copy()
            st.session_state.running = True
        if c2.button("Reinitialiser", use_container_width=True):
            reset_simulation()
            st.rerun()

    if load_errors:
        for error in load_errors:
            st.warning(error)
    if not lstm_ready:
        st.info("Le mode IA sera active des que le modele et les deux scalers seront chargeables.")

    elapsed_time = max(st.session_state.drone_pid.time, st.session_state.drone_lstm.time)
    current_target = reference_target(
        final_target,
        elapsed_time,
        trajectory_mode,
        trajectory_duration,
        st.session_state.reference_start,
    )
    ref_curve = reference_path(final_target, trajectory_mode, trajectory_duration, st.session_state.reference_start)

    if st.session_state.running:
        current_target = run_steps(
            final_target,
            mode,
            trajectory_mode,
            trajectory_duration,
            steps=speed,
            lstm_ready=lstm_ready,
        )

    compare = mode == "Comparaison PID vs LSTM" and lstm_ready
    if compare:
        col_pid, col_lstm = st.columns(2)
        with col_pid:
            st.plotly_chart(
                make_scene(
                    st.session_state.drone_pid,
                    current_target,
                    "PID",
                    final_target=final_target,
                    reference_curve=ref_curve,
                ),
                use_container_width=True,
            )
        with col_lstm:
            st.plotly_chart(
                make_scene(
                    st.session_state.drone_lstm,
                    current_target,
                    "LSTM",
                    final_target=final_target,
                    reference_curve=ref_curve,
                ),
                use_container_width=True,
            )
    elif mode == "IA LSTM" and lstm_ready:
        st.plotly_chart(
            make_scene(
                st.session_state.drone_lstm,
                current_target,
                "IA LSTM",
                final_target=final_target,
                reference_curve=ref_curve,
            ),
            use_container_width=True,
        )
    else:
        st.plotly_chart(
            make_scene(
                st.session_state.drone_pid,
                current_target,
                "PID",
                final_target=final_target,
                reference_curve=ref_curve,
            ),
            use_container_width=True,
        )

    if compare:
        m1, m2 = st.columns(2)
        with m1:
            metric_panel("Mesures PID", st.session_state.drone_pid, current_target)
        with m2:
            metric_panel("Mesures LSTM", st.session_state.drone_lstm, current_target)
    elif mode == "IA LSTM" and lstm_ready:
        metric_panel("Mesures LSTM", st.session_state.drone_lstm, current_target)
    else:
        metric_panel("Mesures PID", st.session_state.drone_pid, current_target)

    st.subheader("Angles en temps reel")
    st.plotly_chart(
        make_angles_chart(st.session_state.drone_pid, st.session_state.drone_lstm, compare),
        use_container_width=True,
    )

    st.subheader("Replay trajectoire")
    replay_drone = st.session_state.drone_lstm if mode == "IA LSTM" and lstm_ready else st.session_state.drone_pid
    max_idx = max(1, len(replay_drone.trajectory) - 1)
    previous_replay_idx = int(st.session_state.get("replay_idx", max_idx))
    replay_value = int(np.clip(previous_replay_idx, 0, max_idx))
    replay_idx = st.slider("Instant de rejeu", 0, max_idx, replay_value, key="replay_idx")
    replay_target = reference_target(
        final_target,
        replay_idx * DT,
        trajectory_mode,
        trajectory_duration,
        st.session_state.reference_start,
    )
    st.plotly_chart(
        make_scene(
            replay_drone,
            replay_target,
            f"Replay t = {replay_idx * DT:.2f} s",
            upto=replay_idx + 1,
            final_target=final_target,
            reference_curve=ref_curve,
        ),
        use_container_width=True,
    )

    if st.session_state.running:
        time.sleep(0.045)
        st.rerun()


if __name__ == "__main__":
    main()
