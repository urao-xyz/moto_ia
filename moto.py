import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.utils.tensorboard import SummaryWriter
from ipywidgets import interact, FloatSlider
import logging
import numpy as np

# ==================================================
# Configuration des paramètres
# ==================================================

# Paramètres physiques de la moto
MASS_MEAN = 200.0  # Masse moyenne en kg
MASS_STD = 10.0     # Écart-type de la masse en kg
DRAG_COEFFICIENT = 0.5  # Coefficient de traînée aérodynamique
MAX_THRUST = 500.0  # Force maximale de propulsion en Newtons
MAX_BRAKE = 300.0   # Force maximale de freinage en Newtons

# Conditions initiales
INITIAL_SPEED = 0.0  # Vitesse initiale en m/s

# Paramètres de simulation
DT = 0.1            # Pas de temps en secondes
SIMULATION_TIME = 30.0  # Durée totale de la simulation en secondes

# Paramètres d'optimisation
TARGET_SPEED = 70.0  # Vitesse cible en m/s
LEARNING_RATE = 0.01  # Taux d'apprentissage
EPOCHS = 100         # Nombre d'epochs pour l'optimisation

# Paramètres d'entrée
THROTTLE_INPUT = 0.8  # Valeur de l'accélérateur (0 à 1)
BRAKE_INPUT = 0.0     # Valeur du frein (0 à 1)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Configuration du dispositif (CPU ou GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================================================
# Définition de la classe Motorcycle
# ==================================================

class Motorcycle(nn.Module):
    def __init__(self, mass_mean=MASS_MEAN, mass_std=MASS_STD, drag_coefficient=DRAG_COEFFICIENT, max_thrust=MAX_THRUST, max_brake=MAX_BRAKE, device='cpu'):
        super(Motorcycle, self).__init__()
        self.device = device

        # Paramètres du modèle (avec gradients activés pour l'optimisation)
        self.mass_mean = nn.Parameter(torch.tensor(mass_mean, dtype=torch.float32, device=device))
        self.mass_std = nn.Parameter(torch.tensor(mass_std, dtype=torch.float32, device=device))
        self.drag_coefficient = nn.Parameter(torch.tensor(drag_coefficient, dtype=torch.float32, device=device))
        self.max_thrust = nn.Parameter(torch.tensor(max_thrust, dtype=torch.float32, device=device))
        self.max_brake = nn.Parameter(torch.tensor(max_brake, dtype=torch.float32, device=device))

        # Conditions initiales
        self.speed = torch.tensor(INITIAL_SPEED, dtype=torch.float32, device=device)
        self.time = torch.tensor(0.0, dtype=torch.float32, device=device)

        # Historique
        self.speed_history = []
        self.time_history = []

    def forward(self, throttle, brake, dt, generator=None):
        # Génération aléatoire de la masse à chaque pas de temps
        if generator is not None:
            noise = torch.randn((), generator=generator, device=self.device)
        else:
            noise = torch.randn((), device=self.device)
        mass = self.mass_mean + self.mass_std * noise

        # Calcul des forces
        thrust = throttle * self.max_thrust
        braking = -brake * self.max_brake
        drag = -self.drag_coefficient * self.speed ** 2

        # Mise à jour de la vitesse
        net_force = thrust + braking + drag
        acceleration = net_force / mass
        self.speed = self.speed + acceleration * dt  # Assure la compatibilité des formes

        # Mise à jour du temps et de l'historique
        self.time = self.time + dt
        self.speed_history.append(self.speed.item())
        self.time_history.append(self.time.item())

        return self.speed

    def reset(self):
        """Réinitialise la moto à ses conditions initiales."""
        self.speed = torch.tensor(INITIAL_SPEED, dtype=torch.float32, device=self.device)
        self.time = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.speed_history = []
        self.time_history = []

# ==================================================
# Fonctions de simulation, visualisation et optimisation
# ==================================================

def simulate(motorcycle, throttle_input, brake_input, dt, simulation_time, writer=None):
    time_steps = torch.arange(0, simulation_time, dt, device=device)
    for t in time_steps:
        # Variation aléatoire de l'accélérateur et du frein
        throttle = throttle_input + torch.normal(0.0, 0.1, size=(), device=device).clamp(0, 1)  # Scalaire
        brake = brake_input + torch.normal(0.0, 0.1, size=(), device=device).clamp(0, 1)  # Scalaire

        # Ajout de perturbations aléatoires
        add_perturbation(motorcycle, dt)

        # Mise à jour de la moto
        speed = motorcycle(throttle, brake, dt)

        # Logging et TensorBoard
        if writer:
            writer.add_scalar('Speed', speed, t.item())
        logging.info(f"Time: {t.item():.2f}s, Speed: {speed.item():.2f}m/s")

def add_perturbation(motorcycle, dt, generator=None):
    """Ajoute une perturbation aléatoire (par exemple, du vent ou des bosses)."""
    if generator is None:
        perturbation_force = torch.normal(0.0, 10.0, size=(), device=device)
    else:
        perturbation_force = torch.normal(0.0, 10.0, size=(), device=device, generator=generator)
    motorcycle.speed += (perturbation_force / motorcycle.mass_mean) * dt

def calculate_energy_consumption(motorcycle, throttle, brake, dt):
    """Calcule la consommation d'énergie en fonction de la force de propulsion et du temps."""
    energy_consumption = (throttle * motorcycle.max_thrust * dt).item()  # en Joules
    return energy_consumption

def plot_speed_history(motorcycle):
    """Affiche l'évolution de la vitesse en fonction du temps."""
    plt.figure(figsize=(10, 6))
    plt.plot(motorcycle.time_history, motorcycle.speed_history, label="Speed (m/s)", color="blue")
    plt.title("Motorcycle Speed Over Time (PyTorch Simulation)")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_additional_graphs(motorcycle):
    """Affiche des graphiques supplémentaires (accélération et force nette)."""
    # Convertir l'historique de vitesse en tenseur PyTorch
    speed_history_tensor = torch.tensor(motorcycle.speed_history, device=device)

    # Calculer l'accélération et la force nette
    acceleration_history = torch.diff(speed_history_tensor) / DT
    net_force_history = motorcycle.mass_mean * acceleration_history

    # Détacher les tenseurs et les convertir en tableaux NumPy
    time_history_np = torch.tensor(motorcycle.time_history[:-1], device=device).detach().cpu().numpy()
    acceleration_history_np = acceleration_history.detach().cpu().numpy()
    net_force_history_np = net_force_history.detach().cpu().numpy()

    # Graphique de l'accélération
    plt.figure(figsize=(10, 6))
    plt.plot(time_history_np, acceleration_history_np, label="Accélération (m/s²)", color="red")
    plt.title("Accélération de la moto en fonction du temps")
    plt.xlabel("Time (s)")
    plt.ylabel("Accélération (m/s²)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Graphique de la force nette
    plt.figure(figsize=(10, 6))
    plt.plot(time_history_np, net_force_history_np, label="Force nette (N)", color="green")
    plt.title("Force nette appliquée à la moto en fonction du temps")
    plt.xlabel("Time (s)")
    plt.ylabel("Force nette (N)")
    plt.grid(True)
    plt.legend()
    plt.show()

def optimize_motorcycle(motorcycle, target_speed, throttle_input, brake_input, dt, simulation_time, lr=LEARNING_RATE, epochs=EPOCHS):
    """Optimise les paramètres de la moto pour atteindre la vitesse cible et retourne les données des epochs."""
    writer = SummaryWriter()
    optimizer = optim.Adam(motorcycle.parameters(), lr=lr)
    
    best_loss = float('inf')
    first_epoch_speed = []
    first_epoch_time = []
    best_epoch_speed = []
    best_epoch_time = []
    
    for epoch in range(epochs):
        motorcycle.reset()
        simulate(motorcycle, throttle_input, brake_input, dt, simulation_time)
        
        # Capture des données de cet epoch
        current_speed = motorcycle.speed_history.copy()
        current_time = motorcycle.time_history.copy()
        
        # Calcul de la perte
        final_speed = motorcycle.speed
        loss = (final_speed - target_speed) ** 2
        
        # Mise à jour des données du premier et meilleur epoch
        if epoch == 0:
            first_epoch_speed = current_speed
            first_epoch_time = current_time
            best_loss = loss.item()
            best_epoch_speed = current_speed
            best_epoch_time = current_time
        else:
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch_speed = current_speed
                best_epoch_time = current_time
        
        # Optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Enregistrement des paramètres dans TensorBoard
        writer.add_scalar('Loss', loss.item(), epoch)
        writer.add_scalar('Mass Mean', motorcycle.mass_mean.item(), epoch)
        writer.add_scalar('Drag Coefficient', motorcycle.drag_coefficient.item(), epoch)
        
        logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Final Speed: {final_speed.item():.2f}m/s")
    
    writer.close()
    return first_epoch_speed, first_epoch_time, best_epoch_speed, best_epoch_time

def plot_epoch_comparison(first_speed, first_time, best_speed, best_time):
    """Affiche la comparaison entre le premier epoch et le meilleur epoch."""
    plt.figure(figsize=(10, 6))
    plt.plot(first_time, first_speed, label="Premier Epoch", color="blue", alpha=0.7)
    plt.plot(best_time, best_speed, label="Meilleur Epoch", color="green", alpha=0.7)
    plt.axhline(y=TARGET_SPEED, color='red', linestyle='--', label="Vitesse Cible")
    plt.title("Comparaison du Premier Epoch et du Meilleur Epoch")
    plt.xlabel("Temps (s)")
    plt.ylabel("Vitesse (m/s)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_acceleration_comparison(first_speed, first_time, best_speed, best_time):
    """Affiche la comparaison des accélérations entre le premier epoch et le meilleur epoch."""
    # Calcul des accélérations à partir des vitesses
    first_acceleration = np.diff(first_speed) / DT
    best_acceleration = np.diff(best_speed) / DT
    
    # Création de la figure
    plt.figure(figsize=(12, 6))
    
    # Sous-graphique pour les accélérations
    plt.subplot(1, 2, 1)
    plt.plot(first_time[:-1], first_acceleration, label="Premier Epoch", color="blue", alpha=0.7)
    plt.plot(best_time[:-1], best_acceleration, label="Meilleur Epoch", color="green", alpha=0.7)
    plt.title("Comparaison des Accélérations")
    plt.xlabel("Temps (s)")
    plt.ylabel("Accélération (m/s²)")
    plt.grid(True)
    plt.legend()

    # Sous-graphique pour les vitesses
    plt.subplot(1, 2, 2)
    plt.plot(first_time, first_speed, label="Premier Epoch", color="blue", alpha=0.7)
    plt.plot(best_time, best_speed, label="Meilleur Epoch", color="green", alpha=0.7)
    plt.axhline(y=TARGET_SPEED, color='red', linestyle='--', label="Cible")
    plt.title("Comparaison des Vitesses")
    plt.xlabel("Temps (s)")
    plt.ylabel("Vitesse (m/s)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def interactive_simulation(motorcycle, dt, simulation_time):
    """Simulation interactive avec contrôle de l'accélérateur et du frein."""
    def update(throttle, brake):
        motorcycle.reset()
        time_steps = torch.arange(0, simulation_time, dt, device=device)
        for t in time_steps:
            speed = motorcycle(throttle, brake, dt)
            logging.info(f"Time: {t.item():.2f}s, Speed: {speed.item():.2f}m/s")

    interact(update,
             throttle=FloatSlider(min=0.0, max=1.0, step=0.1, value=0.8),
             brake=FloatSlider(min=0.0, max=1.0, step=0.1, value=0.0))

# ==================================================
# Point d'entrée
# ==================================================

if __name__ == "__main__":
    # Initialisation
    motorcycle = Motorcycle(device=device)
    throttle_input = torch.tensor(THROTTLE_INPUT, dtype=torch.float32, device=device)
    brake_input = torch.tensor(BRAKE_INPUT, dtype=torch.float32, device=device)
    dt = torch.tensor(DT, dtype=torch.float32, device=device)
    simulation_time = torch.tensor(SIMULATION_TIME, dtype=torch.float32, device=device)

    # Simulation interactive
    interactive_simulation(motorcycle, dt, simulation_time)

    # Optimisation et récupération des données
    first_speed, first_time, best_speed, best_time = optimize_motorcycle(
        motorcycle, 
        target_speed=TARGET_SPEED,
        throttle_input=throttle_input,
        brake_input=brake_input,
        dt=dt,
        simulation_time=simulation_time
    )

    # Visualisation de la comparaison
    #plot_epoch_comparison(first_speed, first_time, best_speed, best_time)
    plot_acceleration_comparison(first_speed, first_time, best_speed, best_time)

    # Visualisation des graphiques supplémentaires
    #plot_additional_graphs(motorcycle)

    '''
    Pour lancer la visualisation sur TensorBoard :
    tensorboard --logdir=runs
    '''