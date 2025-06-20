# Simulation et Optimisation d'une Moto avec PyTorch

Ce projet simule et optimise le comportement d'une moto en utilisant PyTorch. Il inclut des fonctionnalités pour simuler la vitesse de la moto, optimiser les paramètres pour atteindre une vitesse cible, et visualiser les résultats.

## Introduction

Ce projet utilise PyTorch pour simuler la dynamique d'une moto en fonction de divers paramètres physiques tels que la masse, la traînée aérodynamique, la force de propulsion et le freinage. Il permet également d'optimiser ces paramètres pour atteindre une vitesse cible spécifique.

## Installation

Pour exécuter ce projet, vous devez avoir Python 3.x installé ainsi que les bibliothèques suivantes :

```bash
pip install torch matplotlib ipywidgets numpy tensorboard gym
```

Pour lancer la simulation simplement :
```bash
python moto.py
```

## Entra\xeenement par renforcement (RL)
Un environnement Gym est disponible via `MotorcycleEnv`. Pour entraîner un agent
actor-critic et sauvegarder le modèle :

```bash
python train_rl.py --episodes 100 --energy-weight 0.1 --save agent.pth
```

Les modèles sauvegardés peuvent être rechargés avec `--load chemin.pth`.
