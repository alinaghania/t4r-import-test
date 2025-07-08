"""
PACKAGE: SYSTÈME DE RECOMMANDATION TRANSFORMER
=============================================

Ce package fournit un système de recommandation complet basé sur
l'architecture Transformer pour la prédiction de produits.

Modules:
- data_generator: Génération de données synthétiques
- data_preprocessor: Preprocessing des séquences
- transformer_model: Architecture du modèle Transformer
- model_trainer: Entraînement du modèle
- recommendation_engine: Moteur d'inférence et recommandations

"""

# Import des classes principales
from .data_generator import SyntheticDataGenerator
from .data_preprocessor import SequenceDataPreprocessor
from .transformer_model import TransformerRecommendationModel
from .model_trainer import ModelTrainer
from .recommendation_engine import RecommendationEngine

# Version du package
__version__ = "1.0.0"

# Classes exportées
__all__ = [
    # Génération de données
    "SyntheticDataGenerator",
    # Preprocessing
    "SequenceDataPreprocessor",
    # Modèle
    "TransformerRecommendationModel",
    # Entraînement
    "ModelTrainer",
    # Inférence
    "RecommendationEngine",
]

# Métadonnées
__author__ = "Assistant AI & Alina Ghani"
__email__ = "alina.ghani@accenture.com"
__description__ = "Système de recommandation Transformer pour produits"
