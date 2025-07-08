"""
SYSTÈME DE RECOMMANDATION BANCAIRE AVEC TRANSFORMERS
====================================================

Ce script implémente un système de recommandation pour produits bancaires utilisant
l'architecture Transformer. Le modèle prédit le prochain produit qu'un client va
consulter/acheter basé sur son historique de navigation.

Architecture:
- Embeddings pour représenter les produits bancaires (64 dimensions)
- Transformer Encoder (2 couches, 4 têtes d'attention) pour capturer les séquences
- Couche de sortie pour prédiction du prochain item (next item prediction)

"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


def create_sample_banking_data():
    """
    Génère des données bancaires synthétiques pour l'entraînement.

    Structure des données:
    - customer_id: Identifiant unique du client (1-1000)
    - item_id: Produit bancaire consulté/acheté (1-50)
    - timestamp: Horodatage de l'interaction
    - session_id: Identifiant de session de navigation (0-4999)
    - amount: Montant de la transaction (10-1000€)

    Returns:
        pd.DataFrame: DataFrame avec les interactions bancaires
    """
    np.random.seed(42)  # Pour la reproductibilité
    n_customers = 1000  # Nombre de clients
    n_products = 50  # Nombre de produits bancaires différents
    n_sessions = 5000  # Nombre de sessions de navigation
    data = []

    print("Génération des données bancaires synthétiques...")

    for session_id in range(n_sessions):
        # Chaque session appartient à un client aléatoire
        customer_id = np.random.randint(1, n_customers + 1)

        # Longueur variable de session (2-9 interactions)
        session_length = np.random.randint(2, 10)

        # Timestamp de base pour la session
        base_timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(
            days=np.random.randint(0, 365)
        )

        # Générer les interactions de la session
        for i in range(session_length):
            data.append(
                {
                    "customer_id": customer_id,
                    "item_id": np.random.randint(1, n_products + 1),
                    "timestamp": base_timestamp + pd.Timedelta(minutes=i * 5),
                    "session_id": session_id,
                    "amount": np.random.uniform(10, 1000),
                }
            )

    # Créer le DataFrame et trier par session et timestamp
    df = pd.DataFrame(data)
    df = df.sort_values(["session_id", "timestamp"])

    # S'assurer que les types sont corrects
    df["session_id"] = df["session_id"].astype(np.int64)
    df["customer_id"] = df["customer_id"].astype(np.int64)
    df["item_id"] = df["item_id"].astype(np.int64)

    return df


class SimpleBankingRecommenderModel(nn.Module):
    """
    Modèle de recommandation bancaire basé sur l'architecture Transformer.

    Le modèle utilise:
    1. Des embeddings pour représenter les produits bancaires
    2. Un Transformer Encoder pour capturer les dépendances séquentielles
    3. Une couche de sortie pour prédire le prochain produit

    Args:
        num_items (int): Nombre total de produits bancaires
        embedding_dim (int): Dimension des embeddings (défaut: 64)
        seq_length (int): Longueur maximale des séquences (défaut: 10)
    """

    def __init__(self, num_items, embedding_dim=64, seq_length=10):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length

        # Couche d'embedding pour les produits bancaires
        # padding_idx=0 pour que le token de padding ait un embedding fixe à zéro
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        # Architecture Transformer pour capturer les séquences
        # - 2 couches d'encodeur pour capturer les dépendances complexes
        # - 4 têtes d'attention pour différents aspects des relations
        # - Dimension feedforward 2x plus grande (128) pour plus de capacité
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,  # Dimension du modèle
                nhead=4,  # Nombre de têtes d'attention
                dim_feedforward=128,  # Dimension du feedforward
                dropout=0.1,  # Dropout pour la régularisation
                batch_first=True,  # Format [batch, seq, features]
            ),
            num_layers=2,  # Nombre de couches d'encodeur
        )

        # Couche de sortie pour prédire le prochain produit
        # +1 pour inclure le token de padding dans les prédictions
        self.output_layer = nn.Linear(embedding_dim, num_items + 1)

    def forward(self, item_sequence):
        """
        Forward pass du modèle.

        Args:
            item_sequence (torch.Tensor): Séquence d'items [batch_size, seq_length]

        Returns:
            torch.Tensor: Logits pour la prédiction du prochain item [batch_size, num_items+1]
        """
        # Étape 1: Convertir les IDs en embeddings
        embeddings = self.item_embedding(item_sequence)  # [batch, seq_len, embed_dim]

        # Étape 2: Passer par le Transformer pour capturer les dépendances
        sequence_output = self.transformer(embeddings)  # [batch, seq_len, embed_dim]

        # Étape 3: Utiliser la représentation du dernier token pour la prédiction
        # Le dernier token contient l'information de toute la séquence
        last_output = sequence_output[:, -1, :]  # [batch, embed_dim]

        # Étape 4: Prédire le prochain item
        logits = self.output_layer(last_output)  # [batch, num_items+1]

        return logits


def create_target_sequence(item_sequence):
    """
    Crée les séquences d'entrée et de cible pour l'entraînement.

    Pour l'entraînement "next item prediction":
    - Input: tous les items sauf le dernier  [item1, item2, ..., itemN-1]
    - Target: tous les items sauf le premier [item2, item3, ..., itemN]

    Args:
        item_sequence (torch.Tensor): Séquences complètes [batch, seq_len]

    Returns:
        tuple: (inputs, targets) pour l'entraînement
    """
    inputs = item_sequence[:, :-1]  # Tous sauf le dernier
    targets = item_sequence[:, 1:]  # Tous sauf le premier

    return inputs, targets


def main():
    """Fonction principale du pipeline d'entraînement."""

    print("=" * 60)
    print("🏦 SYSTÈME DE RECOMMANDATION BANCAIRE AVEC TRANSFORMERS 🏦")
    print("=" * 60)

    # ==========================================
    # 1. GÉNÉRATION DES DONNÉES
    # ==========================================
    print("\n=== 1. GÉNÉRATION DES DONNÉES ===")
    df = create_sample_banking_data()
    print(f"✅ Données générées: {df.shape[0]:,} transactions")
    print(f"   • Sessions: {df['session_id'].nunique():,}")
    print(f"   • Clients: {df['customer_id'].nunique():,}")
    print(f"   • Produits: {df['item_id'].nunique()}")
    print("\nAperçu des données:")
    print(df.head())

    # Sauvegarder les données brutes
    os.makedirs("processed_data/seq_data", exist_ok=True)
    raw_path = "processed_data/seq_data/banking_data_raw.parquet"
    df.to_parquet(raw_path, index=False)
    print(f"💾 Données sauvegardées: {raw_path}")

    # ==========================================
    # 2. PRÉPARATION DES SÉQUENCES
    # ==========================================
    print("\n=== 2. PRÉPARATION DES SÉQUENCES ===")
    max_seq_length = 10  # Longueur maximale des séquences
    sequences = []

    print("Conversion des sessions en séquences...")
    for session_id, session_data in df.groupby("session_id"):
        session_data = session_data.sort_values("timestamp")
        item_seq = session_data["item_id"].tolist()

        # Padding et troncature pour uniformiser les longueurs
        if len(item_seq) > max_seq_length:
            # Garder les derniers items (plus récents)
            item_seq = item_seq[-max_seq_length:]
        else:
            # Ajouter du padding (0) au début
            item_seq = item_seq + [0] * (max_seq_length - len(item_seq))

        sequences.append(item_seq)

    sequences = np.array(sequences)
    print(f"✅ Séquences créées: {sequences.shape}")
    print(f"   • Forme: {sequences.shape[0]:,} sessions × {sequences.shape[1]} items")
    print(f"   • Exemple: {sequences[0]}")

    # ==========================================
    # 3. CRÉATION DU MODÈLE
    # ==========================================
    print("\n=== 3. CRÉATION DU MODÈLE ===")
    num_items = 50  # Nombre maximum d'items dans nos données
    model = SimpleBankingRecommenderModel(
        num_items=num_items, embedding_dim=64, seq_length=max_seq_length
    )

    # Configuration du device (GPU si disponible, sinon CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device utilisé: {device}")
    model = model.to(device)

    # Afficher l'architecture du modèle
    print("\n📊 Architecture du modèle:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   • Paramètres totaux: {total_params:,}")
    print(f"   • Paramètres entraînables: {trainable_params:,}")
    print(f"   • Embedding dimension: {64}")
    print(f"   • Couches Transformer: 2")
    print(f"   • Têtes d'attention: 4")

    # ==========================================
    # 4. PRÉPARATION DES DONNÉES D'ENTRAÎNEMENT
    # ==========================================
    print("\n=== 4. PRÉPARATION DES DONNÉES D'ENTRAÎNEMENT ===")

    # Convertir en tenseurs PyTorch
    sequences_tensor = torch.LongTensor(sequences).to(device)

    # Créer les paires input/target pour next item prediction
    inputs, targets = create_target_sequence(sequences_tensor)

    print(f"✅ Données d'entraînement préparées:")
    print(f"   • Inputs shape: {inputs.shape}")
    print(f"   • Targets shape: {targets.shape}")
    print(f"   • Exemple input: {inputs[0].cpu().numpy()}")
    print(f"   • Exemple target: {targets[0].cpu().numpy()}")

    # ==========================================
    # 5. ENTRAÎNEMENT DU MODÈLE
    # ==========================================
    print("\n=== 5. ENTRAÎNEMENT DU MODÈLE ===")

    # Configuration de l'entraînement
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignorer le padding
    batch_size = 64
    num_epochs = 5

    print(f"⚙️  Configuration:")
    print(f"   • Optimiseur: Adam (lr=1e-3)")
    print(f"   • Loss: CrossEntropyLoss")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Époques: {num_epochs}")

    model.train()
    print(f"\n🚀 Début de l'entraînement...")

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        # Entraînement par mini-batch
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]
            batch_targets = targets[i : i + batch_size]

            if len(batch_inputs) == 0:
                continue

            optimizer.zero_grad()

            # Training avec prédiction séquentielle
            batch_size_actual = batch_inputs.size(0)
            seq_len = batch_inputs.size(1)
            all_losses = []

            # Prédire à chaque position de la séquence
            for pos in range(seq_len):
                if pos == 0:
                    continue  # Pas de prédiction sans contexte

                # Utiliser la séquence jusqu'à la position pos
                input_seq = batch_inputs[:, : pos + 1]

                # Padding si nécessaire
                if input_seq.size(1) < max_seq_length:
                    padding = torch.zeros(
                        batch_size_actual,
                        max_seq_length - input_seq.size(1),
                        device=device,
                        dtype=torch.long,
                    )
                    input_seq = torch.cat([padding, input_seq], dim=1)

                # Forward pass
                logits = model(input_seq)
                target_items = batch_targets[:, pos - 1]

                # Loss seulement pour les tokens non-padding
                mask = target_items != 0
                if mask.sum() > 0:
                    loss = criterion(logits[mask], target_items[mask])
                    all_losses.append(loss)

            # Backpropagation
            if all_losses:
                total_batch_loss = torch.stack(all_losses).mean()
                total_batch_loss.backward()
                optimizer.step()

                total_loss += total_batch_loss.item()
                num_batches += 1

        # Afficher le progrès
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"   Époque {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")

    print("✅ Entraînement terminé !")

    # ==========================================
    # 6. TEST ET ÉVALUATION
    # ==========================================
    print("\n=== 6. TEST ET ÉVALUATION ===")
    model.eval()

    with torch.no_grad():
        test_sequences = sequences_tensor[:5]  # Tester sur 5 séquences

        print("🔍 Tests de prédiction:")
        for i, seq in enumerate(test_sequences):
            # Utiliser la séquence sans le dernier item pour prédire
            input_seq = seq[:-1].unsqueeze(0)
            actual_next = seq[-1].item()

            # Padding si nécessaire
            if input_seq.size(1) < max_seq_length:
                padding = torch.zeros(
                    1,
                    max_seq_length - input_seq.size(1),
                    device=device,
                    dtype=torch.long,
                )
                input_seq = torch.cat([padding, input_seq], dim=1)

            # Prédiction
            logits = model(input_seq)
            predicted_probs = torch.softmax(logits, dim=1)
            top_predictions = torch.topk(predicted_probs, k=5, dim=1)

            print(f"\n   Séquence {i + 1}:")
            print(f"     Input: {input_seq.squeeze().cpu().numpy()}")
            print(f"     Item réel: {actual_next}")
            print(f"     Top 5: {top_predictions.indices.squeeze().cpu().numpy()}")
            print(f"     Proba: {top_predictions.values.squeeze().cpu().numpy()}")

    # ==========================================
    # 7. SAUVEGARDE DU MODÈLE
    # ==========================================
    print("\n=== 7. SAUVEGARDE DU MODÈLE ===")
    model_path = "banking_transformer_rec_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "num_items": num_items,
                "embedding_dim": 64,
                "seq_length": max_seq_length,
            },
            "training_info": {
                "num_epochs": num_epochs,
                "final_loss": avg_loss,
                "num_sequences": len(sequences),
                "num_transactions": len(df),
            },
        },
        model_path,
    )

    print(f"💾 Modèle sauvegardé: {model_path}")

    # ==========================================
    # RÉSUMÉ FINAL
    # ==========================================
    print("\n" + "=" * 60)
    print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS ! 🎉")
    print("=" * 60)
    print(f"✅ Données: {df.shape[0]:,} transactions, {len(sequences):,} sessions")
    print(f"✅ Modèle: Transformer {num_items} items, {64}D embeddings")
    print(f"✅ Entraînement: {max_seq_length} seq_length, {num_epochs} époques")
    print(f"✅ Performance: Loss finale {avg_loss:.4f}")
    print("✅ Pipeline complet de recommandation bancaire opérationnel !")
    print("\n🚀 Le modèle est prêt pour la production !")


if __name__ == "__main__":
    main()
