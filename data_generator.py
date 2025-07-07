"""
This module generates realistic interaction data with sequential patterns
test
"""

import pandas as pd
import numpy as np
import datetime
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class BankingProduct:
    """Banking product configuration"""
    id: int
    name: str
    category: str
    avg_amount: float
    revenue: float


@dataclass
class CustomerSegment:
    """Customer segment configuration"""
    name: str
    age_range: Tuple[int, int]
    income_range: Tuple[int, int]
    preferred_products: List[int]
    frequency: float


class BankingDataGenerator:
    """
    Generates realistic banking interaction sequences for recommendation models.
    
    Creates customer profiles with temporal interaction patterns based on
    business rules and behavioral segments.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        random.seed(seed)
        
        self.products = self._define_products()
        self.segments = self._define_segments()
        self.channels = ["MOBILE", "ONLINE", "AGENCE", "ATM", "PHONE"]
    
    def _define_products(self) -> Dict[int, BankingProduct]:
        """Define banking products catalog"""
        return {
            1: BankingProduct(1, "Dépôt", "BANKING", 500, 5),
            2: BankingProduct(2, "Retrait", "BANKING", 200, 2),
            3: BankingProduct(3, "Virement", "BANKING", 800, 3),
            4: BankingProduct(4, "Carte Bleue", "PAYMENT", 50, 15),
            5: BankingProduct(5, "Carte Gold", "PAYMENT", 120, 45),
            6: BankingProduct(6, "Prêt Personnel", "LOAN", 15000, 350),
            7: BankingProduct(7, "Prêt Immobilier", "LOAN", 200000, 1200),
            8: BankingProduct(8, "Assurance Auto", "INSURANCE", 800, 120),
            9: BankingProduct(9, "Assurance Vie", "INSURANCE", 5000, 400),
            10: BankingProduct(10, "Livret A", "SAVING", 2000, 25),
            11: BankingProduct(11, "PEL", "SAVING", 8000, 60),
            12: BankingProduct(12, "Compte Pro", "BUSINESS", 1000, 80),
        }
    
    def _define_segments(self) -> Dict[str, CustomerSegment]:
        """Define customer segments with behavioral patterns"""
        return {
            "YOUNG": CustomerSegment("YOUNG", (18, 30), (25000, 45000), [1, 2, 3, 4, 10], 0.35),
            "FAMILY": CustomerSegment("FAMILY", (30, 50), (40000, 80000), [1, 2, 3, 4, 6, 7, 8, 11], 0.30),
            "SENIOR": CustomerSegment("SENIOR", (50, 70), (35000, 70000), [1, 2, 3, 9, 10, 11], 0.25),
            "PREMIUM": CustomerSegment("PREMIUM", (35, 65), (80000, 200000), [1, 2, 3, 5, 6, 7, 9, 12], 0.10),
        }
    
    def _get_sequential_products(self, last_product_id: int, available_products: List[int]) -> List[int]:
        """
        Apply sequential business logic to product selection.
        
        Args:
            last_product_id: Previous product used by customer
            available_products: List of products available for this segment
            
        Returns:
            Updated list of products with sequential preferences
        """
        if last_product_id in [1, 2]:  # After deposit/withdrawal -> cards or savings
            return [4, 10, 11] + available_products
        elif last_product_id == 4:  # After basic card -> premium card
            return [5] + available_products
        elif last_product_id in [6, 7]:  # After loans -> insurance
            return [8, 9] + available_products
        return available_products
    
    def _select_channel(self, segment_name: str) -> str:
        """Select interaction channel based on customer segment"""
        if segment_name == "YOUNG":
            return np.random.choice(["MOBILE", "ONLINE"], p=[0.7, 0.3])
        elif segment_name == "SENIOR":
            return np.random.choice(["AGENCE", "PHONE", "ATM"], p=[0.5, 0.3, 0.2])
        else:
            return np.random.choice(self.channels)
    
    def generate_dataset(self, 
                        n_customers: int = 3000, 
                        avg_interactions_per_customer: int = 20) -> pd.DataFrame:
        """
        Generate complete banking interaction dataset.
        
        Args:
            n_customers: Number of unique customers
            avg_interactions_per_customer: Average sequence length per customer
            
        Returns:
            DataFrame with banking interactions
        """
        print(f"Generating dataset with {n_customers} customers...")
        
        data = []
        
        # Generate segments probabilities
        segment_names = list(self.segments.keys())
        segment_probs = [self.segments[s].frequency for s in segment_names]
        
        for customer_id in range(n_customers):
            if customer_id % 500 == 0:
                print(f"Processing customer {customer_id}/{n_customers}")
            
            # Assign customer segment
            segment_name = np.random.choice(segment_names, p=segment_probs)
            segment = self.segments[segment_name]
            
            # Generate customer characteristics
            age = np.random.randint(segment.age_range[0], segment.age_range[1])
            income = np.random.randint(segment.income_range[0], segment.income_range[1])
            
            # Generate interaction sequence
            n_interactions = max(5, np.random.poisson(avg_interactions_per_customer))
            
            # Session management
            session_changes = sorted(
                np.random.choice(n_interactions, size=max(1, n_interactions // 8), replace=False)
            )
            
            base_time = datetime.datetime.now() - datetime.timedelta(days=365)
            session_id = 0
            
            for i in range(n_interactions):
                # Update session if needed
                if i in session_changes:
                    session_id += 1
                
                # Select product with sequential logic
                available_products = segment.preferred_products.copy()
                if i > 0:
                    last_product = data[-1]["item_id"]
                    available_products = self._get_sequential_products(last_product, available_products)
                
                item_id = np.random.choice(available_products)
                product = self.products[item_id]
                
                # Generate transaction amount with variance
                amount = max(10, np.random.normal(product.avg_amount, product.avg_amount * 0.3))
                
                # Generate timestamp
                time_offset = datetime.timedelta(
                    days=np.random.randint(0, 30),
                    hours=np.random.randint(8, 20),
                    minutes=np.random.randint(0, 60),
                )
                timestamp = base_time + datetime.timedelta(days=i * 7) + time_offset
                
                # Select channel
                channel = self._select_channel(segment_name)
                
                data.append({
                    "customer_id": customer_id,
                    "item_id": item_id,
                    "product_name": product.name,
                    "product_category": product.category,
                    "session_id": customer_id * 100 + session_id,
                    "timestamp": timestamp,
                    "amount": round(amount, 2),
                    "channel": channel,
                    "customer_age": age,
                    "customer_income": income,
                    "customer_segment": segment_name,
                    "position_in_sequence": i + 1,
                    "revenue_potential": product.revenue,
                })
        
        df = pd.DataFrame(data)
        df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
        
        print(f"Dataset generated: {len(df)} interactions for {n_customers} customers")
        return df
    
    def get_dataset_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate dataset statistics.
        
        Args:
            df: Generated dataset
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "total_interactions": len(df),
            "unique_customers": df["customer_id"].nunique(),
            "unique_products": df["item_id"].nunique(),
            "avg_sequence_length": df.groupby("customer_id").size().mean(),
            "date_range": (df["timestamp"].min(), df["timestamp"].max()),
            "segments_distribution": df["customer_segment"].value_counts().to_dict(),
            "products_distribution": df["product_name"].value_counts().to_dict(),
            "channels_distribution": df["channel"].value_counts().to_dict(),
            "total_revenue_potential": df["revenue_potential"].sum(),
        }
        return stats


if __name__ == "__main__":
    # Example usage
    generator = BankingDataGenerator(seed=42)
    
    # Generate dataset
    dataset = generator.generate_dataset(n_customers=1000, avg_interactions_per_customer=15)
    
    # Show statistics
    stats = generator.get_dataset_stats(dataset)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Save dataset
    dataset.to_csv("banking_interactions.csv", index=False)
    print("\nDataset saved to banking_interactions.csv")