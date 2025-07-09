#!/usr/bin/env python3
"""
INVESTIGATION: FORMAT DES ARRAYS DANS T4REC_DF
==============================================

Script pour identifier pourquoi aucun array n'est détecté
et analyser le format réel des données dans les colonnes.
"""

import dataiku
import numpy as np
import pandas as pd
import json
import ast
import pickle
import re
from collections import Counter


def investigate_string_arrays():
    """Analyse approfondie du format des données dans les colonnes."""

    print("INVESTIGATION: FORMAT DES ARRAYS")
    print("=" * 50)

    try:
        # Charger la table
        dataset = dataiku.Dataset("t4_rec_df")
        df = dataset.get_dataframe()

        print(f"Table chargée: {len(df)} lignes x {len(df.columns)} colonnes")
        print(f"Taille mémoire: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        # 1. ANALYSE DU CONTENU DES PREMIÈRES COLONNES
        print(f"\n1. ANALYSE DU CONTENU (premières 10 colonnes)")
        print("-" * 50)

        sample_analysis = analyze_sample_content(df)

        # 2. DÉTECTION DE PATTERNS D'ARRAYS
        print(f"\n2. DÉTECTION DE PATTERNS D'ARRAYS")
        print("-" * 50)

        pattern_analysis = detect_array_patterns(df)

        # 3. TEST DE CONVERSIONS
        print(f"\n3. TEST DE CONVERSIONS")
        print("-" * 50)

        conversion_analysis = test_conversions(df, pattern_analysis)

        # 4. RECOMMANDATIONS SPÉCIFIQUES
        print(f"\n4. RECOMMANDATIONS ET CODE DE CONVERSION")
        print("-" * 50)

        recommendations = generate_conversion_recommendations(
            sample_analysis, pattern_analysis, conversion_analysis
        )

        # 5. RÉSUMÉ FINAL COMPLET
        print(f"\n5. RÉSUMÉ FINAL COMPLET")
        print("=" * 50)
        print_final_summary(df, pattern_analysis, conversion_analysis, recommendations)

        return {
            "sample_analysis": sample_analysis,
            "pattern_analysis": pattern_analysis,
            "conversion_analysis": conversion_analysis,
            "recommendations": recommendations,
        }

    except Exception as e:
        print(f"ERREUR lors du chargement: {str(e)}")
        import traceback

        print(f"Détails: {traceback.format_exc()}")
        return None


def analyze_sample_content(df):
    """Analyse le contenu des premières colonnes."""

    sample_results = {}

    # Analyser les 10 premières colonnes
    cols_to_analyze = df.columns[:10]

    for i, col in enumerate(cols_to_analyze):
        print(f"\nColonne {i + 1}: {col}")
        print(f"  Dtype: {df[col].dtype}")
        print(f"  Valeurs nulles: {df[col].isnull().sum()}")
        print(f"  Valeurs uniques: {df[col].nunique()}")

        try:
            # Récupérer quelques échantillons
            samples = df[col].dropna().head(3).tolist()

            col_analysis = {
                "dtype": str(df[col].dtype),
                "null_count": df[col].isnull().sum(),
                "unique_count": df[col].nunique(),
                "samples": samples,
                "sample_types": [type(s).__name__ for s in samples],
                "sample_lengths": [],
                "looks_like_array": False,
                "possible_format": "unknown",
            }

            # Analyser chaque échantillon
            for j, sample in enumerate(samples[:3]):
                print(f"  Échantillon {j + 1}:")
                print(f"    Type Python: {type(sample).__name__}")
                print(f"    Longueur: {len(str(sample)) if sample is not None else 0}")

                # Afficher le contenu avec échappement pour éviter les problèmes
                sample_str = str(sample)[:200] if sample is not None else "None"
                print(f"    Contenu: {repr(sample_str)}")

                if sample is not None:
                    col_analysis["sample_lengths"].append(len(str(sample)))

                    # Tests de format
                    format_detected = detect_format(sample)
                    if format_detected != "unknown":
                        col_analysis["possible_format"] = format_detected
                        col_analysis["looks_like_array"] = True
                        print(f"    FORMAT DETECTE: {format_detected}")

            sample_results[col] = col_analysis

        except Exception as e:
            print(f"  ERREUR analyse colonne {col}: {str(e)}")
            sample_results[col] = {"error": str(e)}

    return sample_results


def detect_format(sample):
    """Détecte le format possible d'un échantillon."""

    if sample is None:
        return "null"

    sample_str = str(sample)

    # Test 1: JSON Array
    if sample_str.strip().startswith("[") and sample_str.strip().endswith("]"):
        try:
            parsed = json.loads(sample_str)
            if isinstance(parsed, list):
                return "json_array"
        except:
            pass

    # Test 2: Python List String
    if sample_str.strip().startswith("[") and sample_str.strip().endswith("]"):
        try:
            parsed = ast.literal_eval(sample_str)
            if isinstance(parsed, list):
                return "python_list_string"
        except:
            pass

    # Test 3: Comma Separated Values
    if "," in sample_str and not sample_str.startswith("["):
        try:
            parts = [x.strip() for x in sample_str.split(",")]
            if len(parts) > 1 and all(
                x.replace(".", "").replace("-", "").isdigit() for x in parts
            ):
                return "csv_numbers"
        except:
            pass

    # Test 4: Space Separated Numbers
    if " " in sample_str and not sample_str.startswith("["):
        try:
            parts = sample_str.split()
            if len(parts) > 1 and all(
                x.replace(".", "").replace("-", "").isdigit() for x in parts
            ):
                return "space_separated_numbers"
        except:
            pass

    # Test 5: Single number
    if sample_str.replace(".", "").replace("-", "").isdigit():
        return "single_number"

    # Test 6: Pickled data
    if isinstance(sample, bytes) or sample_str.startswith("\\x"):
        return "binary_pickled"

    # Test 7: numpy array string representation
    if "array(" in sample_str:
        return "numpy_string"

    return "unknown"


def detect_array_patterns(df):
    """Détecte les patterns d'arrays dans toutes les colonnes."""

    pattern_stats = {
        "json_array": [],
        "python_list_string": [],
        "csv_numbers": [],
        "space_separated_numbers": [],
        "numpy_string": [],
        "binary_pickled": [],
        "single_number": [],
        "unknown": [],
    }

    print("Scan de toutes les 384 colonnes pour patterns...")

    # Analyser un échantillon de chaque colonne
    for i, col in enumerate(df.columns):
        try:
            # Prendre le premier échantillon non-null
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                sample = non_null_values.iloc[0]
                format_detected = detect_format(sample)
                pattern_stats[format_detected].append(col)

            # Progress
            if (i + 1) % 50 == 0:
                print(f"  Analysé {i + 1}/{len(df.columns)} colonnes...")

        except Exception as e:
            pattern_stats["unknown"].append(col)

    # Résumé des patterns
    print(f"\nPATTERNS DÉTECTÉS (RÉSUMÉ COMPLET):")
    total_columns = len(df.columns)
    for pattern, cols in pattern_stats.items():
        if cols:
            percentage = (len(cols) / total_columns) * 100
            print(f"  {pattern}: {len(cols)} colonnes ({percentage:.1f}%)")
            if len(cols) <= 5:
                print(f"    Colonnes: {cols}")
            else:
                print(f"    Premières colonnes: {cols[:3]}")
                print(f"    + {len(cols) - 3} autres colonnes...")

    return pattern_stats


def test_conversions(df, pattern_analysis):
    """Teste les conversions possibles pour chaque pattern."""

    conversion_results = {}

    print("Test des conversions pour chaque pattern détecté...")

    for pattern, columns in pattern_analysis.items():
        if not columns or pattern == "unknown":
            continue

        print(f"\nTest conversion pour pattern: {pattern}")
        print(f"Nombre de colonnes concernées: {len(columns)}")

        # Prendre la première colonne de ce pattern pour tester
        test_col = columns[0]
        print(f"Colonne de test: {test_col}")
        conversion_results[pattern] = test_pattern_conversion(df, test_col, pattern)

    return conversion_results


def test_pattern_conversion(df, col, pattern):
    """Teste la conversion d'une colonne selon son pattern."""

    result = {
        "column": col,
        "pattern": pattern,
        "success": False,
        "converted_samples": [],
        "conversion_method": "",
        "errors": [],
    }

    try:
        # Prendre quelques échantillons
        samples = df[col].dropna().head(5).tolist()
        print(f"  Test sur {len(samples)} échantillons...")

        for i, sample in enumerate(samples):
            try:
                if pattern == "json_array":
                    converted = json.loads(str(sample))
                    result["converted_samples"].append(converted)
                    result["conversion_method"] = "json.loads()"

                elif pattern == "python_list_string":
                    converted = ast.literal_eval(str(sample))
                    result["converted_samples"].append(converted)
                    result["conversion_method"] = "ast.literal_eval()"

                elif pattern == "csv_numbers":
                    converted = [float(x.strip()) for x in str(sample).split(",")]
                    result["converted_samples"].append(converted)
                    result["conversion_method"] = 'split(",") + float()'

                elif pattern == "space_separated_numbers":
                    converted = [float(x) for x in str(sample).split()]
                    result["converted_samples"].append(converted)
                    result["conversion_method"] = "split() + float()"

                elif pattern == "numpy_string":
                    # Plus complexe, nécessite numpy
                    converted = eval(str(sample))  # ATTENTION: dangereux en prod
                    result["converted_samples"].append(
                        converted.tolist()
                        if hasattr(converted, "tolist")
                        else converted
                    )
                    result["conversion_method"] = "eval() [DANGEREUX]"

                if i == 0:  # Afficher le premier exemple
                    print(f"    Échantillon original: {repr(str(sample)[:100])}")
                    print(f"    Après conversion: {converted}")

            except Exception as e:
                result["errors"].append(f"Échantillon {i}: {str(e)}")

        if result["converted_samples"]:
            result["success"] = True
            print(f"  SUCCÈS: Conversion réussie avec {result['conversion_method']}")
            print(
                f"  Échantillons convertis: {len(result['converted_samples'])}/{len(samples)}"
            )
        else:
            print(f"  ÉCHEC: Aucune conversion réussie")
            if result["errors"]:
                print(f"  Erreurs: {result['errors'][:3]}")

    except Exception as e:
        result["errors"].append(f"Erreur générale: {str(e)}")
        print(f"  ERREUR: {str(e)}")

    return result


def generate_conversion_recommendations(
    sample_analysis, pattern_analysis, conversion_analysis
):
    """Génère des recommandations de conversion."""

    print("GÉNÉRATION DES RECOMMANDATIONS")

    recommendations = []

    # Compter les succès de conversion
    successful_patterns = []
    for pattern, result in conversion_analysis.items():
        if result.get("success", False):
            successful_patterns.append((pattern, len(pattern_analysis[pattern])))

    if successful_patterns:
        print(f"\nCONVERSIONS POSSIBLES:")
        for pattern, count in successful_patterns:
            method = conversion_analysis[pattern]["conversion_method"]
            print(f"  {pattern}: {count} colonnes -> {method}")

            recommendations.append(
                {
                    "pattern": pattern,
                    "columns_count": count,
                    "method": method,
                    "priority": "HIGH" if count > 50 else "MEDIUM",
                }
            )

    # Pattern le plus fréquent
    if successful_patterns:
        best_pattern, best_count = max(successful_patterns, key=lambda x: x[1])
        print(f"\nRECOMMANDATION PRINCIPALE:")
        print(f"   Pattern dominant: {best_pattern} ({best_count} colonnes)")
        print(f"   Méthode: {conversion_analysis[best_pattern]['conversion_method']}")

        # Code de conversion recommandé
        print(f"\nCODE DE CONVERSION RECOMMANDÉ:")
        generate_conversion_code(best_pattern, conversion_analysis[best_pattern])

    else:
        print(f"\nAUCUNE CONVERSION AUTOMATIQUE POSSIBLE")
        print(f"   Investigation manuelle nécessaire")
        recommendations.append(
            {
                "pattern": "manual_investigation",
                "message": "Aucun pattern standard détecté",
                "priority": "CRITICAL",
            }
        )

    return recommendations


def generate_conversion_code(pattern, result):
    """Génère le code de conversion pour un pattern."""

    method = result["conversion_method"]

    if pattern == "json_array":
        print("""
import json
import pandas as pd

def convert_json_arrays(df):
    print("Conversion des arrays JSON en cours...")
    for i, col in enumerate(df.columns):
        try:
            df[col] = df[col].apply(lambda x: json.loads(str(x)) if pd.notna(x) else x)
            if (i + 1) % 50 == 0:
                print(f"  Converti {i + 1}/{len(df.columns)} colonnes...")
        except Exception as e:
            print(f"  Erreur colonne {col}: {str(e)}")
    print("Conversion terminée.")
    return df

# Utilisation:
# df_converted = convert_json_arrays(df.copy())
""")

    elif pattern == "python_list_string":
        print("""
import ast
import pandas as pd

def convert_python_lists(df):
    print("Conversion des listes Python en cours...")
    for i, col in enumerate(df.columns):
        try:
            df[col] = df[col].apply(lambda x: ast.literal_eval(str(x)) if pd.notna(x) else x)
            if (i + 1) % 50 == 0:
                print(f"  Converti {i + 1}/{len(df.columns)} colonnes...")
        except Exception as e:
            print(f"  Erreur colonne {col}: {str(e)}")
    print("Conversion terminée.")
    return df

# Utilisation:
# df_converted = convert_python_lists(df.copy())
""")

    elif pattern == "csv_numbers":
        print("""
import pandas as pd

def convert_csv_numbers(df):
    print("Conversion des nombres CSV en cours...")
    for i, col in enumerate(df.columns):
        try:
            df[col] = df[col].apply(lambda x: [float(i.strip()) for i in str(x).split(',')] if pd.notna(x) else x)
            if (i + 1) % 50 == 0:
                print(f"  Converti {i + 1}/{len(df.columns)} colonnes...")
        except Exception as e:
            print(f"  Erreur colonne {col}: {str(e)}")
    print("Conversion terminée.")
    return df

# Utilisation:
# df_converted = convert_csv_numbers(df.copy())
""")


def print_final_summary(df, pattern_analysis, conversion_analysis, recommendations):
    """Affiche un résumé final complet."""

    print("RÉSUMÉ FINAL COMPLET")
    print("=" * 50)

    print(f"TABLE ANALYSÉE:")
    print(f"  Lignes: {len(df):,}")
    print(f"  Colonnes: {len(df.columns):,}")
    print(f"  Taille: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    print(f"\nDISTRIBUTION DES PATTERNS:")
    total_cols = len(df.columns)
    convertible_cols = 0

    for pattern, cols in pattern_analysis.items():
        if cols:
            percentage = (len(cols) / total_cols) * 100
            is_convertible = pattern in conversion_analysis and conversion_analysis[
                pattern
            ].get("success", False)
            status = "CONVERTIBLE" if is_convertible else "NON-CONVERTIBLE"
            print(f"  {pattern}: {len(cols)} colonnes ({percentage:.1f}%) - {status}")
            if is_convertible:
                convertible_cols += len(cols)

    print(f"\nCOMPATIBILITÉ T4REC:")
    if convertible_cols > 0:
        compat_score = (convertible_cols / total_cols) * 100
        print(
            f"  Colonnes convertibles: {convertible_cols}/{total_cols} ({compat_score:.1f}%)"
        )

        if compat_score >= 80:
            print(f"  STATUT: EXCELLENT - Prêt pour T4Rec après conversion")
        elif compat_score >= 60:
            print(f"  STATUT: BON - Conversion partielle possible")
        elif compat_score >= 40:
            print(f"  STATUT: MOYEN - Travail de nettoyage requis")
        else:
            print(f"  STATUT: FAIBLE - Restructuration majeure nécessaire")
    else:
        print(f"  STATUT: INCOMPATIBLE - Aucune conversion automatique possible")

    print(f"\nPROCHAINES ÉTAPES:")
    if convertible_cols > 0:
        best_pattern = max(
            pattern_analysis.keys(),
            key=lambda x: len(pattern_analysis[x])
            if x in conversion_analysis and conversion_analysis[x].get("success", False)
            else 0,
        )
        if best_pattern in conversion_analysis and conversion_analysis[
            best_pattern
        ].get("success", False):
            print(
                f"  1. Appliquer la conversion {best_pattern} sur {len(pattern_analysis[best_pattern])} colonnes"
            )
            print(f"  2. Vérifier les résultats avec le diagnostic original")
            print(f"  3. Tester avec model_trainer")
    else:
        print(f"  1. Investigation manuelle du format des données")
        print(f"  2. Identifier la source et le processus de génération")
        print(f"  3. Développer une stratégie de conversion personnalisée")


# EXÉCUTION
if __name__ == "__main__":
    print("INVESTIGATION DU FORMAT DES ARRAYS")
    print("=" * 50)

    results = investigate_string_arrays()

    if results:
        print(f"\nINVESTIGATION TERMINÉE AVEC SUCCÈS")
        print(f"Résultats disponibles dans la variable 'results'")
    else:
        print(f"\nINVESTIGATION ÉCHOUÉE")

