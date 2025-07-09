#!/usr/bin/env python3
"""
DIAGNOSTIC COMPLET: TABLE T4REC_DF (384 COLONNES)
===============================================

Script de diagnostic pour analyser la table t4_rec_df dans Dataiku
et vérifier la compatibilité avec le système T4Rec.
"""

import dataiku
import numpy as np
import pandas as pd
import torch
from collections import Counter
import traceback


def load_and_analyze_t4rec_df():
    """Charge et analyse complètement la table t4_rec_df."""

    print("🔍 DIAGNOSTIC COMPLET: TABLE T4REC_DF")
    print("=" * 60)

    try:
        # 1. CHARGEMENT DE LA TABLE
        print("\n📂 1. CHARGEMENT DE LA TABLE")
        print("-" * 30)

        dataset = dataiku.Dataset("t4_rec_df")
        df = dataset.get_dataframe()

        print(f"✅ Table chargée avec succès:")
        print(f"  • Lignes: {len(df):,}")
        print(f"  • Colonnes: {len(df.columns):,}")
        print(
            f"  • Taille mémoire: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
        )

        # 2. ANALYSE DES TYPES DE COLONNES
        print("\n🔢 2. ANALYSE DES TYPES DE COLONNES")
        print("-" * 30)

        type_analysis = analyze_column_types(df)

        # 3. ANALYSE DES DIMENSIONS (ARRAYS)
        print("\n📏 3. ANALYSE DES DIMENSIONS DES ARRAYS")
        print("-" * 30)

        dimension_analysis = analyze_array_dimensions(df)

        # 4. VÉRIFICATIONS SPÉCIFIQUES T4REC
        print("\n🎯 4. VÉRIFICATIONS SPÉCIFIQUES T4REC")
        print("-" * 30)

        t4rec_analysis = analyze_t4rec_compatibility(df)

        # 5. VÉRIFICATIONS DLPACK/MERLIN
        print("\n🔄 5. VÉRIFICATIONS DLPACK/MERLIN")
        print("-" * 30)

        dlpack_analysis = analyze_dlpack_merlin_compatibility(df)

        # 6. RAPPORT FINAL
        print("\n📊 6. RAPPORT FINAL")
        print("-" * 30)

        final_report = generate_final_report(
            df, type_analysis, dimension_analysis, t4rec_analysis, dlpack_analysis
        )

        return {
            "dataframe": df,
            "type_analysis": type_analysis,
            "dimension_analysis": dimension_analysis,
            "t4rec_analysis": t4rec_analysis,
            "dlpack_analysis": dlpack_analysis,
            "final_report": final_report,
        }

    except Exception as e:
        print(f"❌ ERREUR lors du chargement:")
        print(f"  • Erreur: {str(e)}")
        print(f"  • Détails: {traceback.format_exc()}")
        return None


def analyze_column_types(df):
    """Analyse les types de toutes les colonnes."""

    type_stats = {
        "array_columns": [],
        "numeric_columns": [],
        "string_columns": [],
        "object_columns": [],
        "bool_columns": [],
        "datetime_columns": [],
        "problematic_columns": [],
    }

    print(f"📊 Analyse de {len(df.columns)} colonnes...")

    for i, col in enumerate(df.columns):
        try:
            dtype = df[col].dtype
            sample_value = df[col].iloc[0] if len(df) > 0 else None

            # Classification des types
            if dtype == "object":
                # Vérifier si c'est des arrays
                if sample_value is not None and isinstance(
                    sample_value, (list, np.ndarray)
                ):
                    type_stats["array_columns"].append(
                        {
                            "column": col,
                            "dtype": str(dtype),
                            "sample_type": type(sample_value).__name__,
                            "sample_value": str(sample_value)[:100] + "..."
                            if len(str(sample_value)) > 100
                            else str(sample_value),
                        }
                    )
                else:
                    type_stats["object_columns"].append(
                        {
                            "column": col,
                            "dtype": str(dtype),
                            "sample_value": str(sample_value)[:50] + "..."
                            if sample_value and len(str(sample_value)) > 50
                            else str(sample_value),
                        }
                    )

            elif "int" in str(dtype):
                type_stats["numeric_columns"].append(
                    {
                        "column": col,
                        "dtype": str(dtype),
                        "min": df[col].min(),
                        "max": df[col].max(),
                    }
                )

            elif "float" in str(dtype):
                type_stats["numeric_columns"].append(
                    {
                        "column": col,
                        "dtype": str(dtype),
                        "min": df[col].min(),
                        "max": df[col].max(),
                    }
                )

            elif "bool" in str(dtype):
                type_stats["bool_columns"].append(
                    {"column": col, "dtype": str(dtype), "true_count": df[col].sum()}
                )

            elif "datetime" in str(dtype):
                type_stats["datetime_columns"].append(
                    {
                        "column": col,
                        "dtype": str(dtype),
                        "min_date": df[col].min(),
                        "max_date": df[col].max(),
                    }
                )

            else:
                type_stats["problematic_columns"].append(
                    {
                        "column": col,
                        "dtype": str(dtype),
                        "sample_value": str(sample_value),
                    }
                )

            # Progress
            if (i + 1) % 50 == 0:
                print(f"  • Analysé {i + 1}/{len(df.columns)} colonnes...")

        except Exception as e:
            type_stats["problematic_columns"].append(
                {"column": col, "dtype": "ERROR", "error": str(e)}
            )

    # Résumé
    print(f"\n📋 RÉSUMÉ DES TYPES:")
    print(f"  • Arrays/Lists: {len(type_stats['array_columns'])}")
    print(f"  • Numériques: {len(type_stats['numeric_columns'])}")
    print(f"  • Strings/Objects: {len(type_stats['object_columns'])}")
    print(f"  • Booléens: {len(type_stats['bool_columns'])}")
    print(f"  • Dates: {len(type_stats['datetime_columns'])}")
    print(f"  • Problématiques: {len(type_stats['problematic_columns'])}")

    # Détails des colonnes arrays
    if type_stats["array_columns"]:
        print(f"\n🔍 DÉTAILS COLONNES ARRAYS (premières 5):")
        for i, arr_col in enumerate(type_stats["array_columns"][:5]):
            print(f"  {i + 1}. {arr_col['column']}")
            print(f"     • Type: {arr_col['sample_type']}")
            print(f"     • Exemple: {arr_col['sample_value']}")

    # Colonnes problématiques
    if type_stats["problematic_columns"]:
        print(f"\n⚠️ COLONNES PROBLÉMATIQUES:")
        for prob_col in type_stats["problematic_columns"][:5]:
            print(
                f"  • {prob_col['column']}: {prob_col.get('error', prob_col['dtype'])}"
            )

    return type_stats


def analyze_array_dimensions(df):
    """Analyse les dimensions des arrays dans les colonnes."""

    dimension_stats = {
        "array_dimensions": {},
        "uniform_lengths": {},
        "dimension_issues": [],
    }

    # Identifier les colonnes avec des arrays
    array_columns = []
    for col in df.columns:
        try:
            sample = df[col].iloc[0]
            if isinstance(sample, (list, np.ndarray)):
                array_columns.append(col)
        except:
            continue

    print(f"📊 Analyse des dimensions pour {len(array_columns)} colonnes arrays...")

    for i, col in enumerate(array_columns[:20]):  # Limite à 20 pour la démo
        try:
            # Analyser quelques échantillons
            sample_sizes = []
            sample_shapes = []

            for idx in range(min(100, len(df))):  # Échantillon de 100 lignes
                try:
                    value = df[col].iloc[idx]
                    if isinstance(value, (list, np.ndarray)):
                        if isinstance(value, list):
                            sample_sizes.append(len(value))
                            sample_shapes.append(f"1D-{len(value)}")
                        else:
                            sample_sizes.append(value.size)
                            sample_shapes.append(str(value.shape))
                except:
                    continue

            if sample_sizes:
                dimension_stats["array_dimensions"][col] = {
                    "min_size": min(sample_sizes),
                    "max_size": max(sample_sizes),
                    "avg_size": np.mean(sample_sizes),
                    "unique_shapes": list(set(sample_shapes[:10])),  # Top 10 shapes
                    "is_uniform": len(set(sample_sizes)) == 1,
                }

                # Vérifier l'uniformité
                if len(set(sample_sizes)) == 1:
                    dimension_stats["uniform_lengths"][col] = sample_sizes[0]
                else:
                    dimension_stats["dimension_issues"].append(
                        {
                            "column": col,
                            "issue": "Variable lengths",
                            "sizes_range": f"{min(sample_sizes)}-{max(sample_sizes)}",
                        }
                    )

            if (i + 1) % 10 == 0:
                print(
                    f"  • Analysé {i + 1}/{min(20, len(array_columns))} colonnes arrays..."
                )

        except Exception as e:
            dimension_stats["dimension_issues"].append(
                {"column": col, "issue": "Analysis error", "error": str(e)}
            )

    # Résumé
    print(f"\n📋 RÉSUMÉ DES DIMENSIONS:")
    print(f"  • Colonnes arrays analysées: {len(dimension_stats['array_dimensions'])}")
    print(f"  • Longueurs uniformes: {len(dimension_stats['uniform_lengths'])}")
    print(f"  • Problèmes détectés: {len(dimension_stats['dimension_issues'])}")

    # Détails
    if dimension_stats["uniform_lengths"]:
        print(f"\n✅ COLONNES AVEC LONGUEURS UNIFORMES:")
        for col, length in list(dimension_stats["uniform_lengths"].items())[:5]:
            print(f"  • {col}: longueur {length}")

    if dimension_stats["dimension_issues"]:
        print(f"\n⚠️ PROBLÈMES DE DIMENSIONS:")
        for issue in dimension_stats["dimension_issues"][:5]:
            print(f"  • {issue['column']}: {issue['issue']}")

    return dimension_stats


def analyze_t4rec_compatibility(df):
    """Vérifie la compatibilité avec T4Rec."""

    t4rec_checks = {
        "tensor_convertible": [],
        "sequence_candidates": [],
        "target_ready": [],
        "compatibility_issues": [],
    }

    print("🎯 Vérification compatibilité T4Rec...")

    # Vérifier les colonnes pour la conversion tensor
    array_columns = []
    for col in df.columns:
        try:
            sample = df[col].iloc[0]
            if isinstance(sample, (list, np.ndarray)):
                array_columns.append(col)
        except:
            continue

    print(f"📊 Test de conversion sur {min(10, len(array_columns))} colonnes arrays...")

    for col in array_columns[:10]:  # Test sur 10 colonnes
        try:
            # Test de conversion en array numpy
            sample_data = df[col].iloc[:5].tolist()  # 5 échantillons

            # Vérifier si c'est convertible en array uniforme
            try:
                test_array = np.array(sample_data)
                if test_array.ndim == 2:  # Séquences 2D
                    # Test de conversion tensor
                    test_tensor = torch.LongTensor(test_array)

                    # Test de création input/target pairs
                    if test_array.shape[1] > 1:
                        inputs = test_tensor[:, :-1]
                        targets = test_tensor[:, 1:]

                        t4rec_checks["tensor_convertible"].append(col)
                        t4rec_checks["sequence_candidates"].append(
                            {
                                "column": col,
                                "shape": test_array.shape,
                                "seq_length": test_array.shape[1],
                                "can_create_targets": True,
                            }
                        )

                        if inputs.shape == targets.shape:
                            t4rec_checks["target_ready"].append(col)

                        print(
                            f"  ✅ {col}: Compatible T4Rec (shape: {test_array.shape})"
                        )
                    else:
                        t4rec_checks["compatibility_issues"].append(
                            {
                                "column": col,
                                "issue": "Séquence trop courte pour targets",
                            }
                        )
                else:
                    t4rec_checks["compatibility_issues"].append(
                        {
                            "column": col,
                            "issue": f"Dimension incorrecte: {test_array.ndim}D",
                        }
                    )

            except Exception as tensor_error:
                t4rec_checks["compatibility_issues"].append(
                    {
                        "column": col,
                        "issue": f"Conversion tensor échouée: {str(tensor_error)}",
                    }
                )

        except Exception as e:
            t4rec_checks["compatibility_issues"].append(
                {"column": col, "issue": f"Erreur analyse: {str(e)}"}
            )

    # Résumé
    print(f"\n📋 COMPATIBILITÉ T4REC:")
    print(f"  • Convertibles en tensor: {len(t4rec_checks['tensor_convertible'])}")
    print(f"  • Candidats séquences: {len(t4rec_checks['sequence_candidates'])}")
    print(f"  • Prêts pour targets: {len(t4rec_checks['target_ready'])}")
    print(f"  • Problèmes: {len(t4rec_checks['compatibility_issues'])}")

    return t4rec_checks


def analyze_dlpack_merlin_compatibility(df):
    """Vérifie la compatibilité DLPack/Merlin."""

    dlpack_checks = {
        "dlpack_compatible": [],
        "merlin_compatible": [],
        "conversion_needed": [],
        "incompatible": [],
    }

    print("🔄 Vérification compatibilité DLPack/Merlin...")

    for col in df.columns[:20]:  # Test sur 20 colonnes
        try:
            dtype = df[col].dtype
            sample = df[col].iloc[0]

            # Check DLPack compatibility
            if dtype in [np.int32, np.int64, np.float32]:
                dlpack_checks["dlpack_compatible"].append(col)
                dlpack_checks["merlin_compatible"].append(col)
            elif "int" in str(dtype) or "float" in str(dtype):
                dlpack_checks["conversion_needed"].append(
                    {
                        "column": col,
                        "current_type": str(dtype),
                        "target_type": "int32" if "int" in str(dtype) else "float32",
                    }
                )
            elif dtype == "object":
                if isinstance(sample, (list, np.ndarray)):
                    dlpack_checks["conversion_needed"].append(
                        {
                            "column": col,
                            "current_type": "object (array)",
                            "target_type": "need array analysis",
                        }
                    )
                else:
                    dlpack_checks["incompatible"].append(
                        {"column": col, "issue": "Object type non-array"}
                    )
            else:
                dlpack_checks["incompatible"].append(
                    {"column": col, "issue": f"Type {dtype} non supporté"}
                )

        except Exception as e:
            dlpack_checks["incompatible"].append(
                {"column": col, "issue": f"Erreur: {str(e)}"}
            )

    # Résumé
    print(f"\n📋 COMPATIBILITÉ DLPACK/MERLIN:")
    print(f"  • Compatible directement: {len(dlpack_checks['dlpack_compatible'])}")
    print(f"  • Compatible Merlin: {len(dlpack_checks['merlin_compatible'])}")
    print(f"  • Conversion nécessaire: {len(dlpack_checks['conversion_needed'])}")
    print(f"  • Incompatible: {len(dlpack_checks['incompatible'])}")

    return dlpack_checks


def generate_final_report(
    df, type_analysis, dimension_analysis, t4rec_analysis, dlpack_analysis
):
    """Génère le rapport final avec recommandations."""

    print("📊 GÉNÉRATION DU RAPPORT FINAL")

    report = {
        "summary": {
            "total_columns": len(df.columns),
            "total_rows": len(df),
            "array_columns": len(type_analysis["array_columns"]),
            "t4rec_ready": len(t4rec_analysis["target_ready"]),
            "major_issues": 0,
        },
        "recommendations": [],
        "next_steps": [],
    }

    # Calcul des problèmes majeurs
    major_issues = 0
    major_issues += len(type_analysis["problematic_columns"])
    major_issues += len(dimension_analysis["dimension_issues"])
    major_issues += len(t4rec_analysis["compatibility_issues"])
    major_issues += len(dlpack_analysis["incompatible"])

    report["summary"]["major_issues"] = major_issues

    # Recommandations
    if len(t4rec_analysis["target_ready"]) > 0:
        report["recommendations"].append(
            f"✅ EXCELLENT: {len(t4rec_analysis['target_ready'])} colonnes prêtes pour T4Rec"
        )

    if len(type_analysis["array_columns"]) > len(t4rec_analysis["target_ready"]):
        diff = len(type_analysis["array_columns"]) - len(t4rec_analysis["target_ready"])
        report["recommendations"].append(
            f"⚠️ {diff} colonnes arrays nécessitent des ajustements pour T4Rec"
        )

    if len(dlpack_analysis["conversion_needed"]) > 0:
        report["recommendations"].append(
            f"🔄 {len(dlpack_analysis['conversion_needed'])} colonnes nécessitent conversion DLPack"
        )

    if major_issues > 0:
        report["recommendations"].append(
            f"❌ {major_issues} problèmes majeurs à résoudre"
        )

    # Étapes suivantes
    if len(t4rec_analysis["target_ready"]) > 0:
        report["next_steps"].append("1. Tester model_trainer avec colonnes prêtes")
        report["next_steps"].append("2. Convertir autres colonnes arrays")
    else:
        report["next_steps"].append("1. Analyser structure des arrays")
        report["next_steps"].append("2. Standardiser dimensions")
        report["next_steps"].append("3. Créer paires input/target")

    if len(dlpack_analysis["conversion_needed"]) > 0:
        report["next_steps"].append("4. Convertir types pour DLPack")

    # Affichage du rapport
    print(f"\n" + "=" * 50)
    print(f"📊 RAPPORT FINAL")
    print(f"=" * 50)

    print(f"\n📈 RÉSUMÉ:")
    for key, value in report["summary"].items():
        print(f"  • {key}: {value:,}")

    print(f"\n💡 RECOMMANDATIONS:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"  {i}. {rec}")

    print(f"\n🚀 PROCHAINES ÉTAPES:")
    for step in report["next_steps"]:
        print(f"  {step}")

    # Score de compatibilité
    if report["summary"]["array_columns"] > 0:
        compatibility_score = (
            report["summary"]["t4rec_ready"] / report["summary"]["array_columns"]
        ) * 100
        print(f"\n🎯 SCORE DE COMPATIBILITÉ T4REC: {compatibility_score:.1f}%")

        if compatibility_score >= 80:
            print("  ✅ EXCELLENT - Prêt pour production")
        elif compatibility_score >= 60:
            print("  ⚠️ BON - Quelques ajustements nécessaires")
        elif compatibility_score >= 40:
            print("  🔄 MOYEN - Travail de préparation requis")
        else:
            print("  ❌ FAIBLE - Restructuration majeure nécessaire")

    return report


# SCRIPT PRINCIPAL POUR DATAIKU
if __name__ == "__main__":
    print("🚀 LANCEMENT DU DIAGNOSTIC T4REC_DF")
    print("=" * 60)

    # Exécuter l'analyse complète
    results = load_and_analyze_t4rec_df()

    if results:
        print(f"\n🎉 DIAGNOSTIC TERMINÉ AVEC SUCCÈS!")
        print(f"📁 Résultats disponibles dans la variable 'results'")

        # Sauvegarder le rapport (optionnel)
        # import pickle
        # with open('/tmp/t4rec_diagnostic_report.pkl', 'wb') as f:
        #     pickle.dump(results, f)
        # print(f"💾 Rapport sauvegardé: /tmp/t4rec_diagnostic_report.pkl")

    else:
        print(f"\n❌ DIAGNOSTIC ÉCHOUÉ")
        print(f"Vérifiez que la table 't4_rec_df' existe dans Dataiku")
