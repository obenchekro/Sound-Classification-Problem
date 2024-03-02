# %%
"""Fonctions pour gérer et analyser les datasets. """
# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import ipywidgets as widgets
from ipywidgets import interact, Dropdown, interactive_output


# %%
# Modification des paramètres d'affichage de pandas
# pd.set_option("display.max_rows", 100)
# pd.set_option("display.max_columns", 30)

# Pour importer depuis un autre dossier
# import sys
# sys.path.append(r'C:\Users\...\code_perso')
# front fonctions_rol import ...

# black dans le jupyternotebook
# import jupyter_black
# jupyter_black.load()


# %%
def load_dataset(file_path):
    """
    Fonction d'import de fichiers de type csv, xls ou xlsx

    :param file_path: Chemin vers le fichier à charger
    :return: Un dataframe contenant les données du fichier
    """
    # Modification des paramètres d'affichage de pandas
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 30)
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() in [".xls", ".xlsx"]:
        # Lecture des fichiers Excel
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            print("Erreur lors de la lecture du fichier Excel :", e)
            return None

    elif file_extension.lower() == ".csv":
        # Détecter le séparateur
        try:
            df = pd.read_csv(file_path, sep=None, engine="python")
        except Exception as e:
            print("Erreur lors de la lecture du fichier CSV :", e)
            return None

    else:
        print(
            "Type de fichier non supporté. Seuls sont pris en charge : csv, xls, xlsx"
        )
        return None

    return df


# %%
def df_info(df):
    """
    Description d'un dataset
    Fonction qui affiche les informations de base du dataframe
    et retourne un dataframe avec des statistiques pour chaque colonne.
    """
    # Calcul de certaines statistiques de base
    len_df = len(df)
    all_columns = len(df.columns)
    all_nan = df.isna().sum().sum()
    list_of_numerics = df.select_dtypes(include=["float", "int"]).columns
    all_num = len(list_of_numerics)
    all_cat = all_columns - len(list_of_numerics)
    # Affichage des informations de base
    print(
        f"""
    Longueur du dataset : {len_df} enregistrements
    Nombre de colonnes : {all_columns}
    Nombre de colonnes numériques : {all_num}
    Nombre de colonnes catégorielles : {all_cat}
    Nombre de valeurs manquantes : {all_nan}
    """
    )

    # Préparation de l'échantillon des colonnes
    echantillonColonnes = []
    for i in df.columns:
        listcolumn = str(list(df[i].head(5)))
        echantillonColonnes.append(listcolumn)

    # Détection des outliers avec la méthode interquartiles.
    def outliers(df):
        outliers = df.apply(
            lambda x: sum(
                (x < (x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25))))
                | (x > (x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25))))
            )
            if x.name in list_of_numerics
            else 0
        )
        return outliers

    # Calcul des statistiques pour chaque colonne
    obs = pd.DataFrame(
        {
            "type": list(df.dtypes),
            "Echantillon": echantillonColonnes,
            "Nbr V manquantes": df.isna().sum(),
            "% de V manquantes": round(df.isna().sum() / len_df * 100, 2),
            "Nbr L dupliquées": (df.duplicated()).sum(),
            "Nbr V unique": df.nunique(),
            "Mode": df.mode().iloc[0],
            "Nbr Outliers": outliers(df),
            "% Outliers": round(outliers(df) / len_df * 100, 2),
            "Moyenne": df[list_of_numerics].mean(),
            "Médiane": df[list_of_numerics].median(),
            "Minimum": df[list_of_numerics].min(),
            "Maximum": df[list_of_numerics].max(),
            "Écart type": df[list_of_numerics].std(),
            "Variance": df[list_of_numerics].var(),
            "Asymétrie (skew)": df[list_of_numerics].skew(),
            "Aplatissement (kurt)": df[list_of_numerics].kurt(),
            ".25 quartile": df[list_of_numerics].quantile(0.25),
            ".50 quartile": df[list_of_numerics].quantile(0.50),
            ".75 quartile": df[list_of_numerics].quantile(0.75),
            ".1 percentile": df[list_of_numerics].quantile(0.1),
            ".9 percentile": df[list_of_numerics].quantile(0.9),
        },
        index=df.columns,
    )

    # Remplacer les statistiques non pertinentes pour les colonnes non numériques par 'N/A'
    cols_to_modify = [
        "Moyenne",
        "Médiane",
        "Minimum",
        "Maximum",
        "Écart type",
        "Variance",
        "Asymétrie (skew)",
        "Aplatissement (kurt)",
        ".25 quartile",
        ".50 quartile",
        ".75 quartile",
        ".1 percentile",
        ".9 percentile",
    ]
    for col in cols_to_modify:
        obs.loc[~obs.index.isin(list_of_numerics), col] = "N/A"

    return obs


#  %%
def distribution(df, specific_col=None, orientation='vertical'):
    sns.set(style="whitegrid")  # More streamlined grid style

    if specific_col is None:

        @interact(col_name=Dropdown(options=df.columns, description="Select Column"),
                  orientation=Dropdown(options=['vertical', 'horizontal'], description="Orientation"))
        def plot_distribution(col_name, orientation):
            _plot_distribution(df, col_name, orientation)

    else:
        _plot_distribution(df, specific_col, orientation)


def _plot_distribution(df, col_name, orientation='vertical'):
    column_type = df[col_name].dtype
    if column_type == "object":
        col_type_str = "catégorielle"
        fig = plt.figure(figsize=(5, 4))
        if orientation == 'vertical':
            sns.countplot(data=df, x=col_name, palette="Set2")
        else:
            sns.countplot(data=df, y=col_name, palette="Set2")
    else:
        col_type_str = "numérique"
        fig = plt.figure(figsize=(5, 4))
        if orientation == 'vertical':
            sns.histplot(df[col_name], bins=15, kde=True, color="steelblue")
        else:
            sns.histplot(df[col_name], bins=15, kde=True, color="steelblue", orientation='horizontal')

    plt.title(
        f"Distribution de la colonne {col_name} ({col_type_str}, {orientation})",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel(col_name if orientation == 'vertical' else "Fréquence", fontsize=12)
    plt.ylabel("Fréquence" if orientation == 'vertical' else col_name, fontsize=12)
    plt.tick_params(axis="both", labelsize=10)
    plt.xticks(rotation=45 if orientation == 'vertical' else 0)
    plt.tight_layout()
    plt.show()



def detect_outliers(df, column_name):
    """
    Détection des outliers pour une colonne spécifique dans un DataFrame.
    Utilise la méthode interquartile (définie dans la fonction df_info).
    """
    # Calcul des quantiles pour la colonne spécifiée
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1

    # Calcul des seuils pour détecter les outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filtrage des outliers dans la colonne spécifiée
    outliers_df = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]

    return outliers_df


def sns_plot(
    dataframe,
    initial_columns=None,
    graph_types=["Ligne", "Barre"],
    x_label_orientations=["Horizontal", "Vertical"],
    sort_orders=["Aucun", "Croissant", "Décroissant"],
):
    """
    Cette fonction crée un graphique interactif à partir d'un DataFrame avec seaborn.
    Elle propose des listes déroulantes pour choisir les colonnes à afficher, le type de graphique,
    l'orientation des labels X et l'ordre de tri, ainsi que des sliders pour choisir la taille et le ratio de la figure.
    Les options de ces choix peuvent être personnalisées en passant des arguments à la fonction.
    """

    # Si aucune colonne initiale spécifique n'est fournie, les premières colonnes du dataframe sont utilisées
    if not initial_columns:
        initial_columns = [dataframe.columns[0], dataframe.columns[1]]
    else:
        initial_columns = initial_columns
    # Création des widgets de sélection
    column_dropdown = widgets.Dropdown(
        options=dataframe.columns,
        value=initial_columns[0],
        description="Colonne Y :",
    )
    x_column_dropdown = widgets.Dropdown(
        options=dataframe.columns,
        value=initial_columns[1],
        description="Colonne X :",
    )
    plot_type_dropdown = widgets.Dropdown(
        options=graph_types, value=graph_types[0], description="Type de graphique:"
    )
    x_labels_dropdown = widgets.Dropdown(
        options=x_label_orientations,
        value=x_label_orientations[0],
        description="Orientation des labels X:",
    )
    sort_order_dropdown = widgets.Dropdown(
        options=sort_orders, value=sort_orders[0], description="Ordre de tri:"
    )
    size_slider = widgets.IntSlider(
        value=6, min=4, max=20, step=1, description="Taille de la figure:"
    )
    ratio_slider = widgets.FloatSlider(
        value=0.8, min=0.5, max=2.0, step=0.1, description="Ratio de la figure:"
    )

    # Fonction de mise à jour du graphique
    def update_plot(
        column, x_column, plot_type, x_labels, sort_order, fig_size, fig_ratio
    ):
        # Vérification et application de l'ordre de tri
        if sort_order != "Aucun":
            dataframe.sort_values(
                by=column, ascending=(sort_order == "Croissant"), inplace=True
            )

        # Création de la figure
        plt.figure(figsize=(fig_size, fig_size * fig_ratio))

        # Choix du type de graphique
        if plot_type == "Ligne":
            sns.lineplot(x=dataframe[x_column], y=dataframe[column])
        elif plot_type == "Barre":
            sns.barplot(x=dataframe[x_column], y=dataframe[column])

        # Configuration des labels et du titre
        plt.xlabel(x_column)
        plt.ylabel(column)
        plt.title(f"{column} par {x_column}")

        # Configuration de l'orientation des labels X
        if x_labels == "Vertical":
            plt.xticks(rotation=90)

        plt.tight_layout()
        plt.show()

    # Création du graphique interactif
    graph_output = interactive_output(
        update_plot,
        {
            "column": column_dropdown,
            "x_column": x_column_dropdown,
            "plot_type": plot_type_dropdown,
            "x_labels": x_labels_dropdown,
            "sort_order": sort_order_dropdown,
            "fig_size": size_slider,
            "fig_ratio": ratio_slider,
        },
    )

    # Affichage des widgets de sélection en deux lignes
    display(widgets.HBox([column_dropdown, x_column_dropdown, plot_type_dropdown]))
    display(
        widgets.HBox(
            [x_labels_dropdown, sort_order_dropdown, size_slider, ratio_slider]
        )
    )
    display(graph_output)
