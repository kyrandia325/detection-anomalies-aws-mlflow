import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px


###################
###  Catalogue  ###
###################

# afficher_informations_dataset(data)
# echantillon_data(data)
# visualisation_variables_quantitatives(data)
# statistiques_descriptives(data)
# stat_biv_deuxqualitatives(data,colonneA,colonneB)

# stat_biv_quantitatives(data)
# visualisation_corr(data)
# stat_biv_boxplot(dataset,qualitative,quantitative)
# graphe_quantitative(data,colonne)
# graphe_qualitative(data)

###################


# affichage bilan sur le jeu de données ##
## 10->
def afficher_informations_dataset(data):  # -> 0123456789
    print("-" * 100)
    print(" " * 30, "Informations sur le jeu de données")
    print("-" * 100)
    print("")
    # Informations sur le jeu de données
    data.info()

    print("")
    print("-" * 100)
    print(" " * 40, "Bilan")
    print("-" * 100)
    # La taille du jeu de données
    print("  ")
    print("1. Il y a ", data.shape[0], " lignes et ", data.shape[1], " colonnes.")
    print("  ")
    print(
        "2. Il y a ",
        data.shape[0] * data.shape[1],
        " cases pour un remplissage total de ",
        100
        - round(data.isnull().sum().sum() / (data.shape[0] * data.shape[1]), 2) * 100,
        " %",
    )
    print("  ")
    # Nombre de valeurs manquantes
    print(
        "3. Le nombre total de valeurs manquantes, toutes variables confondues : ",
        data.isnull().sum().sum(),
    )
    print("  ")
    # Affichage des parts de types de données
    print("4. Le graphique suivant représente :")
    plt.pie(
        data.dtypes.value_counts(),
        labels=data.dtypes.unique(),
        autopct=lambda x: str(round(x, 2)) + "%",
        pctdistance=1.4,
        labeldistance=2,
    )
    plt.legend(
        title="Parts des types de données",
        loc="upper right",
        bbox_to_anchor=(2, 1),
        title_fontsize="x-large",
    )
    plt.show()
    print("  ")
    print("-" * 100)

    doublons = []
    for i in data.columns:
        doublons.append(data.loc[data[i].duplicated() == True].shape[0])

    # Affichage d'un tableau de remplissage des variables
    Remplissage = pd.DataFrame()
    Remplissage["Nombre de valeurs manquantes"] = data.isnull().sum()
    Remplissage["Nombre de valeurs présentes"] = data.notnull().sum()
    Remplissage["Taux de remplissage"] = round(
        ((data.shape[0] - Remplissage["Nombre de valeurs manquantes"]) / data.shape[0])
        * 100,
        2,
    )
    Remplissage["Nombre de valeurs uniques"] = data.nunique()
    Remplissage["Doublons"] = doublons
    Remplissage.sort_values(by="Taux de remplissage", ascending=False)
    return Remplissage


## affichage de plusieurs lignes du jeu de données ##
## 2
def echantillon_data(data):
    # //Utilisationde la méthode "setOption()"
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    # Les trois premières, trois au hasard et trois dernières lignes du jeu de données
    PremierDernier = pd.DataFrame()
    PremierDernier = pd.concat([data.head(3), data.sample(3), data.tail(3)])
    return PremierDernier


## courbe de densité et boîte à moustaches pour chaque variable quantitative ##
## 3
def visualisation_variables_quantitatives(data):
    # grille blanche
    sns.set_style("whitegrid")

    # sélection des variables quantitatives
    quantitative_columns = data.select_dtypes(["int64", "float64"]).columns.tolist()

    # Déterminer le nombre de lignes et de colonnes pour les subplots
    n_rows = int(np.ceil(len(quantitative_columns)))

    # Créer une figure et des axes pour les subplots
    fig, axes = plt.subplots(
        n_rows, 2, gridspec_kw={"width_ratios": [3, 1]}, figsize=(15, n_rows * 5)
    )
    axes = axes.flatten()

    # Créer un graphique pour chaque variable
    for idx, col in enumerate(quantitative_columns):
        # Supprimer les lignes avec des données manquantes pour la colonne
        dataset = data[col].dropna()

        # Créer l'histogramme et la courbe de densité
        sns.histplot(
            data=dataset, bins=30, kde=True, alpha=0.4, linewidth=2, ax=axes[2 * idx]
        )

        # Personnaliser le graphique
        axes[2 * idx].set_title(
            f"Histogramme et courbe de densité de '{col}'", fontsize=14
        )
        axes[2 * idx].set_xlabel(col, fontsize=12)
        axes[2 * idx].set_ylabel("Fréquence", fontsize=12)

        # Créer la boîte à moustaches
        sns.boxplot(y=dataset, color="lightblue", ax=axes[2 * idx + 1])

        # Personnaliser le graphique
        axes[2 * idx + 1].set_title(f"Boîte à moustaches de '{col}'", fontsize=14)
        axes[2 * idx + 1].set_xlabel("Boxplot", fontsize=12)
        axes[2 * idx + 1].set_ylabel(col, fontsize=12)

    # # Masquer les axes vides s'il y en a
    # if len(quantitative_columns) % 2 != 0:
    #     axes[-4].axis('off')
    #     axes[-3].axis('off')
    #     axes[-2].axis('off')
    #     axes[-1].axis('off')

    # Ajuster l'espacement entre les subplots
    plt.tight_layout()

    # Afficher les graphiques
    plt.show()
    return


## tableau de statistiques complet ##
## 4
def statistiques_descriptives(data):
    # variables quantitatives
    quantitative_columns = data.select_dtypes(["int64", "float64"]).columns.tolist()

    # statistiques
    stat = round(data[quantitative_columns].describe(), 2)
    return stat


## diagramme en barres cumulées pour deux variables qualitatives ##
## 5
def stat_biv_deuxqualitatives(
    data,
    colonneA,
    colonneB,
    figsize_x,
    figsize_y,
    labelsize_x,
    labelsize_y,
):
    # Créer un tableau croisé entre la variable de la colonneA et la variable de la colonneB
    cross_tab = pd.crosstab(data[colonneA], data[colonneB])

    # Créer un diagramme en barres cumulées
    ax = cross_tab.plot.bar(
        stacked=True, figsize=(figsize_x, figsize_y), colormap="viridis"
    )

    # Personnaliser le graphique
    ax.set_title(
        f"Diagramme en barres cumulées entre {colonneA} et {colonneB}", fontsize=16
    )
    ax.set_xlabel(f"{colonneA}", fontsize=14)
    ax.set_ylabel("Nombre d'occurrences", fontsize=14)
    ax.legend(
        title=f"{colonneA}", fontsize=16, bbox_to_anchor=(1.05, 1), loc="upper left"
    )

    # Ajuster les étiquettes de l'axe des x
    plt.xticks(rotation=30, fontsize=labelsize_x)
    plt.yticks(fontsize=labelsize_y)
    # Afficher le graphique
    plt.show()
    return


## tableau de corrélations sur variables quantitatives ##
## 6
def stat_biv_quantitatives(data, x, y, taille_chiffres):
    # Créer un graphique à partir du tableau de corrélation
    float_columns = data.select_dtypes(include=["float64", "int64"])
    correlation_matrix = float_columns.corr()
    plt.figure(figsize=(x, y))
    sns.heatmap(
        round(correlation_matrix, 2),
        annot=True,
        annot_kws={"fontsize": taille_chiffres},
        cmap="PuOr",
        vmin=-1,
        vmax=1,
        fmt=".1f",
        linewidth=0.1,
    )
    # Personnaliser le graphique
    plt.title(
        "Tableau de corrélations entre les variables quantitatives",
        fontsize=16,
        position=(0.5, 2),
    )
    # Afficher le graphique
    plt.show()


## tableau de corrélations toutes variables ##
## 7
def visualisation_corr(data, fig_sizex, fig_sizey, rot, x_size, y_size):
    # visualisation des corrélations entre les variables du dataset
    # observation de l'évolution selon les modifications apportées
    float_columns = data.select_dtypes(include=["float64", "int64"])
    correlation_matrix = float_columns.corr()
    plt.figure(figsize=(fig_sizex, fig_sizey))
    sns.heatmap(
        round(correlation_matrix, 2),
        annot=True,
        annot_kws={"fontsize": 18},
        cmap="PuOr",
        vmin=-1,
        vmax=1,
        fmt=".1f",
        linewidth=0.1,
    )
    # Personnaliser le graphique
    plt.title("Corrélations du jeu de données", position=(0.5, 2), fontsize=18)
    plt.xticks(rotation=rot, fontsize=x_size)
    plt.yticks(fontsize=y_size)
    plt.show()
    return


## boîtes à moustaches entre une quantitative et une qualitative ##
## 8
def stat_biv_boxplot(dataset, qualitative, quantitative):
    df = dataset.copy()
    df = df.sort_values(qualitative)
    # Sélectionner uniquement les colonnes de type 'float64'
    float_columns = df[quantitative]

    # Configurer les options d'affichage
    sns.set_style("whitegrid")

    # Déterminer le nombre de lignes et de colonnes pour les subplots
    n_rows = int(np.ceil(len(float_columns) / 2))

    # Créer une figure et des axes pour les subplots
    fig, axes = plt.subplots(n_rows, 2, figsize=(20, n_rows * 5))
    axes = axes.flatten()

    # Créer un graphique pour chaque variable numérique
    for idx, col in enumerate(float_columns):
        sns.boxplot(x=qualitative, y=col, data=df, ax=axes[idx])

        # Personnaliser le graphique
        axes[idx].set_title(f"Boxplot de '{col}' par {qualitative}", fontsize=16)
        axes[idx].set_xlabel(f"{qualitative}", fontsize=14)
        axes[idx].set_ylabel(col, fontsize=14)
        axes[idx].tick_params(axis="x", labelsize=14)

    # Masquer les axes vides s'il y en a
    if len(float_columns) % 2 != 0:
        axes[-1].axis("off")

    # Ajuster l'espacement entre les subplots
    plt.tight_layout()

    # Afficher les graphiques
    plt.show()
    return


## graphique de représentation d'une variable quantitative
## 9
def graphe_quantitative(data, colonne):
    # régler la taille de la fonte ou police
    plt.rcParams.update({"font.size": 13})
    # taille de la figure
    plt.figure(figsize=(10, 6))

    sns.set_style("whitegrid")
    sns.histplot(data[colonne])
    plt.title(f"Distribution de {colonne}", fontsize=14)
    plt.show()
    return


## graphique de représentation d'une variable qualitative
## 10
def graphe_qualitative(data):
    qualitative_vars = data.select_dtypes(
        include=["object", "bool"]
    )  # Sélectionne les variables qualitatives du dataframe

    for column in qualitative_vars:
        if float(data[column].nunique()) < 10:
            nombre = qualitative_vars[
                column
            ].value_counts()  # Compte les occurrences de chaque valeur
            labels = nombre.index.tolist()  # Utilise les valeurs uniques comme libellés
            values = nombre.values.tolist()  # Utilise les occurrences comme données

            plt.figure()
            plt.pie(values, labels=labels, autopct="%1.1f%%")
            plt.legend(
                title="Parts des types de données",
                loc="upper right",
                bbox_to_anchor=(2, 1),
                title_fontsize="large",
            )
            plt.axis("equal")
            plt.title(column)
            plt.show()
        else:
            print("")
            print(f". {column} a trop de catégories")
            print("")
    return


## graphique affichant une ligne d'une variable choisie sur un tableau de corrélations
## 11
def ligne_correlation_matrix(data, variable_of_interest):
    # Création de la matrice de corrélations
    corr_matrix = data.corr()

    # Filtrage des corrélations uniquement pour la variable d'intérêt
    corr_matrix_filtered = corr_matrix.loc[[variable_of_interest]]
    corr_matrix_filtered.drop(variable_of_interest, axis=1, inplace=True)

    # Affichage de la matrice de corrélations filtrée
    plt.figure(figsize=(14, 1))
    sns.heatmap(
        corr_matrix_filtered,
        annot=True,
        cmap="PuOr",
        vmin=-1,
        vmax=1,
        fmt=".1f",
        linewidth=0.1,
        center=0,
    )
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.title(f"Corrélations de {variable_of_interest}")
    plt.show()
    return


## boîte à moustache entre une variable qualitative et une variable quantitative
## 12
def plot_boxplot(data, qualitative_var, quantitative_var, title):
    sns.boxplot(x=qualitative_var, y=quantitative_var, data=data)
    sns.set_style("whitegrid")
    plt.xticks(rotation=30, fontsize=12)  # Rotation de 30° des noms sur l'axe x
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=14)  # Ajout du titre
    plt.show()
    return


## décompte de valeurs pour chaque catégorie d'une variable qualitative
## 13
def decompte_qualitative(data, nom):
    decompte = pd.DataFrame()
    temporaire = data.value_counts()
    pourcentage = round((temporaire / temporaire.sum()) * 100, 2)
    serie_temp = pd.Series(temporaire, name=nom)
    serie_pour = pd.Series(pourcentage, name="Pourcentage %")
    decompte = pd.concat([decompte, serie_temp, serie_pour], axis=1)
    return decompte


## tableau de contingence entre deux variables qualitatives
## 14
def tableau_de_contingence(colonne_a, colonne_b):
    cross_tab_a = pd.crosstab(colonne_a, colonne_b)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cross_tab_a,
        annot=True,
        annot_kws={"fontsize": 13},
        cmap="viridis",
        vmin=0,
        vmax=565,
        fmt=".1f",
        linewidth=0.1,
    )
    plt.xticks(rotation=30, fontsize=13)
    plt.yticks(fontsize=13)
    plt.title("Tableau de contingence", fontsize=16)
    plt.show()
    plt.close()
    return
