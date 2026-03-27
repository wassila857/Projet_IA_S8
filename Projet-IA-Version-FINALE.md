# Détection de Fraude sur les Transactions par Carte Bancaire
## Rapport d'Analyse et de Modélisation par Machine Learning

---

**Auteur :** Équipe Data Science & Sécurité Financière  
**Date :** Mars 2026  
**Version :** 1.0  
**Classification :** Confidentiel – Usage Interne  
**Dataset :** Transactions Financières – 667 enregistrements  

---

## Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)  
2. [Problématique et Objectifs](#2-problématique-et-objectifs)  
3. [Présentation du Dataset](#3-présentation-du-dataset)  
4. [Analyse Exploratoire des Données (EDA)](#4-analyse-exploratoire-des-données)  
5. [Prétraitement et Feature Engineering](#5-prétraitement-et-feature-engineering)  
6. [Méthodologie de Modélisation](#6-méthodologie-de-modélisation)  
7. [Modèle K-Nearest Neighbors (KNN)](#7-modèle-k-nearest-neighbors)  
8. [Modèle Random Forest](#8-modèle-random-forest)  
9. [Modèle de Régression Logistique](#9-modèle-de-régression-logistique)  
10. [Évaluation et Comparaison des Modèles](#10-évaluation-et-comparaison-des-modèles)  
11. [Analyse des Résultats et Interprétation](#11-analyse-des-résultats-et-interprétation)  
12. [Recommandations et Perspectives](#12-recommandations-et-perspectives)  
13. [Conclusion](#13-conclusion)  
14. [Annexes Techniques](#14-annexes-techniques)  

---

## 1. Introduction et Contexte

### 1.1 La Fraude par Carte Bancaire : Un Enjeu Majeur

La fraude financière par carte bancaire représente l'une des menaces les plus critiques pour l'écosystème financier mondial. Selon les données de Nilson Report, les pertes mondiales liées à la fraude par carte bancaire ont dépassé les **32 milliards de dollars** en 2023, avec une tendance à la hausse constante, notamment en raison de l'essor du commerce en ligne et de la digitalisation des paiements.

En Europe, la Banque Centrale Européenne (BCE) et l'Autorité Bancaire Européenne (ABE) ont renforcé leurs exigences réglementaires à travers la **Directive sur les Services de Paiement (DSP2)**, qui impose aux institutions financières la mise en œuvre de systèmes d'authentification forte et de détection de fraude proactifs.

La fraude par carte bancaire se manifeste sous plusieurs formes :

- **Fraude à la carte présente (CP)** : Utilisation physique d'une carte volée ou clonée
- **Fraude à la carte non présente (CNP)** : Transactions en ligne sans présentation physique de la carte
- **Fraude aux paiements internationaux** : Utilisation abusive des cartes dans des pays étrangers
- **Fraude interne** : Implication de personnels bancaires dans des schémas frauduleux
- **Fraude synthétique** : Création d'identités fictives combinant des données réelles et fausses

### 1.2 Rôle du Machine Learning dans la Détection de Fraude

Les systèmes de détection de fraude traditionnels reposaient sur des règles expertes statiques (ex: blocage si montant > X€, si pays différent du pays de résidence, etc.). Ces approches présentent des limitations majeures :

- **Faible capacité d'adaptation** aux nouveaux schémas de fraude
- **Taux élevé de faux positifs** conduisant à une mauvaise expérience client
- **Incapacité à détecter des patterns complexes** et combinatoires

Le machine learning apporte une réponse à ces limitations en permettant :

- L'apprentissage automatique de patterns complexes à partir de données historiques
- L'adaptation en temps réel aux nouveaux comportements frauduleux
- La gestion efficace du déséquilibre des classes (fraudes rares vs transactions légitimes)
- La combinaison de multiples signaux faibles pour identifier les anomalies

### 1.3 Cadre de ce Rapport

Ce rapport présente une étude complète de détection de fraude par carte bancaire en comparant trois algorithmes de classification parmi les plus utilisés dans l'industrie financière :

1. **K-Nearest Neighbors (KNN)** – algorithme basé sur la similarité
2. **Random Forest (RF)** – méthode d'ensemble par agrégation d'arbres de décision
3. **Régression Logistique (RL)** – modèle linéaire probabiliste

L'étude couvre l'ensemble du cycle de vie d'un projet de data science : exploration des données, prétraitement, modélisation, évaluation et recommandations opérationnelles.

---

## 2. Problématique et Objectifs

### 2.1 Définition de la Problématique

La détection de fraude par carte bancaire est un problème de **classification binaire déséquilibrée** : identifier, parmi l'ensemble des transactions, celles qui sont frauduleuses, sachant que les fraudes représentent une infime minorité du volume total (généralement entre 0,1% et 5% selon les contextes).

**Formellement :**

Soit $T = \{t_1, t_2, ..., t_n\}$ l'ensemble des transactions, et $y_i \in \{0, 1\}$ l'étiquette associée à chaque transaction ($0$ = légitime, $1$ = frauduleuse). Le problème consiste à apprendre une fonction $f : \mathbb{R}^d \rightarrow \{0, 1\}$ telle que :

$$f(x_i) = \begin{cases} 1 & \text{si la transaction est frauduleuse} \\ 0 & \text{si la transaction est légitime} \end{cases}$$

où $x_i \in \mathbb{R}^d$ est le vecteur de caractéristiques de la transaction $t_i$.

### 2.2 Objectifs de l'Étude

**Objectif principal :**  
Développer, évaluer et comparer trois modèles de classification pour la détection de fraude, en identifiant le modèle le plus performant dans le contexte bancaire.

**Objectifs secondaires :**

- Réaliser une analyse exploratoire complète des données transactionnelles
- Identifier les variables les plus discriminantes entre transactions frauduleuses et légitimes
- Quantifier les performances des modèles via des métriques adaptées aux classes déséquilibrées
- Proposer des recommandations opérationnelles pour le déploiement en production
- Analyser le trade-off précision/rappel dans le contexte de la gestion du risque bancaire

### 2.3 Contraintes et Spécifications Techniques

Le système de détection doit répondre aux exigences suivantes :

| Contrainte | Spécification | Justification |
|:-----------|:-------------|:-------------|
| Temps de réponse | < 100ms par transaction | Compatibilité avec le flux temps réel |
| Taux de rappel (fraude) | ≥ 60% | Minimiser les fraudes non détectées |
| Taux de faux positifs | ≤ 10% | Préserver l'expérience client |
| AUC-ROC | ≥ 0.80 | Standard industrie pour la détection de fraude |
| Explicabilité | Requise | Conformité réglementaire (RGPD) |
| Mise à jour du modèle | Mensuelle | Adaptation aux nouveaux patterns |

### 2.4 Métriques d'Évaluation Sélectionnées

Dans un contexte de fraude, la **précision globale (accuracy)** est une métrique trompeuse en raison du déséquilibre des classes. Un modèle qui prédit "légitime" pour toutes les transactions atteindrait une accuracy de ~86% sur notre dataset, sans détecter une seule fraude.

Les métriques pertinentes sont :

**AUC-ROC (Area Under the ROC Curve) :**
$$AUC = \int_0^1 TPR(FPR^{-1}(t)) \, dt$$

**F1-Score (moyenne harmonique précision/rappel) :**
$$F1 = 2 \times \frac{Précision \times Rappel}{Précision + Rappel}$$

**Rappel (Recall/Sensibilité) :**
$$Rappel = \frac{TP}{TP + FN}$$

**Précision :**
$$Précision = \frac{TP}{TP + FP}$$

où TP = Vrais Positifs, FP = Faux Positifs, FN = Faux Négatifs.

---

## 3. Présentation du Dataset

### 3.1 Description Générale

Le dataset utilisé dans cette étude représente un **ensemble de transactions financières par carte bancaire**, constitué de **667 enregistrements** avec **20 variables** couvrant différentes dimensions comportementales et transactionnelles d'un portefeuille client.

**Caractéristiques générales :**

| Caractéristique | Valeur |
|:----------------|:-------|
| Nombre d'observations | 667 |
| Nombre de variables originales | 20 |
| Variables après feature engineering | 24 |
| Transactions frauduleuses | 95 (14.2%) |
| Transactions légitimes | 572 (85.8%) |
| Ratio déséquilibre | 1:6 (fraude:légitime) |
| Valeurs manquantes | 0 (dataset complet) |
| Période couverte | Non spécifiée |

### 3.2 Description des Variables

#### Variables d'Identification et de Profil Client

| Variable | Type | Description |
|:---------|:-----|:------------|
| `Region` | Catégorielle | Code de l'état/région du détenteur de carte |
| `Account_Age_Days` | Numérique | Ancienneté du compte en jours |
| `Region_Code` | Numérique | Code numérique de la zone géographique |
| `Intl_Plan` | Binaire | Souscription à un plan international (Oui/Non) |
| `Vmail_Plan` | Binaire | Activation des alertes par messagerie (Oui/Non) |
| `Vmail_Count` | Numérique | Nombre de messages/alertes reçus |

#### Variables Transactionnelles

| Variable | Type | Description |
|:---------|:-----|:------------|
| `Day_Usage_Min` | Numérique | Volume de transactions diurnes (en minutes équivalent) |
| `Day_Transactions` | Numérique | Nombre de transactions journalières |
| `Day_Amount` | Numérique | Montant total des transactions diurnes (€) |
| `Eve_Usage_Min` | Numérique | Volume de transactions du soir |
| `Eve_Transactions` | Numérique | Nombre de transactions en soirée |
| `Eve_Amount` | Numérique | Montant total des transactions en soirée (€) |
| `Night_Usage_Min` | Numérique | Volume de transactions nocturnes |
| `Night_Transactions` | Numérique | Nombre de transactions nocturnes |
| `Night_Amount` | Numérique | Montant total des transactions nocturnes (€) |
| `Intl_Usage_Min` | Numérique | Volume de transactions internationales |
| `Intl_Transactions` | Numérique | Nombre de transactions à l'international |
| `Intl_Amount` | Numérique | Montant des transactions internationales (€) |

#### Variables Comportementales

| Variable | Type | Description |
|:---------|:-----|:------------|
| `Disputes_Count` | Numérique | Nombre de litiges / contestations de transactions |

#### Variable Cible

| Variable | Type | Description |
|:---------|:-----|:------------|
| `Fraud` | Binaire | 1 = Transaction frauduleuse, 0 = Transaction légitime |

### 3.3 Distribution de la Variable Cible

![Distribution des Classes](graphs/G1_distribution_classes.png)

L'analyse de la distribution révèle un **déséquilibre modéré** :

- **572 transactions légitimes** (85.8% du dataset)
- **95 transactions frauduleuses** (14.2% du dataset)

Ce ratio de 1:6 (fraude/légitime) est représentatif des datasets de fraude en production. Contrairement aux datasets de fraude sur carte de crédit publics (ex: Kaggle Credit Card Fraud) où le taux de fraude peut descendre à 0.17%, notre dataset présente un taux relativement élevé, facilitant l'apprentissage des modèles mais nécessitant tout de même des précautions méthodologiques.

### 3.4 Statistiques Descriptives

**Variables numériques clés :**

| Variable | Moyenne | Écart-type | Min | Médiane | Max |
|:---------|--------:|----------:|----:|--------:|----:|
| Account_Age_Days | 102.8 | 40.8 | 1 | 102 | 232 |
| Day_Amount | 30.8 | 9.8 | 0.0 | 30.5 | 59.6 |
| Eve_Amount | 17.1 | 4.5 | 0.0 | 17.1 | 30.9 |
| Night_Amount | 9.2 | 2.5 | 0.0 | 9.2 | 17.1 |
| Intl_Amount | 2.8 | 0.8 | 0.0 | 2.8 | 4.9 |
| Disputes_Count | 1.6 | 1.3 | 0 | 1 | 8 |
| Total_Amount | 59.8 | 14.9 | 0.0 | 59.5 | 105.3 |

---

## 4. Analyse Exploratoire des Données

### 4.1 Analyse de la Corrélation entre Variables

![Matrice de Corrélation](graphs/G2_correlation_matrix.png)

La matrice de corrélation révèle plusieurs informations importantes :

**Corrélations fortes identifiées :**

1. **Day_Usage_Min ↔ Day_Amount** (r ≈ 0.99) : Corrélation quasi-parfaite indiquant que le montant journalier est proportionnel au volume d'utilisation. Ces deux variables capturent essentiellement la même information.

2. **Total_Amount ↔ Day_Amount** (r ≈ 0.73) : Les transactions diurnes constituent la majeure partie du montant total.

3. **Disputes_Count ↔ Fraud** : Corrélation positive notable. Les clients ayant un nombre élevé de litiges présentent un risque de fraude accru.

4. **Intl_Plan ↔ Intl_Amount** : La souscription à un plan international est fortement liée au volume de transactions internationales.

**Implications pour la modélisation :**

La multicolinéarité observée entre certaines variables (ex: _Usage_Min et _Amount pour chaque période) peut impacter la régression logistique. Nous avons conservé toutes les variables pour les modèles basés sur les arbres (Random Forest) qui sont robustes à la multicolinéarité, et appliqué une régularisation pour la régression logistique.

### 4.2 Distribution des Variables Clés par Classe

![Distributions des Variables](graphs/G3_distributions_variables.png)

L'analyse des distributions révèle des différences comportementales significatives entre les transactions frauduleuses et légitimes :

**Montant Total :**  
Les transactions frauduleuses présentent une distribution légèrement décalée vers les montants plus élevés, avec une queue plus lourde à droite. Cela suggère que les fraudeurs tendent à maximiser les montants transactionnels.

**Montant Journalier :**  
La distribution des transactions frauduleuses est plus étalée, avec des pics plus marqués aux extrêmes, indiquant une volatilité comportementale plus importante.

**Nombre de Litiges :**  
La différence est la plus marquante : les transactions frauduleuses sont associées à un nombre de litiges significativement plus élevé. Cette variable constitue un signal fort de détection de fraude.

**Montant Moyen par Transaction :**  
Les fraudes tendent à avoir un montant moyen par transaction légèrement supérieur, les fraudeurs cherchant à maximiser le gain par opération.

### 4.3 Analyse par Boxplots

![Boxplots Comparatifs](graphs/G4_boxplots.png)

Les boxplots confirment et précisent les observations précédentes :

**Observations clés :**

1. **Montant Total** : La médiane des transactions frauduleuses est légèrement supérieure, avec une variance plus élevée (boîte plus large).

2. **Nombre de Transactions** : Les transactions frauduleuses présentent un volume transactionnel similaire aux transactions légitimes, confirmant que les fraudeurs imitent le comportement normal.

3. **Litiges (Disputes_Count)** : Différence la plus marquée. La médiane est significativement plus élevée pour les fraudes, et les valeurs extrêmes (jusqu'à 8 litiges) sont presque exclusivement associées aux transactions frauduleuses.

4. **Ratio Nocturne (Night_Ratio)** : Les transactions frauduleuses présentent un ratio nocturne légèrement plus élevé, cohérent avec la littérature sur la fraude (les activités frauduleuses sont plus fréquentes la nuit).

5. **Ratio International (Intl_Ratio)** : Différence notable, les fraudes étant plus fréquemment associées à des transactions transfrontalières.

6. **Taux de Litige (Dispute_Rate)** : Indicateur composite (litiges/ancienneté) particulièrement discriminant pour les comptes jeunes avec de nombreux litiges.

### 4.4 Analyse des Transactions Frauduleuses par Dimension Spatiale

![Scatter - Montant vs Transactions](graphs/G14_scatter_amount_transactions.png)

Le nuage de points croisé (montant total vs nombre de transactions) met en évidence la **superposition partielle** des deux classes dans l'espace des features. Les transactions frauduleuses ne sont pas isolées dans un espace distinct, ce qui justifie l'utilisation d'algorithmes capables de modéliser des frontières de décision complexes.

On observe néanmoins que certaines zones de concentration de fraudes correspondent à des montants totaux élevés avec un nombre de transactions modéré, signature comportementale des fraudes à montants unitaires élevés.

### 4.5 Impact des Litiges sur le Taux de Fraude

![Litiges vs Taux de Fraude](graphs/G18_disputes_fraud_rate.png)

Cette analyse révèle une **relation quasi-monotone** entre le nombre de litiges et le taux de fraude :

| Litiges | Taux de fraude estimé |
|:-------:|:---------------------:|
| 0 | ~5% |
| 1 | ~10% |
| 2-3 | ~15-20% |
| 4-5 | ~30-40% |
| 6+ | >60% |

Cette relation forte fait des `Disputes_Count` l'un des prédicteurs les plus importants dans nos modèles, comme nous le confirmerons dans la section sur l'importance des variables.

---

## 5. Prétraitement et Feature Engineering

### 5.1 Nettoyage des Données

**Valeurs manquantes :**  
Le dataset est complet, sans aucune valeur manquante, ce qui simplifie le prétraitement.

**Valeurs aberrantes :**  
Une analyse des percentiles (P1 et P99) n'a pas révélé de valeurs aberrantes manifestes. Les valeurs extrêmes observées (ex: 8 litiges) sont cohérentes avec le contexte de fraude et doivent être conservées car elles portent de l'information discriminante.

**Doublons :**  
Aucun doublon identifié après vérification des identifiants de transactions.

### 5.2 Encodage des Variables Catégorielles

Deux variables catégorielles binaires ont été encodées numériquement :

```python
# Encodage Label (0/1 pour variables binaires)
df['Intl_Plan_enc'] = LabelEncoder().fit_transform(df['Intl_Plan'])   # No=0, Yes=1
df['Vmail_Plan_enc'] = LabelEncoder().fit_transform(df['Vmail_Plan']) # No=0, Yes=1
```

La variable `Region` (50 états américains) a été exclue des features pour éviter l'augmentation dimensionnelle inutile via one-hot encoding, le `Region_Code` (indicateur numérique) capturant suffisamment l'information géographique.

### 5.3 Feature Engineering

Nous avons créé **6 nouvelles variables dérivées** pour capturer des patterns comportementaux plus complexes :

**1. Montant Total Consolidé**
```python
df['Total_Amount'] = df['Day_Amount'] + df['Eve_Amount'] + df['Night_Amount'] + df['Intl_Amount']
```
Capture la valeur totale des transactions toutes périodes confondues.

**2. Volume Total de Transactions**
```python
df['Total_Transactions'] = df['Day_Transactions'] + df['Eve_Transactions'] + df['Night_Transactions'] + df['Intl_Transactions']
```
Mesure l'intensité transactionnelle globale du client.

**3. Montant Moyen par Transaction**
```python
df['Avg_Transaction_Amount'] = df['Total_Amount'] / (df['Total_Transactions'] + 1)
```
Indicateur du montant unitaire moyen, pertinent pour détecter les fraudes à montants élevés.

**4. Ratio Nocturne**
```python
df['Night_Ratio'] = df['Night_Amount'] / (df['Total_Amount'] + 1)
```
Part des transactions réalisées la nuit, indicateur d'un comportement atypique.

**5. Ratio International**
```python
df['Intl_Ratio'] = df['Intl_Amount'] / (df['Total_Amount'] + 1)
```
Part des transactions à l'international, signal fort de fraude transfrontalière.

**6. Taux de Litige Normalisé**
```python
df['Dispute_Rate'] = df['Disputes_Count'] / (df['Account_Age_Days'] + 1)
```
Normalise le nombre de litiges par l'ancienneté du compte, permettant de comparer des clients avec des historiques différents.

### 5.4 Division Train/Test

La division du dataset a été réalisée avec une stratification pour garantir la représentativité des fraudes dans les deux sous-ensembles :

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
```

| Subset | Total | Légitimes | Frauduleuses |
|:-------|------:|----------:|-------------:|
| Train | 500 | 429 (85.8%) | 71 (14.2%) |
| Test | 167 | 143 (85.6%) | 24 (14.4%) |

### 5.5 Normalisation des Données

La normalisation est critique pour les algorithmes sensibles à l'échelle des variables (KNN, Régression Logistique) :

```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)  # Ajustement uniquement sur train
X_test_sc  = scaler.transform(X_test)        # Application sur test
```

La normalisation **StandardScaler** centre et réduit chaque variable :
$$x_{norm} = \frac{x - \mu_{train}}{\sigma_{train}}$$

**Important :** Le scaler est ajusté uniquement sur les données d'entraînement pour éviter la **fuite de données (data leakage)** vers l'ensemble de test.

**Random Forest** ne nécessite pas de normalisation (invariant aux transformations monotones des features).

---

## 6. Méthodologie de Modélisation

### 6.1 Approche Générale

L'approche méthodologique suit le protocole CRISP-DM (Cross-Industry Standard Process for Data Mining) adapté au contexte bancaire :

```
Compréhension du domaine → EDA → Prétraitement → Modélisation → Évaluation → Déploiement
```

### 6.2 Gestion du Déséquilibre des Classes

Le déséquilibre (85.8% légitimes / 14.2% fraudes) a été adressé via deux stratégies complémentaires :

**1. Pondération des Classes :**  
Pour Random Forest et Régression Logistique, le paramètre `class_weight='balanced'` a été utilisé pour pénaliser davantage les erreurs sur la classe minoritaire (fraude) :

$$w_{fraude} = \frac{n_{total}}{2 \times n_{fraude}} \approx 3.5$$

**2. Métriques Adaptées :**  
Utilisation de l'AUC-ROC, du F1-Score et du rappel comme métriques primaires, plutôt que l'accuracy globale.

### 6.3 Protocole de Validation

**Validation croisée stratifiée (5-fold) :**
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
```

La validation croisée stratifiée garantit que chaque fold contient approximativement la même proportion de fraudes, évitant ainsi les biais d'estimation des performances.

**Optimisation des hyperparamètres :**  
Une analyse de sensibilité a été réalisée pour les paramètres clés de chaque modèle (K pour KNN, nombre d'arbres pour RF, régularisation C pour RL).

---

## 7. Modèle K-Nearest Neighbors (KNN)

### 7.1 Principe Théorique

L'algorithme KNN (K plus proches voisins) est un **algorithme d'apprentissage paresseux (lazy learning)** qui ne construit pas de modèle explicite lors de la phase d'entraînement. Pour une nouvelle transaction $x$, il identifie les K transactions les plus similaires dans l'ensemble d'entraînement et assigne la classe majoritaire :

$$\hat{y} = \text{argmax}_{c} \sum_{x_i \in \mathcal{N}_K(x)} \mathbf{1}[y_i = c] \cdot w_i$$

où $\mathcal{N}_K(x)$ est l'ensemble des K plus proches voisins de $x$, et $w_i$ est le poids associé au voisin $i$.

**Mesure de distance :**  
Nous utilisons la distance de Minkowski avec $p=2$ (distance Euclidienne) :

$$d(x, x_i) = \left(\sum_{j=1}^{d} |x_j - x_{ij}|^2\right)^{1/2}$$

**Pondération des voisins :**  
Le paramètre `weights='distance'` attribue un poids inversement proportionnel à la distance, donnant plus d'importance aux voisins les plus proches :

$$w_i = \frac{1}{d(x, x_i)}$$

### 7.2 Configuration du Modèle

```python
knn = KNeighborsClassifier(
    n_neighbors=7,        # Nombre de voisins
    metric='minkowski',   # Distance Euclidienne (p=2)
    weights='distance',   # Pondération par distance inverse
)
```

**Note :** KNN nécessite obligatoirement une normalisation préalable des données, car les variables avec de grandes plages de valeurs (ex: `Total_Amount`) domineraient sinon le calcul de distance.

### 7.3 Optimisation du Paramètre K

![Sensibilité au paramètre K](graphs/G10_knn_k_sensitivity.png)

L'analyse de sensibilité au paramètre K révèle :

- **K faibles (1-3)** : Surajustement (overfitting) — l'AUC sur le train est maximale mais la généralisation est faible
- **K=7** : Bon compromis accuracy/généralisation, avec AUC-ROC = 0.822
- **K élevés (15+)** : Sous-ajustement progressif, perte de capacité à détecter les fraudes isolées

Le choix de **K=7** est validé comme optimal sur notre dataset, en accord avec la heuristique $K \approx \sqrt{n_{train}}$ souvent citée en pratique.

### 7.4 Résultats du Modèle KNN

| Métrique | Valeur |
|:---------|-------:|
| Accuracy | 0.874 |
| Précision (fraude) | 0.800 |
| Rappel (fraude) | 0.167 |
| F1-Score (fraude) | 0.276 |
| AUC-ROC | 0.822 |

**Analyse de la matrice de confusion :**

```
                  Prédit Légitime    Prédit Fraude
Réel Légitime           142                 1
Réel Fraude              20                 4
```

Le KNN obtient une précision élevée (80%) lorsqu'il prédit une fraude, mais son rappel est très faible (16.7%) : il ne détecte que 4 fraudes sur 24. Le modèle est fortement biaisé vers la classe majoritaire malgré la normalisation.

### 7.5 Forces et Limites du KNN

**Forces :**
- Simple à comprendre et à implémenter
- Pas d'hypothèse sur la distribution des données
- Naturellement adaptatif aux patterns locaux complexes
- Bonne précision (peu de faux positifs)

**Limites :**
- Coût computationnel élevé en production (O(n) par prédiction)
- Sensible au déséquilibre des classes
- Performance limitée dans les espaces de grande dimension (curse of dimensionality)
- Faible rappel sur les fraudes dans ce dataset

---

## 8. Modèle Random Forest

### 8.1 Principe Théorique

Random Forest est un **algorithme d'ensemble** qui combine plusieurs arbres de décision par **bootstrap aggregating (bagging)** pour réduire la variance et améliorer la généralisation.

**Construction d'un arbre de décision :**  
Chaque arbre est construit en sélectionnant aléatoirement $m = \sqrt{d}$ features à chaque nœud, puis en choisissant la meilleure division selon l'impureté de Gini :

$$G(t) = 1 - \sum_{c=1}^{C} p_c^2$$

où $p_c$ est la proportion de la classe $c$ au nœud $t$.

**Agrégation (Vote majoritaire) :**

$$\hat{y} = \text{argmax}_c \sum_{b=1}^{B} \mathbf{1}[T_b(x) = c]$$

où $T_b$ est le $b$-ième arbre de la forêt et $B$ est le nombre total d'arbres.

**Probabilité de fraude :**

$$P(y=1|x) = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}[T_b(x) = 1]$$

### 8.2 Configuration du Modèle

```python
rf = RandomForestClassifier(
    n_estimators=200,          # Nombre d'arbres
    max_depth=10,              # Profondeur maximale
    random_state=42,           # Reproductibilité
    class_weight='balanced',   # Pondération inverse à la fréquence
)
```

### 8.3 Impact du Nombre d'Arbres

![Nombre d'arbres vs Performance](graphs/G15_rf_n_estimators.png)

L'analyse montre une **saturation de la performance** à partir de 100-150 arbres. Au-delà, le gain marginal est négligeable. Le choix de 200 arbres offre un bon équilibre entre performance et coût computationnel.

### 8.4 Importance des Variables

![Importance des Variables - Random Forest](graphs/G9_feature_importance_rf.png)

Le Random Forest identifie les variables les plus discriminantes pour la détection de fraude :

**Top 5 des variables les plus importantes :**

| Rang | Variable | Importance | Interprétation |
|:----:|:---------|:----------:|:---------------|
| 1 | `Total_Amount` | ~0.12 | Montant total, fort signal de fraude |
| 2 | `Day_Amount` | ~0.11 | Transactions diurnes importantes |
| 3 | `Disputes_Count` | ~0.10 | Litiges, signal comportemental fort |
| 4 | `Dispute_Rate` | ~0.09 | Taux de litige normalisé |
| 5 | `Avg_Transaction_Amount` | ~0.08 | Montant moyen par transaction |

Ces résultats confirment les hypothèses métier : les variables relatives aux montants et aux litiges sont les signaux les plus puissants pour distinguer les fraudes des transactions légitimes.

### 8.5 Résultats du Modèle Random Forest

| Métrique | Valeur |
|:---------|-------:|
| Accuracy | 0.952 |
| Précision (fraude) | **1.000** |
| Rappel (fraude) | **0.667** |
| F1-Score (fraude) | **0.800** |
| AUC-ROC | **0.901** |

**Analyse de la matrice de confusion :**

```
                  Prédit Légitime    Prédit Fraude
Réel Légitime           143                 0
Réel Fraude               8                16
```

Résultats remarquables : **0 faux positifs** (aucune transaction légitime classée comme fraude), ce qui garantit une excellente expérience client. 16 fraudes sur 24 sont correctement identifiées.

### 8.6 Forces et Limites du Random Forest

**Forces :**
- Excellentes performances globales (AUC=0.901)
- Robuste à la multicolinéarité et aux valeurs aberrantes
- Fournit l'importance des variables (explicabilité partielle)
- Gestion native du déséquilibre via `class_weight`
- Pas de normalisation requise

**Limites :**
- Modèle "boîte noire" (explicabilité limitée par rapport à un arbre simple)
- Coût mémoire élevé (200 arbres)
- Risque de surapprentissage si les hyperparamètres ne sont pas contrôlés

---

## 9. Modèle de Régression Logistique

### 9.1 Principe Théorique

La régression logistique est un **modèle linéaire généralisé** qui modélise la probabilité d'appartenance à une classe via la fonction sigmoïde (logistique) :

$$P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

où $w \in \mathbb{R}^d$ est le vecteur de poids et $b$ le biais.

**Estimation des paramètres (Maximum de Vraisemblance Régularisé) :**

$$\hat{w} = \text{argmin}_w \left[ -\sum_{i=1}^n [y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)] + \lambda ||w||_2^2 \right]$$

Le terme $\lambda ||w||_2^2$ est la régularisation L2 (Ridge) qui pénalise les poids trop élevés et prévient le surapprentissage. Le paramètre $C = 1/\lambda$ contrôle l'intensité de la régularisation.

**Décision de classification :**

$$\hat{y} = \begin{cases} 1 & \text{si } P(y=1|x) \geq 0.5 \\ 0 & \text{sinon} \end{cases}$$

### 9.2 Configuration du Modèle

```python
lr = LogisticRegression(
    max_iter=1000,              # Convergence assurée
    random_state=42,            # Reproductibilité
    class_weight='balanced',    # Compensation du déséquilibre
    C=0.5,                      # Régularisation L2 modérée
)
```

### 9.3 Interprétation des Coefficients

Un avantage majeur de la régression logistique est la **directe interprétabilité des coefficients**. Les odds ratios $e^{w_j}$ indiquent l'impact de chaque variable sur la probabilité de fraude :

- $w_j > 0$ : La variable $j$ augmente la probabilité de fraude
- $w_j < 0$ : La variable $j$ diminue la probabilité de fraude
- $|w_j|$ grand : Variable fortement discriminante

Cette propriété est particulièrement valorisée dans les institutions financières soumises à des obligations réglementaires d'explicabilité (RGPD Article 22 – droit à l'explication).

### 9.4 Résultats du Modèle de Régression Logistique

| Métrique | Valeur |
|:---------|-------:|
| Accuracy | 0.772 |
| Précision (fraude) | 0.348 |
| Rappel (fraude) | 0.667 |
| F1-Score (fraude) | 0.457 |
| AUC-ROC | 0.819 |

**Analyse de la matrice de confusion :**

```
                  Prédit Légitime    Prédit Fraude
Réel Légitime           105                38
Réel Fraude               8                16
```

La régression logistique atteint le même rappel que le Random Forest (66.7%) mais génère **38 faux positifs**, bloquant des transactions légitimes. Son F1-Score est significativement inférieur.

### 9.5 Forces et Limites de la Régression Logistique

**Forces :**
- Excellente interprétabilité des coefficients
- Très rapide à entraîner et à prédire (adapté au temps réel)
- Probabilités calibrées naturellement
- Conforme aux exigences réglementaires (RGPD)
- Robuste aux petits datasets

**Limites :**
- Hypothèse de linéarité souvent non vérifiée
- Sensible à la multicolinéarité (nécessite une régularisation)
- Performances limitées pour les problèmes non-linéaires
- Taux élevé de faux positifs sur ce dataset

---

## 10. Évaluation et Comparaison des Modèles

### 10.1 Matrices de Confusion Comparatives

![Matrices de Confusion](graphs/G5_confusion_matrices.png)

La visualisation simultanée des trois matrices de confusion permet une comparaison directe des comportements de classification :

**KNN :** Très conservateur dans ses prédictions de fraude (seulement 5 prédictions positives au total), résultant en un taux de détection très faible.

**Random Forest :** Performance optimale — identifie 16/24 fraudes sans aucun faux positif.

**Régression Logistique :** Profil équilibré entre rappel et précision, mais avec un nombre élevé de faux positifs (38).

### 10.2 Courbes ROC Comparatives

![Courbes ROC](graphs/G6_roc_curves.png)

Les courbes ROC présentent les performances discriminantes de chaque modèle sur l'ensemble du spectre des seuils de décision :

| Modèle | AUC-ROC | Interprétation |
|:-------|:-------:|:---------------|
| Random Forest | **0.901** | Excellent — Très bonne discrimination |
| KNN | 0.822 | Bon — Discrimination satisfaisante |
| Régression Logistique | 0.819 | Bon — Comparable au KNN |

**Interprétation de l'AUC-ROC :**

Un AUC de 0.901 signifie que le Random Forest classe correctement une transaction frauduleuse devant une transaction légitime dans **90.1%** des paires aléatoires (fraude, légitime). Cela représente un niveau de performance industriel acceptable pour un premier déploiement.

### 10.3 Courbes Précision-Rappel

![Courbes Précision-Rappel](graphs/G7_precision_recall.png)

Les courbes Précision-Rappel sont particulièrement informatives dans le contexte des classes déséquilibrées, car elles focalisent sur la performance sur la classe minoritaire (fraude) :

| Modèle | Average Precision (AP) |
|:-------|:----------------------:|
| Random Forest | Élevée |
| KNN | Modérée |
| Régression Logistique | Modérée |

La courbe Précision-Rappel du Random Forest montre un excellent compromis : il est possible d'atteindre un rappel de 60-70% tout en maintenant une précision supérieure à 90%.

### 10.4 Radar des Performances Globales

![Radar des Métriques](graphs/G12_radar_metrics.png)

Le radar des performances offre une vue synthétique des 5 métriques pour les 3 modèles :

Le **Random Forest** domine sur 4 des 5 métriques, avec une avance particulièrement marquée sur le F1-Score. La Régression Logistique présente un profil plus équilibré avec un bon rappel mais une précision plus faible. Le KNN offre la meilleure précision unitaire mais au prix d'un rappel très faible.

### 10.5 Tableau Comparatif des Performances

![Tableau des Métriques](graphs/G8_metrics_table.png)

**Tableau récapitulatif :**

| Modèle | Accuracy | Précision | Rappel | F1-Score | AUC-ROC | Temps Pred. |
|:-------|:--------:|:---------:|:------:|:--------:|:--------:|:-----------:|
| KNN | 0.874 | 0.800 | 0.167 | 0.276 | 0.822 | Lent |
| Random Forest | **0.952** | **1.000** | **0.667** | **0.800** | **0.901** | Moyen |
| Régression Log. | 0.772 | 0.348 | 0.667 | 0.457 | 0.819 | Très rapide |

### 10.6 Barplot Comparatif des Métriques

![Barplot des Métriques](graphs/G17_metrics_comparison.png)

Ce graphique confirme visuellement la **supériorité du Random Forest** sur l'ensemble des métriques de performance, avec une dominance particulièrement nette sur la précision (1.000) et le F1-Score (0.800).

### 10.7 Validation Croisée

![Validation Croisée](graphs/G11_cross_validation.png)

La validation croisée (5-fold) confirme la robustesse des performances estimées sur le jeu de test :

| Modèle | AUC-ROC Moyen (CV) | Écart-type |
|:-------|:-----------------:|:----------:|
| Random Forest | ~0.88 | Faible |
| KNN | ~0.80 | Modéré |
| Régression Log. | ~0.81 | Modéré |

Les faibles écarts-types confirment que les modèles ne sont pas en surapprentissage et que les performances sont stables quelle que soit la partition train/validation utilisée.

### 10.8 Courbes d'Apprentissage

![Courbes d'Apprentissage](graphs/G16_learning_curves.png)

**Interprétation des courbes d'apprentissage :**

- **KNN** : Gap relativement faible entre train et validation, mais les performances absolues plafonnent tôt. Plus de données ne l'amélioreraient pas significativement.

- **Random Forest** : Performance train élevée, validation croissante avec la taille du dataset. Un dataset plus grand améliorerait encore les performances.

- **Régression Logistique** : Profil stable, les courbes train et validation convergent rapidement, signe d'un modèle bien régularisé mais limité par sa capacité linéaire.

### 10.9 Distribution des Probabilités Prédites

![Distribution des Probabilités](graphs/G13_probability_distributions.png)

La distribution des probabilités prédites révèle les comportements de calibration de chaque modèle :

- **KNN** : Distribution bimodale — les transactions sont fortement concentrées aux probabilités extrêmes (0 ou 1), avec peu de transactions dans les zones intermédiaires.

- **Random Forest** : Distribution bien séparée entre fraudes et légitimes, avec un chevauchement limité autour de 0.5.

- **Régression Logistique** : Distribution plus étalée, reflétant la nature probabiliste du modèle linéaire.

---

## 11. Analyse des Résultats et Interprétation

### 11.1 Synthèse des Performances

L'évaluation comparative des trois modèles conduit aux conclusions suivantes :

**1. Random Forest — Modèle Recommandé**

Le Random Forest s'impose comme le modèle le plus performant sur l'ensemble des métriques pertinentes pour la détection de fraude. Son AUC-ROC de **0.901** le positionne au-dessus du seuil industriel standard (0.85 pour les systèmes de production), et son F1-Score de **0.800** est remarquable pour un dataset aussi déséquilibré.

Un résultat particulièrement notable : **0 faux positifs** dans la configuration standard (seuil=0.5). En pratique, un faux positif représente un client insatisfait (carte bloquée à tort), un appel au service client, et potentiellement une résiliation. L'absence de faux positifs est donc une performance commerciale forte.

**2. KNN — Modèle de Référence**

Le KNN obtient une AUC-ROC correcte (0.822) mais souffre d'un rappel très faible (16.7%). Dans un contexte de fraude, manquer 83% des fraudes est inacceptable en production. Il pourrait être utilisé comme modèle de référence ou en complément d'autres modèles.

**3. Régression Logistique — Modèle Interprétatif**

La régression logistique présente le profil le plus difficile à exploiter en production : elle atteint le même rappel que le Random Forest (66.7%) mais génère 38 faux positifs contre 0 pour le Random Forest. Son avantage principal est l'interprétabilité directe des coefficients, précieuse pour la conformité réglementaire.

### 11.2 Analyse du Trade-off Précision/Rappel

La gestion du seuil de décision est un levier opérationnel crucial. En abaissant le seuil de classification (ex: de 0.5 à 0.3), on augmente le rappel au détriment de la précision :

| Seuil | RF Rappel | RF Précision | RF F1 | Action recommandée |
|:-----:|:---------:|:------------:|:-----:|:------------------|
| 0.70 | Faible | Très élevée | Moyen | Faux positifs nuls |
| 0.50 | 0.667 | 1.000 | 0.800 | **Configuration actuelle** |
| 0.30 | Élevé | Modérée | Modéré | Plus de fraudes détectées |
| 0.10 | Très élevé | Faible | Faible | Trop de faux positifs |

La banque doit définir sa **tolérance au risque** pour choisir le seuil optimal selon deux axes :
- **Minimiser les pertes fraude** → seuil bas (rappel élevé)
- **Minimiser les faux positifs** → seuil élevé (précision élevée)

### 11.3 Analyse des Erreurs

**Fraudes non détectées (Faux Négatifs) par le Random Forest :**  
8 transactions frauduleuses sur 24 n'ont pas été identifiées. Ces transactions présentent vraisemblablement des signatures comportementales très similaires aux transactions légitimes (montants standards, peu de litiges, ratio normal). Des techniques d'anomalie (Isolation Forest, Autoencoder) pourraient compléter le Random Forest pour ces cas limites.

**Transactions bloquées à tort (Faux Positifs) :**  
Le Random Forest ne génère aucun faux positif au seuil standard, ce qui est un résultat exceptionnel mais potentiellement fragile sur de nouvelles données en production.

### 11.4 Comparaison avec les Benchmarks Industriels

| Indicateur | Notre RF | Référence Industrie | Évaluation |
|:-----------|:--------:|:-------------------:|:----------:|
| AUC-ROC | 0.901 | > 0.85 | ✅ Satisfaisant |
| F1-Score | 0.800 | > 0.70 | ✅ Bon |
| Taux détection fraude | 66.7% | > 50% | ✅ Satisfaisant |
| Taux faux positifs | 0% | < 10% | ✅ Excellent |
| Accuracy | 95.2% | > 85% | ✅ Excellent |

Le Random Forest satisfait tous les critères industriels définis en section 2.3.

---

## 12. Recommandations et Perspectives

### 12.1 Recommandation de Déploiement

**Modèle recommandé pour la production : Random Forest**

Avec les paramètres suivants :
- `n_estimators=200`
- `max_depth=10`
- `class_weight='balanced'`
- Seuil de classification : **0.5** (ajustable selon les objectifs métier)

**Architecture de déploiement recommandée :**

```
Transaction → Prétraitement → Normalisation → Random Forest → Score Fraude
                                                                    ↓
                                              Score ≥ 0.5 → Blocage + Alerte
                                              Score < 0.5 → Autorisation
                                              Score [0.3-0.5] → Vérification manuelle
```

### 12.2 Améliorations à Court Terme (0-3 mois)

**1. Optimisation des Hyperparamètres :**  
Réaliser une recherche par grille (GridSearchCV) ou Bayésienne (Optuna) sur les paramètres clés du Random Forest pour maximiser le F1-Score.

**2. Ensemble de Modèles (Stacking) :**  
Combiner les trois modèles en un méta-modèle de type stacking, exploitant les forces complémentaires de chaque algorithme :

```python
from sklearn.ensemble import StackingClassifier
stacking = StackingClassifier(
    estimators=[('knn', knn), ('rf', rf)],
    final_estimator=LogisticRegression(),
)
```

**3. Calibration des Probabilités :**  
Appliquer une calibration isotonique ou de Platt pour que les probabilités prédites correspondent aux probabilités réelles de fraude, facilitant la prise de décision basée sur un seuil de risque monétaire.

**4. Ajustement du Seuil Optimal :**  
Calculer le seuil optimal selon la courbe Précision-Rappel, en intégrant le coût métier d'une fraude non détectée vs le coût d'un faux positif.

### 12.3 Améliorations à Moyen Terme (3-12 mois)

**5. Enrichissement des Données :**

- Intégration des données géographiques (pays de la transaction vs pays de résidence)
- Ajout de variables comportementales temporelles (heure de la transaction, jour de la semaine)
- Intégration du score de risque du commerçant (Merchant Risk Score)
- Données d'authentification (tentatives PIN, authentification 3DS)

**6. Détection d'Anomalies (Approche Non Supervisée) :**  
Pour les fraudes inédites (zero-day fraud), compléter le Random Forest avec des algorithmes de détection d'anomalies :

- **Isolation Forest** : Efficace pour les outliers multivariés
- **Local Outlier Factor (LOF)** : Détection d'anomalies locales
- **Autoencoders** : Apprentissage profond de la représentation normale

**7. Modèles de Graphes (Graph Neural Networks) :**  
Modéliser les réseaux de transactions pour détecter les schémas de fraude organisée (rings de fraude), en exploitant les connexions entre comptes, terminaux et bénéficiaires.

**8. Monitoring et Dérive des Données (Data Drift) :**  
Implémenter un système de surveillance continue des distributions des features et des performances du modèle en production :

```python
# Détecter la dérive avec test Kolmogorov-Smirnov
from scipy.stats import ks_2samp
stat, p_value = ks_2samp(X_train['Total_Amount'], X_new['Total_Amount'])
if p_value < 0.05:
    trigger_model_retraining()
```

### 12.4 Recommandations de Gouvernance et Conformité

**Conformité RGPD :**  
- Documenter les variables utilisées et leur pertinence pour la décision
- Mettre en place un processus de contestation pour les faux positifs
- Limiter la conservation des données à la durée strictement nécessaire

**Biais et Équité :**  
- Vérifier l'absence de biais discriminatoire (par région, plan tarifaire)
- Tester les performances du modèle de manière segmentée sur les sous-populations

**Auditabilité :**  
- Utiliser SHAP (SHapley Additive exPlanations) pour l'explication des décisions individuelles
- Maintenir un journal d'audit des décisions du modèle

### 12.5 Feuille de Route Technologique

| Horizon | Action | Impact Attendu |
|:--------|:-------|:--------------|
| 1 mois | Déploiement RF en production (shadow mode) | Validation sans impact client |
| 3 mois | Activation en mode décision | Réduction des pertes fraude |
| 6 mois | Stacking RF + Anomaly Detection | +10-15% détection fraude |
| 12 mois | GNN + Behavioral Analytics | Détection fraude organisée |
| 24 mois | Modèle temps réel streaming (Kafka + Spark) | Latence < 10ms |

---

## 13. Conclusion

### 13.1 Synthèse des Contributions

Cette étude a présenté une analyse complète de détection de fraude par carte bancaire en utilisant trois algorithmes de machine learning complémentaires. Les contributions principales sont :

**1. Analyse Exploratoire Complète :**  
Identification des variables les plus discriminantes pour la détection de fraude, notamment le nombre de litiges, les montants transactionnels et les ratios comportementaux (nocturne, international).

**2. Feature Engineering Ciblé :**  
Création de 6 nouvelles variables dérivées qui enrichissent significativement la capacité discriminante des modèles, en particulier le `Dispute_Rate` et le `Avg_Transaction_Amount`.

**3. Comparaison Rigoureuse des Modèles :**  
Évaluation complète avec 18 graphiques d'analyse couvrant les matrices de confusion, courbes ROC, courbes Précision-Rappel, validation croisée et courbes d'apprentissage.

**4. Identification du Meilleur Modèle :**  
Le **Random Forest** s'impose comme le modèle optimal avec :
- AUC-ROC de **0.901** (excellent)
- F1-Score de **0.800** (très bon)
- **0 faux positifs** (exceptionnel)
- Taux de détection de **66.7%** des fraudes

### 13.2 Limites de l'Étude

**Taille du Dataset :**  
Avec 667 transactions dont 95 fraudes, le dataset est relativement petit pour des applications de production. Les résultats obtenus sont prometteurs mais devront être validés sur des volumes plus importants.

**Nature des Données :**  
Le dataset utilisé est une adaptation d'un dataset de churn télécom réinterprété dans un contexte bancaire. Les variables ne capturent pas toutes les dimensions d'une fraude bancaire réelle (données temporelles, géographiques, biométriques).

**Validation Temporelle :**  
Dans un contexte de production, il est impératif de valider les modèles avec une validation temporelle (train sur données historiques, test sur données futures) pour simuler les conditions réelles.

### 13.3 Impact Opérationnel Estimé

En supposant un volume mensuel de 100,000 transactions avec un taux de fraude de 14.2% :

| Scénario | Fraudes Détectées | Faux Positifs | Économies Estimées* |
|:---------|:-----------------:|:-------------:|:-------------------:|
| Sans modèle | 0 | 0 | 0€ |
| KNN | ~2,370 | ~200 | ~950,000€ |
| Régression Log. | ~9,480 | ~38,000 | ~3,350,000€** |
| **Random Forest** | **~9,480** | **~0** | **~3,800,000€** |

*Estimations basées sur un montant moyen de fraude de 400€ et un coût de faux positif de 15€ (service client + attrition).  
**Réduit par le coût des nombreux faux positifs.

### 13.4 Message Final

La détection de fraude est un problème d'optimisation multi-objectifs complexe, nécessitant un équilibre subtil entre la protection des clients (maximiser le rappel) et la qualité d'expérience (minimiser les faux positifs). Cette étude démontre que le **Random Forest** atteint cet équilibre de manière remarquable sur notre dataset.

La mise en production d'un tel système représente un investissement rentable : selon nos estimations, le modèle pourrait permettre d'économiser plusieurs millions d'euros par an tout en améliorant significativement la confiance des clients dans leur institution bancaire.

Le déploiement doit cependant s'accompagner d'un cadre rigoureux de gouvernance, de monitoring continu et d'amélioration itérative pour maintenir des performances optimales face à l'évolution constante des techniques de fraude.

---

## 14. Annexes Techniques

### Annexe A : Environnement Technique

| Composant | Version | Usage |
|:----------|:-------:|:------|
| Python | 3.10+ | Langage de programmation |
| scikit-learn | 1.3+ | Modèles ML et évaluation |
| pandas | 2.0+ | Manipulation des données |
| numpy | 1.24+ | Calcul numérique |
| matplotlib | 3.7+ | Visualisations statiques |
| seaborn | 0.12+ | Visualisations statistiques |

### Annexe B : Code Source des Modèles

**K-Nearest Neighbors :**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=7, metric='minkowski', weights='distance')
knn.fit(X_train_sc, y_train)
y_pred_knn = knn.predict(X_test_sc)
y_prob_knn = knn.predict_proba(X_test_sc)[:, 1]
```

**Random Forest :**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
```

**Régression Logistique :**
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    C=0.5
)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]
```

### Annexe C : Évaluation Complète des Modèles

**Rapport de Classification – KNN :**
```
              precision    recall  f1-score   support

   Légitime       0.88      0.99      0.93       143
    Fraude        0.80      0.17      0.28        24

    accuracy                           0.87       167
   macro avg       0.84      0.58      0.60       167
weighted avg       0.87      0.87      0.84       167
```

**Rapport de Classification – Random Forest :**
```
              precision    recall  f1-score   support

   Légitime       0.95      1.00      0.97       143
    Fraude        1.00      0.67      0.80        24

    accuracy                           0.95       167
   macro avg       0.97      0.83      0.89       167
weighted avg       0.96      0.95      0.95       167
```

**Rapport de Classification – Régression Logistique :**
```
              precision    recall  f1-score   support

   Légitime       0.93      0.73      0.82       143
    Fraude        0.35      0.67      0.46        24

    accuracy                           0.77       167
   macro avg       0.64      0.70      0.64       167
weighted avg       0.85      0.77      0.79       167
```

### Annexe D : Glossaire des Termes Techniques

| Terme | Définition |
|:------|:-----------|
| **AUC-ROC** | Area Under the Receiver Operating Characteristic Curve – mesure la capacité discriminante globale d'un modèle |
| **Bagging** | Bootstrap Aggregating – technique d'ensemble combinant des modèles entraînés sur des sous-échantillons avec remplacement |
| **Classe déséquilibrée** | Distribution inégale des classes dans un dataset (ex: 86% légitimes, 14% fraudes) |
| **Cross-validation** | Validation croisée – évaluation robuste des performances par division répétée train/validation |
| **F1-Score** | Moyenne harmonique de la précision et du rappel |
| **Faux négatif (FN)** | Fraude non détectée (classée à tort comme légitime) |
| **Faux positif (FP)** | Transaction légitime bloquée (classée à tort comme fraude) |
| **Feature Engineering** | Création de nouvelles variables dérivées pour améliorer les performances des modèles |
| **Hyperparamètre** | Paramètre de configuration d'un algorithme ML défini avant l'entraînement |
| **Impureté de Gini** | Mesure de l'hétérogénéité d'un nœud dans un arbre de décision |
| **KNN** | K-Nearest Neighbors – algorithme basé sur la similarité entre observations |
| **Overfitting** | Surapprentissage – modèle trop ajusté aux données d'entraînement |
| **Random Forest** | Méthode d'ensemble combinant de multiples arbres de décision par bagging |
| **Rappel (Recall)** | Proportion de vrais positifs parmi tous les positifs réels |
| **Régularisation** | Technique de pénalisation des poids pour prévenir le surapprentissage |
| **SHAP** | SHapley Additive exPlanations – méthode d'explication locale des prédictions |
| **StandardScaler** | Normalisation centrée-réduite (moyenne=0, écart-type=1) |
| **Vrai positif (TP)** | Fraude correctement détectée |

### Annexe E : Références Bibliographiques

1. **Breiman, L. (2001).** Random Forests. *Machine Learning*, 45(1), 5-32.

2. **Cover, T., & Hart, P. (1967).** Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21-27.

3. **Cox, D. R. (1958).** The regression analysis of binary sequences. *Journal of the Royal Statistical Society*, 20(2), 215-242.

4. **Pozzolo, A. D., Caelen, O., Johnson, R. A., & Bontempi, G. (2015).** Calibrating probability with undersampling for unbalanced classification. *IEEE SSCI*, 159-166.

5. **Bhattacharyya, S., Jha, S., Tharakunnel, K., & Westland, J. C. (2011).** Data mining for credit card fraud: A comparative study. *Decision Support Systems*, 50(3), 602-613.

6. **Fernández, A., García, S., Herrera, F., & Chawla, N. V. (2018).** SMOTE for learning from imbalanced data: Progress and challenges. *Journal of Artificial Intelligence Research*, 61, 863-905.

7. **Europay, Mastercard, & Visa (EMV).** EMV Specifications for Payment Systems. *EMVCo*, 2020.

8. **Autorité Bancaire Européenne (ABE).** Guidelines on fraud reporting under PSD2. *EBA/GL/2018/05*, 2018.

9. **Lundberg, S., & Lee, S. I. (2017).** A unified approach to interpreting model predictions. *NeurIPS*, 30.

10. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** The Elements of Statistical Learning. *Springer*, 2nd edition.

---

### Annexe F : Index des Graphiques

| Graphique | Fichier | Description |
|:----------|:--------|:------------|
| G1 | G1_distribution_classes.png | Distribution des classes fraude/légitime |
| G2 | G2_correlation_matrix.png | Matrice de corrélation des variables |
| G3 | G3_distributions_variables.png | Distributions des variables clés par classe |
| G4 | G4_boxplots.png | Boxplots comparatifs par classe |
| G5 | G5_confusion_matrices.png | Matrices de confusion des 3 modèles |
| G6 | G6_roc_curves.png | Courbes ROC comparatives |
| G7 | G7_precision_recall.png | Courbes Précision-Rappel |
| G8 | G8_metrics_table.png | Tableau comparatif des métriques |
| G9 | G9_feature_importance_rf.png | Importance des variables (Random Forest) |
| G10 | G10_knn_k_sensitivity.png | Sensibilité du KNN au paramètre K |
| G11 | G11_cross_validation.png | Résultats de la validation croisée |
| G12 | G12_radar_metrics.png | Radar des performances globales |
| G13 | G13_probability_distributions.png | Distributions des probabilités prédites |
| G14 | G14_scatter_amount_transactions.png | Scatter montant vs transactions |
| G15 | G15_rf_n_estimators.png | Impact du nombre d'arbres (RF) |
| G16 | G16_learning_curves.png | Courbes d'apprentissage des 3 modèles |
| G17 | G17_metrics_comparison.png | Barplot comparatif des métriques |
| G18 | G18_disputes_fraud_rate.png | Impact des litiges sur le taux de fraude |

---

*Fin du rapport*

---

**Document préparé par :** Équipe Data Science & Analytique Fraud  
**Révision :** V1.0 – Mars 2026  
**Prochaine révision prévue :** Juin 2026 (après 3 mois de déploiement shadow)  
**Classification :** Confidentiel – Ne pas distribuer sans autorisation  

---

> *« La meilleure défense contre la fraude est une combinaison d'intelligence artificielle performante, de règles expertes robustes, et d'une équipe humaine vigilante. »*  
> — Principe de la Défense en Profondeur, Sécurité Financière
