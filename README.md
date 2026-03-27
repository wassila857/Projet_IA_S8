🛡️ Détection de fraude sur les transactions par carte bancaire
📌 Description du projet
Ce projet vise à détecter automatiquement les transactions frauduleuses à l’aide de modèles de machine learning.
Problème traité : 
0 : transaction légitime 
1 : transaction frauduleuse 
Objectif : maximiser la détection des fraudes tout en minimisant les faux positifs.

📊 Dataset
Nombre d’observations : 667 
Variables : 20 → 24 après feature engineering 
Fraudes : 95 (14.2%) 
Légitimes : 572 (85.8%) 

⚙️ Pipeline
Analyse exploratoire (EDA) 
Prétraitement (encodage, normalisation) 
Feature engineering 
Modélisation 
Évaluation 

🤖 Modèles
KNN 
Random Forest ⭐ 
Régression Logistique 

📈 Résultats
Modèle
Accuracy
Précision
Rappel
F1-score
AUC
KNN
0.87
0.80
0.17
0.27
0.82
Random Forest
0.95
1.00
0.67
0.80
0.90
Régression Logistique
0.77
0.35
0.67
0.46
0.82

🏆 Modèle retenu
👉 Random Forest

🚀 Déploiement
Transaction → Prétraitement → Modèle → Score 
Score ≥ 0.5 → Blocage 
Score 0.3 - 0.5 → Vérification 
Score < 0.3 → Acceptation 

🔧 Améliorations
Hyperparameter tuning 
Stacking 
Détection d’anomalies 
Ajout de données réelles 

🧰 Technologies
Python 
pandas / numpy 
scikit-learn 

💡 Conclusion
Le Random Forest offre le meilleur compromis avec une forte détection et 0 faux positifs.
