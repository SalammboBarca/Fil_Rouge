# Fil_Rouge

## Architecture
> Fil_rouge
>> filrouge: contient tout les modules .py
>>> multilabel_to_multiclass.py: objet Multiclass qui retourne les labels du dataset d'entrée sous forme de multiclasses. Les 6 labels initiaux se transforment en 63 labels représentant les combinaisons de chacun (e.g le label '123456' signifie que le commentaire a tout les labels (toxic, severe_toxic, etc.)).

>>> neural_network.py: classification de ces multiclasses via un réseau de neuronne

>>> vectorize_comments.py: vectorize les commentaires avec TfIdf

>> notebooks: contient les notebooks sur les différentes techniques testées 
>>> word2vec.ipynb: notebook sur word2vec

>>> stat_desc.ipynb: notebook sur le jeu de données à traiter

>>> vector.ipynb: notebook pour une premiere prise en main des données sur la classification

> README.md documentation
> requirements.txt: packages nécessaires 
> config.dist: exemple du format utile pour la configuration du projet