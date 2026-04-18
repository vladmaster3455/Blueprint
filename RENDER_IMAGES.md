# Gestion des images sur Render

Render n'interdit pas les images. Ce qui compte, c'est la manière dont l'application les obtient.

## Cas supportés

### 1. Upload utilisateur

C'est le cas le plus simple et le plus fiable:
- l'utilisateur envoie une image depuis le navigateur
- Flask la reçoit
- le modèle exécute l'inférence
- le résultat annoté est renvoyé dans l'interface

Ce mode fonctionne bien sur Render et ne dépend pas du dataset local.

### 2. Images d'exemple emballées dans le projet

Vous pouvez inclure quelques petites images dans le dépôt, par exemple dans:
- `data/processed/11060/images/test`
- ou un dossier léger dédié comme `static/examples`

Elles seront alors servies comme de simples fichiers statiques ou via une route Flask.

## Cas à éviter sur Render

- dépendre uniquement de fichiers présents seulement sur votre machine locale
- supposer qu'un dossier de dataset volumineux sera toujours disponible au runtime
- stocker de grosses images de dataset dans le dépôt si cela alourdit trop le push

## Stratégie recommandée

Pour ce projet, la meilleure approche est:
1. conserver l'upload manuel comme mode principal
2. garder quelques images d'exemple très légères pour la démo
3. ne pas dépendre du dataset complet côté Render

## Si vous voulez une vraie galerie sur Render

Deux options propres:
- intégrer 3 à 5 images légères dans le repo
- ou héberger les images d'exemple sur un stockage public et les charger par URL

## Conclusion

Render accepte les images si elles transitent par l'application ou si elles sont accessibles depuis le projet déployé.
Le plus robuste reste l'upload utilisateur, avec éventuellement une petite galerie d'exemples embarquée.
