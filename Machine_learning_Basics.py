# TP2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Paramètres de la relation linéaire
a_param = 2.5  # Pente
b_param = 50.0  # Ordonnée à l'origine

# Générer des données avec une relation linéaire et du bruit
np.random.seed(42)  # Pour assurer la reproductibilité
t = np.random.rand(100) * 30  # 100 valeurs aléatoires pour la température entre 0 et 30
bruit = np.random.randn(100) * 10  # Bruit gaussien
v = a_param * t + b_param + bruit

# Tracer la relation linéaire originale

plt.figure(figsize=(12, 4))
plt.scatter(t, v, label='Données avec bruit')
plt.plot(t, a_param * t + b_param, color='green', label='Relation Linéaire Originale')
plt.title('Relation Linéaire Originale')
plt.xlabel('Température (t)')
plt.ylabel('Ventes de glaces (v)')
plt.legend()

plt.show()

# Fonction des moindres carrés ordinaires
def moindres_carres(x, y):
    n = len(x)
    a = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
    b = (np.sum(y))/n - a * (np.sum(x))/n
    return a, b

# Fonction coût pour la descente de gradient
def cost_function(a, b, x, y):
   
    errors = a * x + b - y
    return np.sum(errors**2)

# Gradient de la fonction coût
def gradient(a, b, x, y):
    n = len(x)
    errors = a * x + b - y
    gradient_a = np.sum(x * errors) / n
    gradient_b = np.sum(errors) / n
    return gradient_a, gradient_b

def gradient_descent(x, y, learning_rate=0.001, epsilon=1e-3, max_iter=1000):
    a, b = 0, 0  # Initial guess
    iter_count = 0

    while True:
        prev_cost = cost_function(a, b, x, y)
        gradient_a, gradient_b = gradient(a, b, x, y)
        a -= learning_rate * gradient_a
        b -= learning_rate * gradient_b
        current_cost = cost_function(a, b, x, y)

        if np.abs(prev_cost - current_cost) < epsilon or iter_count > max_iter:
            break
        iter_count += 1

    return a, b

# Générer des données aléatoires
np.random.seed(42)
x = np.random.rand(100) * 30
erreur  = np.random.randn(100) * 10
y = 2.5 * x + 50 + erreur 

# Appliquer les méthodes des moindres carrés et de la descente de gradient
a_moindres_carres, b_moindres_carres = moindres_carres(x, y)
predictions_moindres_carres = a_moindres_carres * x + b_moindres_carres

a_gradient, b_gradient = gradient_descent(x, y)
predictions_gradient = a_gradient * x + b_gradient

# Évaluer les modèles
rmse_moindres_carres, r2_moindres_carres = np.sqrt(mean_squared_error(y, predictions_moindres_carres)), r2_score(y, predictions_moindres_carres)
rmse_gradient, r2_gradient = np.sqrt(mean_squared_error(y, predictions_gradient)), r2_score(y, predictions_gradient)

# Tracer les graphiques
plt.figure(figsize=(12, 6))

# Tracer pour la méthode des moindres carrés
plt.subplot(1, 2, 1)
plt.scatter(x, y, label='Données Aléatoires')
plt.plot(x, predictions_moindres_carres, color='red', label='Moindres Carrés')
plt.title('Régression Linéaire - Moindres Carrés')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Tracer pour la méthode de descente de gradient
plt.subplot(1, 2, 2)
plt.scatter(x, y, label='Données Aléatoires')
plt.plot(x, predictions_gradient, color='green', label='Descente de Gradient')
plt.title('Régression Linéaire - Descente de Gradient')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.show()

# Afficher les résultats d'évaluation
print("Méthode des Moindres Carrés - RMSE:", rmse_moindres_carres, ", R²:", r2_moindres_carres)
print("Méthode de Descente de Gradient - RMSE:", rmse_gradient, ", R²:", r2_gradient)

# Données du Tableau 1.2 (remplacez par des données réelles)
temperature_tableau = np.array([14.2, 16.4, 11.9, 15.2, 18.5, 22.1, 19.4, 25.1, 23.4, 18.1, 22.6, 17.2])
ventes_glaces_tableau = np.array([215, 325, 185, 332, 406, 522, 412, 614, 544, 421, 445, 408])

# Appliquer les méthodes de régression aux données du tableau
a_tableau, b_tableau = moindres_carres(temperature_tableau, ventes_glaces_tableau)
predictions_tableau = a_tableau * temperature_tableau + b_tableau

# Tracer les données aléatoires et leur régression linéaire
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x, y, label='Données Aléatoires')
plt.plot(x, predictions_moindres_carres, color='red', label='Moindres Carrés - Aléatoire')
plt.plot(x, predictions_gradient, color='green', label='Descente de Gradient - Aléatoire')
plt.title('Régression sur Données Aléatoires')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Tracer les données du tableau et leur régression linéaire
plt.subplot(1, 2, 2)
plt.scatter(temperature_tableau, ventes_glaces_tableau, label='Données du Tableau 1.2')
plt.plot(temperature_tableau, predictions_tableau, color='blue', label='Moindres Carrés - Tableau 1.2')
plt.title('Régression sur Données du Tableau 1.2')
plt.xlabel('Température (°C)')
plt.ylabel('Ventes de glaces')
plt.legend()

plt.show()

#Quelles seront les ventes de glaces estimées pour des températures de 13,20 et 27 °C?

# Paramètres de la régression linéaire obtenus 
a = 2.5  # Pente
b = 50.0  # Ordonnée à l'origine

# Températures pour lesquelles nous voulons estimer les ventes de glaces
temperatures = [13, 20, 27]

# Estimations des ventes de glaces pour les températures données
ventes_estimees = [a * t + b for t in temperatures]


print(ventes_estimees)

 #Supposons que le glacier a vendu 450 equand la température était a 21°C. Combien devrait-il faire de ventes pour des températures de 13, 20 et 27 °C ?

# Inclusion de la nouvelle donnée dans l'ensemble de données existant
temperature_tableau = np.array([14.2, 16.4, 11.9, 15.2, 18.5, 22.1, 19.4, 25.1, 23.4, 18.1, 22.6, 17.2, 21])
ventes_glaces_tableau = np.array([215, 325, 185, 332, 406, 522, 412, 614, 544, 421, 445, 408, 450])

# Recalcul des paramètres de la régression linéaire avec la nouvelle donnée
a_tableau_ajuste, b_tableau_ajuste = moindres_carres(temperature_tableau, ventes_glaces_tableau)

# Estimations des ventes de glaces pour les températures 13, 20, et 27 °C
ventes_estimees_ajustees = [a_tableau_ajuste * t + b_tableau_ajuste for t in temperatures]
print(ventes_estimees_ajustees)