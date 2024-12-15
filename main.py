import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.animation as animation

# Chargement du dataset
dataset = pd.read_csv('dataset.txt', sep='\t', header=None)
dataset.columns = ["X", "y"]

# Extraction des données
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

# Affichage des données
plt.scatter(x, y)
plt.show()

# Redimensionnement des données
x = np.reshape(x, (len(x), 1))  # Assurer que x a bien 10 lignes
y = np.reshape(y, (len(y), 1))  # Assurer que y a bien 10 lignes

# Initialisation des paramètres
theta = np.array([1, 0])
theta = np.reshape(theta, (2, 1))  # Assurer que theta a 2 paramètres
x1 = np.ones((len(x), 2))  # Dimensions adaptées à x (10, 2)
x1[:, 1:] = x

m = len(x1)
alpha = 0.000212

# Initialisation de la figure
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'r-')  # Ligne rouge initiale

while True:
    y1 = theta[0] + theta[1] * x  # Prédictions basées sur la régression
    plt.scatter(x, y, c='b')  # Affichage des points de données en bleu
    line1.set_ydata(y1)       # Mise à jour de la ligne de régression
    fig.canvas.draw()          # Mise à jour du graphique
    fig.canvas.flush_events()  # Rafraîchissement des événements
    
    h = x1.dot(theta)          # Calcul des prédictions
    error = h - y              # Calcul de l'erreur
    sqrd_error = np.square(error)  # Erreur quadratique
    sum_sqrd_error = np.sum(sqrd_error)
    cost = (sum_sqrd_error / (2 * m))  # Coût J(θ)
    
    xT = x1.T                   # Transposée de x1
    grad = (xT.dot(error)) / m  # Gradient de la fonction de coût
    theta = theta - alpha * grad  # Mise à jour de θ
