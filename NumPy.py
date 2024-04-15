import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
import math

#tp1

#Lois discretes
def Lois_discretes():
    # Paramètres lambda à étudier
    lambdas = [1, 15, 40]

    # Fonction de masse de probabilité théorique de la loi de Poisson
    def poisson_pmf(k, lam):
    
        return (np.exp(-lam) * lam**k) /  factorial(k)

    # Tracer la fonction densité de probabilité pour chaque lambda
    plt.figure(figsize=(12, 4))
    for lam in lambdas:
        k_values = np.arange(0, 2 * lam + 1)
        plt.plot(k_values, poisson_pmf(k_values, lam),'o-', label=f'λ = {lam}')

    plt.title('Fonction Densité de Probabilité (Poisson)')
    plt.xlabel('k')
    plt.ylabel('Probabilité')
    plt.legend()
    plt.show()

    # Tracer la fonction de répartition pour chaque lambda
    plt.figure(figsize=(12, 4))
    for lam in lambdas:
        k_values = np.arange(0, 2 * lam + 1)
        plt.step(k_values, np.cumsum(poisson_pmf(k_values, lam)), label=f'λ = {lam}', where='post')

    plt.title('Fonction de Répartition (Poisson)')
    plt.xlabel('k')
    plt.ylabel('Probabilité cumulative')
    plt.legend()
    plt.show()

    # Fonction pour calculer la probabilité binomiale
    def binomial_pmf(k, n, p):
        return np.math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    # Fonction pour dessiner la fonction densité et la fonction de répartition pour la loi binomiale
    def plot_binomial(n, p):
        # Générer des valeurs aléatoires selon la loi binomiale
        data = np.random.binomial(n, p, 10000)
    
        # Calculer la fonction de densité théorique
        binomial = np.array([binomial_pmf(k, n, p) for k in range(n + 1)])
    
    
        # Tracer la fonction densité simulée et théorique
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.hist(data, bins=np.arange(0, n + 1) - 0.5, density=True, alpha=0.5, color='green', label='Simulée')
        plt.stem(np.arange(0, n + 1), binomial, markerfmt='ro', linefmt='r-', basefmt='r-', label='Théorique')
        plt.title(f'Loi Binomiale (n={n}, p={p}) - Fonction densité')
        plt.xlabel('Valeurs')
        plt.ylabel('Probabilité')
    

        # Tracer la fonction de répartition
        plt.subplot(1, 2, 2)
        plt.hist(data, bins=np.arange(0, n + 1) - 0.5, density=True, cumulative=True, alpha=0.5, color='green', label='Simulée')
        plt.plot(np.arange(0, n + 1), np.cumsum(binomial), 'r-', label='Théorique')
        plt.title(f'Loi Binomiale (n={n}, p={p}) - Fonction de répartition')
        plt.xlabel('Valeurs')
        plt.ylabel('Probabilité cumulative')
    

        plt.show()

    # Tester avec différentes valeurs de (n, p) pour la loi binomiale
    parameters_binomial = [(50, 0.5), (50, 0.85), (50, 0.15)]
    for n, p in parameters_binomial:
        plot_binomial(n, p)



#Loi Continues
def Loi_Continues():
    # Fonction densité de probabilité (PDF) pour la loi normale
    def norm_pdf(x, mu, sigma_squared):
        return (1 / np.sqrt(2 * np.pi * sigma_squared)) * np.exp(-(x - mu)**2 / (2 * sigma_squared))

    # Fonction de répartition cumulative (CDF) pour la loi normale
    def norm_cdf(x, mu, sigma_squared):
        z = (x - mu) / np.sqrt(sigma_squared)
        return (1 + math.erf(z / np.sqrt(2))) / 2

    # Fonction densité de probabilité (PDF) pour la loi exponentielle
    def exp_pdf(x, lambda_):
        return lambda_ * np.exp(-lambda_ * x)

    # Fonction de répartition cumulative (CDF) pour la loi exponentielle
    def exp_cdf(x, lambda_):
        return 1 - np.exp(-lambda_ * x)

    # Loi Normale N(µ, σ)
    mu = 20
    sigma_squared = 50 * 0.25
    sample_size = 1000
    data_normal = np.random.normal(mu, np.sqrt(sigma_squared), sample_size)
    x_values = np.linspace(mu - 4 * np.sqrt(sigma_squared), mu + 4 * np.sqrt(sigma_squared), 100)

    # Tracer la fonction densité de probabilité (PDF) pour la loi normale
    plt.figure(figsize=(12, 4))
    plt.hist(data_normal, bins=30, density=True, alpha=0.7, label='Simulée (Normale)')
    plt.plot(x_values, [norm_pdf(x, mu, sigma_squared) for x in x_values], 'r-', label='Théorique (Normale)')
    plt.title('Fonction Densité de Probabilité (Normale)')
    plt.xlabel('Valeurs')
    plt.ylabel('Densité de probabilité')
    plt.legend()
    plt.show()

    # Tracer la fonction de répartition cumulative (CDF) pour la loi normale
    plt.figure(figsize=(12, 4))
    plt.hist(data_normal, bins=30, density=True, cumulative=True, alpha=0.7, label='Simulée (Normale)')
    plt.plot(x_values, [norm_cdf(x, mu, sigma_squared) for x in x_values], 'r-', label='Théorique (Normale)')
    plt.title('Fonction de Répartition Cumulative (Normale)')
    plt.xlabel('Valeurs')
    plt.ylabel('Probabilité cumulative')
    plt.legend()
    plt.show()

    # Loi Exponentielle E(λ)
    lambdas_exponential = [0.5, 0.7, 2]
    sample_size = 1000
    data_exponential = [np.random.exponential(1/lambda_, sample_size) for lambda_ in lambdas_exponential]
    x_values_exp = np.linspace(0, np.max(np.concatenate(data_exponential)), 100)

    # Tracer la fonction densité de probabilité (PDF) pour la loi exponentielle
    plt.figure(figsize=(12, 4))
    for i, lambda_ in enumerate(lambdas_exponential):
        plt.hist(data_exponential[i], bins=30, density=True, alpha=0.7, label=f'Simulée (Exponentielle, λ={lambda_})')
        plt.plot(x_values_exp, [exp_pdf(x, lambda_) for x in x_values_exp], label=f'Théorique (Exponentielle, λ={lambda_})')
    plt.title('Fonction Densité de Probabilité (Exponentielle)')
    plt.xlabel('Valeurs')
    plt.ylabel('Densité de probabilité')
    plt.legend()
    plt.show()

    # Tracer la fonction de répartition cumulative (CDF) pour la loi exponentielle
    plt.figure(figsize=(12, 4))
    for i, lambda_ in enumerate(lambdas_exponential):
        plt.hist(data_exponential[i], bins=30, density=True, cumulative=True, alpha=0.7, label=f'Simulée (Exponentielle, λ={lambda_})')
        plt.plot(x_values_exp, [exp_cdf(x, lambda_) for x in x_values_exp], label=f'Théorique (Exponentielle, λ={lambda_})')
    plt.title('Fonction de Répartition Cumulative (Exponentielle)')
    plt.xlabel('Valeurs')
    plt.ylabel('Probabilité cumulative')
    plt.legend()
    plt.show()



#Intervalle de confiance
def Intervalle_de_confiance():
    # Fonction pour calculer l'intervalle de confiance
    def confidence_interval(sample, alpha=0.05):
        n = len(sample)
        mean = np.mean(sample)
        s = np.std(sample, ddof=1)  # Utiliser ddof=1 pour l'estimation non biaisée de l'écart-type
        t_alpha = np.abs(np.percentile(np.random.standard_t(df=n-1, size=10000), alpha/2 * 100))  # Quantile de la loi de Student

        lower_bound = mean - t_alpha * s / np.sqrt(n)
        upper_bound = mean + t_alpha * s / np.sqrt(n)

        return lower_bound, upper_bound

    # Générer un échantillon de taille n à partir d'une loi normale
    np.random.seed(42)  # Pour la reproductibilité
    sample = np.random.normal(loc=20, scale=np.sqrt(50*0.25), size=100)

    # Calculer l'intervalle de confiance avec un niveau de confiance de 95%
    confidence_interval_result = confidence_interval(sample, alpha=0.05)

    # Afficher les résultats
    print("\nÉchantillon Mean:", np.mean(sample))
    print("Intervalle de Confiance (95%):", confidence_interval_result)



#Temps de réaction
def Temps_de_réaction():
    # Temps de réaction des conducteurs
    reaction_times = np.array([0.98, 1.4, 0.84, 0.86, 0.54, 0.68, 1.35, 0.76, 0.79, 0.99, 0.88, 0.75, 0.45, 1.09, 0.68,
                            0.60, 1.13, 1.30, 1.20, 0.91, 0.74, 1.03, 0.61, 0.98, 0.91])

    # Paramètres
    variance = 0.25
    alpha_95 = 0.05
    alpha_99 = 0.01

    # Calcul de la moyenne empirique
    mean_empirical = np.mean(reaction_times)

    # Tracer de l'histogramme
    plt.hist(reaction_times, bins=10, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Temps de Réaction')
    plt.ylabel('Fréquence relative')
    plt.title('Histogramme des Temps de Réaction')
    plt.show()

    # Intervalles de confiance à 95% et 99% avec la variance connue
    ci_95_var = (mean_empirical - 1.96 * np.sqrt(variance/len(reaction_times)),
                   mean_empirical + 1.96 * np.sqrt(variance/len(reaction_times)))

    ci_99_var = (mean_empirical - 2.576 * np.sqrt(variance/len(reaction_times)),
                   mean_empirical + 2.576 * np.sqrt(variance/len(reaction_times)))

    print(f"\nIntervalles de confiance à 95% (variance connue): {ci_95_var}")
    print(f"Intervalles de confiance à 99% (variance connue): {ci_99_var}")

    # Intervalles de confiance à 95% et 99% avec la variance empirique
    var_empirical = np.var(reaction_times, ddof=1)

    ci_95_empirical_var = (mean_empirical - 2.064 * np.sqrt(var_empirical/len(reaction_times)),
                       mean_empirical + 2.064 * np.sqrt(var_empirical/len(reaction_times)))

    ci_99_empirical_var = (mean_empirical - 2.787 * np.sqrt(var_empirical/len(reaction_times)),
                       mean_empirical + 2.787 * np.sqrt(var_empirical/len(reaction_times)))

    print(f"\nIntervalles de confiance à 95% (variance empirique): {ci_95_empirical_var}")
    print(f"Intervalles de confiance à 99% (variance empirique): {ci_99_empirical_var}")



#Estimation d’une proportion
def Estimation_dune_proportion():
    # Données
    nombre_etudiants = 1000
    nombre_etudiants_suivant_algorithmique = 673
    proportion_estimee = nombre_etudiants_suivant_algorithmique / nombre_etudiants

    # Niveau de confiance (80%)
    confidence_level = 0.80

    # Calcul du score z critique
    z_critical = 1.28  # Pour un niveau de confiance de 80%

    # Calcul de l'intervalle de confiance
    margin_of_error = z_critical * np.sqrt((proportion_estimee * (1 - proportion_estimee)) / nombre_etudiants)
    confidence_interval = (proportion_estimee - margin_of_error, proportion_estimee + margin_of_error)

    print(f"\nIntervalle de confiance à {confidence_level * 100}% : {confidence_interval}")




def afficher_menu():
    print("\n1. Lois discretes")
    print("2. Loi Continues")
    print("3. Intervalle de confiance")
    print("4. Temps de réaction")
    print("5. Estimation d’une proportion")
    print("6. Quitter\n")

def traiter_option(option):
    if option == 1:
        Lois_discretes()
    elif option == 2:
        Loi_Continues()
    elif option == 3:
        Intervalle_de_confiance()
    elif option == 4:
        Temps_de_réaction()
    elif option == 5:
        Estimation_dune_proportion()
    elif option == 6:
        print("Au revoir !")
        exit()
    else:
        print("Choix invalide. Veuillez entrer un nombre entre 1 et 6.")

if __name__ == "__main__":
    while True:
        afficher_menu()
        
        try:
            choix = int(input("Entrez votre choix : "))
            traiter_option(choix)
        except ValueError:
            print("Erreur : Veuillez entrer un nombre.")
