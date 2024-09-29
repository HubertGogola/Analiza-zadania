import numpy as np
import matplotlib.pyplot as plt

def lorenz_curve(points):
    # Sortowanie punktów po wartościach xi
    points_sorted = sorted(points, key=lambda x: x[0])
    
    # Obliczenie sum kumulatywnych
    cumulative_sum_yi = np.cumsum([point[1] for point in points_sorted])
    cumulative_sum_xi = np.cumsum([point[0] for point in points_sorted])
    
    total_y = cumulative_sum_yi[-1]
    total_x = cumulative_sum_xi[-1]
    
    # Punkty krzywej Lorenza
    L_x = cumulative_sum_xi / total_x
    L_y = cumulative_sum_yi / total_y
    
    # Obliczenie współczynnika Giniego
    area_under_lorenz = np.trapz(L_y, L_x)
    area_perfect_equal_distribution = 0.5  # Powierzchnia pod linią 45 stopni
    
    gini_coefficient = (area_perfect_equal_distribution - area_under_lorenz) / area_perfect_equal_distribution
    
    # Wykres krzywej Lorenza
    plt.figure(figsize=(8, 6))
    plt.plot(L_x, L_y, label='Krzywa Lorenza')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Równomierny rozkład')
    plt.fill_between(L_x, L_y, L_x, color='skyblue', alpha=0.4)
    plt.title('Krzywa Lorenza')
    plt.xlabel('Skumulowany udział w wartości (L(x))')
    plt.ylabel('Skumulowany udział w liczności (L(y))')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return gini_coefficient

# Przykładowe dane - wartości xi i odpowiadające im liczności yi
sample_data = [(1000, 5), (2000, 15), (3000, 25), (4000, 35), (5000, 45)]

# Obliczenie i wykreslenie krzywej Lorenza oraz współczynnika Giniego
gini_index = lorenz_curve(sample_data)
print(f"Współczynnik Giniego: {gini_index}")