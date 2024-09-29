import numpy as np
import pandas as pd
from scipy.stats import variation, skew, kurtosis

def descriptive_stats(vector):
    # Obliczenie statystyk opisowych
    mean_value = np.mean(vector)
    std_dev = np.std(vector)
    coef_var = variation(vector)
    min_value = np.min(vector)
    percentile_10 = np.percentile(vector, 10)
    q1 = np.percentile(vector, 25)
    median = np.median(vector)
    q3 = np.percentile(vector, 75)
    percentile_90 = np.percentile(vector, 90)
    max_value = np.max(vector)
    data_range = np.ptp(vector)
    iqr = q3 - q1
    skewness = skew(vector)
    kurt = kurtosis(vector)

    # Tworzenie ramki danych z wynikami
    stats_df = pd.DataFrame({
        'Statystyka opisowa': ['Średnia', 'Odchylenie standardowe', 'Współczynnik zmienności',
                               'Minimum', '10 percentyl', '1 kwartyl', 'Mediana', '3 kwartyl',
                               '90 percentyl', 'Maksimum', 'Rozstęp danych', 'Rozstęp międzykwartylowy',
                               'Skośność', 'Kurtoza'],
        'Wartość': [mean_value, std_dev, coef_var, min_value, percentile_10, q1, median,
                    q3, percentile_90, max_value, data_range, iqr, skewness, kurt]
    })

    return stats_df

# Przykładowy wektor danych
np.random.seed(0)
sample_vector = np.random.normal(loc=10, scale=2, size=100)

# Obliczenie i wyświetlenie statystyk opisowych
results = descriptive_stats(sample_vector)
print(results)