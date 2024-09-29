import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Przykładowe dane - miesięczne średnie temperatury
# Można wczytać rzeczywiste dane z pliku lub stworzyć szereg czasowy
np.random.seed(0)
months = pd.date_range('2018-01-01', periods=60, freq='M')
original_data = pd.Series(np.random.normal(loc=15, scale=5, size=len(months)), index=months)

# Dodanie efektu sezonowego (np. dla przykładu symulacyjnego)
seasonal_effect = np.sin(np.linspace(0, 2*np.pi, len(months)))
data_with_seasonality = original_data + seasonal_effect

# Wyrównanie sezonowe za pomocą dekompozycji
decomposition = sm.tsa.seasonal_decompose(data_with_seasonality, model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Porównanie oryginalnego szeregu czasowego z wyrównanym sezonowo
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(original_data, label='Oryginalne dane')
plt.legend(loc='upper left')
plt.subplot(4, 1, 2)
plt.plot(data_with_seasonality, label='Dane z efektem sezonowym')
plt.legend(loc='upper left')
plt.subplot(4, 1, 3)
plt.plot(seasonal, label='Efekt sezonowy')
plt.legend(loc='upper left')
plt.subplot(4, 1, 4)
plt.plot(residual, label='Reszty po usunięciu trendu i sezonowości')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Ocena wpływu wahania sezonowego
original_std = original_data.std()
seasonal_std = seasonal.std()
seasonal_effect_ratio = seasonal_std / original_std
print(f'Odchylenie standardowe oryginalnych danych: {original_std:.2f}')
print(f'Odchylenie standardowe efektu sezonowego: {seasonal_std:.2f}')
print(f'Współczynnik zmienności efektu sezonowego w stosunku do oryginalnych danych: {seasonal_effect_ratio:.2f}')

# Wady i zalety użycia danych wyrównanych sezonowo
print('\nWady i zalety użycia danych wyrównanych sezonowo:')
print('- Zalety:')
print('  - Usunięcie efektów sezonowych pozwala na lepsze zrozumienie trendów i zmian długoterminowych.')
print('  - Dane stają się bardziej porównywalne między różnymi okresami.')
print('- Wady:')
print('  - Trudność w interpretacji danych po usunięciu efektów sezonowych.')
print('  - Utrata informacji sezonowej, która może być istotna w niektórych analizach.')