#%%
from pathlib import Path
import pandas as pd
from pandas.plotting import scatter_matrix
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

#%%
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url="https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url,tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")
    
    return pd.read_csv(Path("datasets/housing/housing.csv"))

#%%
housing = load_housing_data()

# Muestra las 10 primeras filas
print(housing.head())

#%%

# Muestra informacion general de cada columna, como cuantos no nulos hay y que tipo de dato es
print(housing.info())
# %%

# Cuando son categoricas, muestra la cantidad de datos de cada categoria
print(housing["ocean_proximity"].value_counts())
# %%

# Muestra datos como conteo, media, mediana, moda, percentiles, etc...
print(housing.describe())
# %%

housing.hist(bins=50, figsize=(12,8))
plt.show()
# %%

def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
# %%
train_set, test_set = shuffle_and_split_data(housing, 0.2)
print("%s   %s" % (len(train_set), len(test_set)))
# %%

# Agrupa los datos en 5 categorias, teniendo en cuenta rangos, esto es para hacer division estratificada
housing["income_cat"] = pd.cut(housing["median_income"], bins = [0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5])
# %%
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid = True)
plt.xlabel("Income Category")
plt.ylabel("Number of districts")
plt.show()

#%%

# Crear muestras estratificadas usando scikit para la tarea, crea 10 splits distintos para test y train, y podemos usar cuales queramos, preferencialmente el primero

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []

for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[ 0]

#%%
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis = 1, inplace = True)
# %%
housing_copy = strat_train_set.copy()
housing = housing.drop("ocean_proximity", axis=1)

#%%
housing.plot(kind="scatter", x = "longitude", y="latitude", grid=True, alpha=0.2)
plt.show() 
# %%
housing.plot(kind="scatter", x = "longitude", y="latitude", grid=True, 
             s=housing["population"]/100,label="population",
             c="median_house_value",cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10,7))
plt.show()
# %%
corr_matrix = housing.corr() # ya que el dataset no es tan grande podemos calcular el coeficiente de correlacion estandar (pearsons r)
print(corr_matrix)
# %%
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))
plt.show()
# %%
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
plt.show()
# %%
