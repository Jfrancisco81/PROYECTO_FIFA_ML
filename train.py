import os
import sys

# sys.path.insert(0, os.path.abspath('..'))
# os.chdir(os.path.dirname(sys.path[0])) # Este comando hace que el notebook sea la ruta principal y poder trabajar en cascada
absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))

# importamos las librerias
from utils.funcion import *


# Cargamos los datos
df = pd.read_csv(r'data\raw\players_fifa23.csv')


"""Empezamos a limpiar los datos"""

# Eliminar los espacios en blanco
df.columns = df.columns.str.strip()

# Dividri el DataFrame en dos DataFrames
df_gk = df[df['BestPosition'] == 'GK']
df_otras_posiciones = df[df['BestPosition'] != 'GK']

# Verificar los nuevos DataFrames
print("DataFrame de los jugadores 'GK':")
print(df_gk)
print("\nDataFrame de los jugadores de otras posiciones:")
print(df_otras_posiciones)


print(df_otras_posiciones.describe())

df_otras_posiciones.sort_values(by='Overall', ascending=False)

# Crear una nueva columna con el BMI
df_otras_posiciones['BMI'] = df_otras_posiciones['Weight'] / ((df_otras_posiciones['Height'] / 100) ** 2)
print(df_otras_posiciones['BMI'])

# Eliminar los duplicados
df_otras_posiciones.drop_duplicates( inplace=True)

# Verificamos los Nulos
columnas_con_nan = df_otras_posiciones.columns[df_otras_posiciones.isnull().any()]

df_con_nan = df_otras_posiciones[columnas_con_nan]

df_con_nan

# Eliminar los Nulos
df_otras_posiciones.drop(df_con_nan, axis=1, inplace=True)

# Verificamos los Nulos
df_otras_posiciones.isnull().sum().sort_values(ascending=False)

# Vemos las columnas que no son numéricas
columnas_object = df_otras_posiciones.select_dtypes(include='object').columns
columnas_object

# Eliminamos los espacios en blanco
df_otras_posiciones.columns = df_otras_posiciones.columns.str.strip()

# Vemos los valores únicos de la columna 'BestPosition'
df_otras_posiciones['BestPosition'].unique()

# Eliminamos las columnas que no son numéricas y menos importantes
columnas_a_eliminar = ['FullName', 'PhotoUrl', 'Nationality', 'Positions',
       'Club', 'NationalTeam', 'PreferredFoot','AttackingWorkRate', 'DefensiveWorkRate',
       'ID', 'Growth','ValueEUR', 'WageEUR', 'ReleaseClause','ClubJoined', 'IntReputation',
       'TotalStats','BaseStats','LWRating', 'LFRating', 'CFRating', 'RFRating', 'RWRating', 'CAMRating',
       'LMRating', 'CMRating', 'RMRating', 'LWBRating', 'CDMRating',
       'RWBRating', 'LBRating', 'CBRating', 'RBRating', 'GKRating','STRating']
df_otras_posiciones = df_otras_posiciones.drop(columnas_a_eliminar, axis=1)

# Crear  nuevas columna con un get_dummies de la columna 'BestPosition'
df_get_dummies = pd.get_dummies(df_otras_posiciones['BestPosition'], drop_first=True).astype(int)
print(df_get_dummies.head())

# Concatenar los DataFrames 
fifa = pd.concat([df_otras_posiciones, df_get_dummies], axis=1)

# Eliminamos las columnas OnLoad y BestPosition
fifa.drop(['OnLoad','BestPosition'] ,axis=1, inplace=True)

# Verificamos los Nulos en la columna 'Overall'
fifa['Overall'].isnull().sum()

# Calcular la media del valor de 'Overall'
media_overall = fifa['Overall'].mean()

# Asignar clasificación de habilidad basada en la media
fifa['Clasificación Habilidad'] = np.where(fifa['Overall'] < media_overall, 'Baja', 'Alta')

# Mostrar el DataFrame con las columnas 'Overall' y 'Clasificación Habilidad'
print(fifa[['Overall', 'Clasificación Habilidad']].head())

# Verificamos los Nulos en la columna 'Clasificación Habilidad'
fifa['Clasificación Habilidad'].isnull().sum()

# Mostrar el DataFrame con las columnas 'Overall' y 'Clasificación Habilidad'
fifa.groupby('Clasificación Habilidad').Overall.mean().sort_values(ascending=False).plot(kind='bar')

# Verificamos los valores únicos de la columna 'Clasificación Habilidad'
fifa['Clasificación Habilidad'].unique()

# Asignar clasificación de habilidad con los valores Baja = 0 y Alta = 1
fifa["Clasificación Habilidad"] = fifa["Clasificación Habilidad"].map({ "Baja": 0, "Alta": 1})


# Obtiene las columnas que se deben escalar (todas excepto 'Name' )
columns_to_scale = fifa.columns.difference(['Name'])

# Crea un objeto MinMaxScaler
scaler = MinMaxScaler()

# Aplica el escalado solo a las columnas seleccionadas
fifa[columns_to_scale] = scaler.fit_transform(fifa[columns_to_scale])

""" Ahora, las columnas han sido escaladas en el rango [0, 1], conservando el DataFrame original.
"""

# Realizamos una clusterización K-Means en las columnas de posiciones
posiciones = fifa[['CB', 'CDM', 'CF','CM', 'LB', 'LM', 'LW', 
                   'LWB', 'RB', 'RM', 'RW', 'RWB', 'ST',]]

scaler = StandardScaler()
datos_normalizados = scaler.fit_transform(posiciones)

numero_clusters = len(posiciones.columns)

modelo = KMeans(n_clusters=numero_clusters)
modelo.fit(datos_normalizados)

etiquetas_clusters = modelo.labels_

fifa['Cluster'] = etiquetas_clusters
fifa.head()


# Obtener el número de clusters presentes en el DataFrame
numero_clusters = fifa['Cluster'].nunique()

# Iterar sobre cada cluster y mostrar los 10 mejores jugadores de cada uno
for cluster_id in range(numero_clusters):
    print(f"Mejores jugadores en el Cluster {cluster_id}:")
    
    # Filtrar los datos correspondientes al cluster actual y ordenar por 'Clasificación Habilidad' en orden descendente
    jugadores_cluster = fifa[fifa['Cluster'] == cluster_id].sort_values(by='Overall', ascending=False)
    
    # Mostrar solo los 10 mejores jugadores del cluster actual
    top_10_jugadores = jugadores_cluster.head(20)[['Name', 'Overall']]
    print(top_10_jugadores)
    print('\n')
    
    
# Obtener el número de clusters presentes en el DataFrame
numero_clusters = fifa['Cluster'].nunique()

# Calcular el número de filas y columnas para la disposición de subplots
num_filas = (numero_clusters + 1) // 2  # Dividir y redondear hacia arriba
num_columnas = 2

# Crear una figura con subplots para mostrar cada gráfica de cada cluster
fig, axs = plt.subplots(num_filas, num_columnas, figsize=(15, 6*num_filas), sharex=True)

# Definir un esquema de colores para los jugadores en todas las gráficas
colores = sns.color_palette('tab20', n_colors=20)

# Iterar sobre cada cluster y generar una gráfica para los 10 mejores jugadores de cada uno
for cluster_id in range(numero_clusters):
    # Filtrar los datos correspondientes al cluster actual y ordenar por 'Overall' en orden descendente
    jugadores_cluster = fifa[fifa['Cluster'] == cluster_id].sort_values(by='Overall', ascending=False)
    
    # Seleccionar solo los 10 mejores jugadores del cluster actual
    top_10_jugadores = jugadores_cluster.head(20)
    
    # Calcular la posición del subplot en la disposición de 2x4
    fila = cluster_id // num_columnas
    columna = cluster_id % num_columnas
    
    # Crear la gráfica de barras horizontal para los 10 mejores jugadores del cluster actual
    axs[fila, columna].barh(top_10_jugadores['Name'], top_10_jugadores['Overall'], color=colores)
    axs[fila, columna].set_xlabel('Overall')
    axs[fila, columna].set_title(f'Cluster {cluster_id}', fontsize=14, fontweight='bold')
    axs[fila, columna].invert_yaxis()  # Invertir el eje y para que los jugadores mejor clasificados estén en la parte superior
    
    # Eliminar los bordes innecesarios en cada subplot
    axs[fila, columna].spines['top'].set_visible(False)
    axs[fila, columna].spines['right'].set_visible(False)
    axs[fila, columna].spines['bottom'].set_visible(False)
    
    # Añadir un fondo gris claro a cada subplot
    axs[fila, columna].set_facecolor('#F5F5F5')

# Ajustar los espacios entre las gráficas para que no se superpongan
plt.tight_layout()

# Mostrar la gráfica
plt.show();

""" Gráfica de  la distribución de la columna 'Overall' """

fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Funcion de densidad
sns.distplot(fifa['Overall'], hist = False, ax=axes[0, 0])
axes[0, 0].set_title("Función de densidad")

# Histograma
sns.distplot(fifa['Overall'],
             kde=False,
             color='slategray',
             ax=axes[0, 1]);

axes[0, 1].set_title("Histograma")

# Funcion de densidad + histograma
sns.distplot(fifa['Overall'],
            kde_kws={"color": "k", "lw": 3, "label": "KDE"},
            hist_kws={"histtype": "step", "linewidth": 3,
                      "alpha": 1, "color": "g"},
             ax=axes[0, 2])


axes[0, 2].set_title("Funcion de densidad + hist.")

# Boxplot
sns.boxplot(fifa['Overall'], color="slategray", ax=axes[1, 0])
axes[1, 0].set_title("Box plot Overall")

# Violin plot
sns.violinplot(fifa['Overall'], color="slategray", ax=axes[1, 1])
axes[1, 1].set_title("Violin plot")


# Funcion de densidad + Clasificación Habilidad
sns.kdeplot(data=fifa, x='Overall', hue = 'Cluster', ax=axes[1, 2])
axes[1, 2].set_title("FDP + Overall");
    
    

# Para ver los missing values, valores uniques y cardin
data_report(fifa)


precent_missing = fifa.isnull().sum()*100/len(fifa)
missing_value_fifa = pd.DataFrame({'column_name': fifa.columns,
                                'percent_missing': precent_missing}).sort_values('percent_missing', ascending=False)
missing_value_fifa


cols_to_drop = missing_value_fifa[missing_value_fifa['percent_missing'] > 50].index.values
print("Cols:", cols_to_drop)

print("Columnas pre drop:", len(fifa.columns))

fifa.drop(columns=cols_to_drop, inplace=True)

print("Columnas post drop:", len(fifa.columns))

""" Separamos los datos en train y test  para empezar a probar los modelos"""

X = fifa.drop(['Overall', 'Name','Clasificación Habilidad'], axis=1)
y = fifa['Overall']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Creamos el modelo de regresión
modelo_lm = LinearRegression()
modelo_lm.fit(X_train, y_train)


# Mostramos los coeficientes de la regresión

coeff_fifa = pd.DataFrame(modelo_lm.coef_,
                          X.columns,columns=['coefficient'])
coeff_graf = coeff_fifa.head(10).sort_values('coefficient', ascending=False)
sns.barplot(x=coeff_graf['coefficient'], y=coeff_graf.index)

# Mostramos las predicciones
predictions = modelo_lm.predict(X_test)
print(predictions)

# Mostramos los errores y el score
print("score:",modelo_lm.score(X_test, y_test))
print("MSE:", metrics.mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("MAE:", metrics.mean_absolute_error(y_test, predictions))


# Mostramos los interceptos
intercept = modelo_lm.intercept_
features = pd.DataFrame(modelo_lm.coef_, X_train.columns, columns=['coefficient'])
features.head().sort_values('coefficient', ascending=False)

# Calcular el valor absoluto de los coeficientes para tener una magnitud de importancia
features['absolute_coefficient'] = abs(features['coefficient'])

# Ordenar las características por importancia en orden descendente
features = features.sort_values('absolute_coefficient', ascending=False)

# Gráfico de barras
plt.figure(figsize=(14, 8))
plt.bar(features.index, features['coefficient'])
plt.axhline(0, color='gray', linewidth=1, linestyle='dashed')  # Línea en y=0 para referencia
plt.xlabel('Característica')
plt.ylabel('Coeficiente')
plt.title('Importancia de las Características en la Predicción del Overall')
plt.xticks(rotation=90, ha='right')  # Rotar las etiquetas del eje x para mejorar la legibilidad
plt.show()

# Crear el modelo Decision Tree
modelo_dtree = DecisionTreeRegressor()
modelo_dtree.fit(X_train, y_train)
y_pred = modelo_dtree.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))


# Modelo de KNN
modelo_KNN = KNeighborsRegressor(n_neighbors=5) 
modelo_KNN.fit(X_train, y_train)
y_pred = modelo_KNN.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Modelo de SVR
modelo_SVM = SVR()
modelo_SVM.fit(X_train, y_train)
y_pred = modelo_SVM.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))


# Modelo de Random Forest
modelo_rf = RandomForestRegressor()
modelo_rf.fit(X_train, y_train)
y_pred = modelo_rf.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

modelo_rf.feature_importances_


# Obtener la importancia de las características del modelo Random Forest
importancia_caracteristicas = modelo_rf.feature_importances_

# Crear un DataFrame para facilitar la visualización
caracteristicas_df = pd.DataFrame({'Característica': X_train.columns, 'Importancia': importancia_caracteristicas})

# Ordenar las características por importancia en orden descendente
caracteristicas_df = caracteristicas_df.sort_values('Importancia', ascending=False).head(15)

# Gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(caracteristicas_df['Característica'], caracteristicas_df['Importancia'])
plt.xlabel('Característica')
plt.ylabel('Importancia')
plt.title('Importancia de las Características en el Modelo Random Forest')
plt.xticks(rotation=45, ha='right')  # Rotar las etiquetas del eje x para mejorar la legibilidad
plt.show()


""" Aplicamos GridSearchCV a todos los modelos y los 
hiperparámetros para obtener el mejor modelo """


# Función que aplica GridSearchCV al modelo especificado
def tune_hyperparameters(model, param_grid):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_rmse = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse_mean = cv_rmse.mean()
    return best_model, best_params, cv_rmse_mean

# Los modelos y los hiperparámetros
models = {
    'SVR': (SVR(), {
        'C': [1, 10, 100],
        'kernel': ['linear', 'rbf']
    }),
    'LinearSVR': (LinearSVR(), {
        'C': [0.1, 1, 10]
    }),
    'Linear Regression': (LinearRegression(), {}),
    'Random Forest': (RandomForestRegressor(), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10]
    }),
    'Gradient Boosting': (GradientBoostingRegressor(), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }),
    'KNN': (KNeighborsRegressor(), {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }),
    'Decision Tree': (DecisionTreeRegressor(), {
        'max_depth': [None, 5, 10]
    }),
    'XGBoost': (XGBRegressor(), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    })
}


# Aplicar GridSearchCV a cada modelo y obtener el mejor modelo con hiperparámetros ajustados
best_models = {}
for model_name, (model, param_grid) in models.items():
    best_model, best_params, cv_rmse_mean = tune_hyperparameters(model, param_grid)
    best_models[model_name] = {
        'best_model': best_model,
        'best_params': best_params,
        'cv_rmse_mean': cv_rmse_mean
    }

# Evaluar cada modelo con los mejores hiperparámetros utilizando el RMSE
for model_name, model_info in best_models.items():
    model = model_info['best_model']
    best_params = model_info['best_params']
    cv_rmse_mean = model_info['cv_rmse_mean']

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"{model_name}:")
    print(f"  Mejores hiperparámetros: {best_params}")
    print(f"  RMSE con mejores hiperparámetros: {rmse:.2f}")
    print(f"  CV RMSE promedio: {cv_rmse_mean:.2f}")



""" Nuestro Modelo de Gradient Boosting lo guardamos en un archivo """


model_GBRegressor = GradientBoostingRegressor(learning_rate= 0.1,max_depth=7, n_estimators=200)

# Entrenar el modelo con los datos de entrenamiento
model_GBRegressor.fit(X_train, y_train)

# Guardar el modelo en un archivo
model_path = "src/model/My_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model_GBRegressor, f)
    
    
    
