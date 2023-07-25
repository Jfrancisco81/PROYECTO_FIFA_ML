from flask import Flask, render_template, request, jsonify
import pandas as pd
import io
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans


app = Flask(__name__)

model_path = "C:\JUAN BOOTCAMP\Proyecto_Ml\src\model\My_model.pkl"

# Cargar el modelo entrenado en la memoria
with open(model_path, "rb") as f:
    model = pickle.load(f)


def limpiar_datos(df):
    df = df.copy()
    
    df = df[df['BestPosition'] != 'GK']
    
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Ordenar por 'Overall' en orden descendente
    df.sort_values(by='Overall', ascending=False, inplace=True)
    
    # Calcular BMI (Índice de Masa Corporal) para los jugadores GK (porteros)
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
  
    # Eliminar filas duplicadas en el DataFrame
    df.drop_duplicates(inplace=True)
    
    columnas_con_nan = df.columns[df.isnull().any()]
    
    df_con_nan = df[columnas_con_nan]
    
    df.drop(df_con_nan, axis=1, inplace=True)
   
    
    # Eliminar columnas no deseadas
    columnas_a_eliminar = ['FullName', 'PhotoUrl', 'Nationality', 'Positions', 'Club', 'NationalTeam', 'PreferredFoot',
                           'AttackingWorkRate', 'DefensiveWorkRate', 'ID', 'Growth', 'ValueEUR', 'WageEUR', 'ReleaseClause',
                           'ClubJoined', 'IntReputation', 'TotalStats', 'BaseStats', 'LWRating', 'LFRating', 'CFRating',
                           'RFRating', 'RWRating', 'CAMRating', 'LMRating', 'CMRating', 'RMRating', 'LWBRating', 'CDMRating',
                           'RWBRating', 'LBRating', 'CBRating', 'RBRating', 'GKRating', 'STRating']
    df.drop(columnas_a_eliminar, axis=1, inplace=True)
    
    # Convertir columnas categóricas en variables numéricas usando one-hot encoding
    df_get_dummies = pd.get_dummies(df['BestPosition'], drop_first=True).astype(int)
    
    df = pd.concat([df, df_get_dummies], axis=1)
   
    # Eliminar la columna original 'BestPosition' y la columna 'OnLoad' sin usar corchetes adicionales
    df.drop(['BestPosition', 'OnLoad'], axis=1, inplace=True)
    
    
    # Calcular la media del valor de 'Overall'
    media_overall = df['Overall'].mean()
    
    # Asignar clasificación de habilidad basada en la media
    df['Clasificación Habilidad'] = np.where(df['Overall'] < media_overall, 'Baja', 'Alta')
   
    # Mapear la columna 'Clasificación Habilidad' a valores numéricos
    df["Clasificación Habilidad"] = df["Clasificación Habilidad"].map({"Baja": 0, "Alta": 1})
   
    # Obtiene las columnas que se deben escalar (todas excepto 'Name')
    columns_to_scale = df.columns.difference(['Name'])
    
    # Crea un objeto MinMaxScaler para escalar los datos al rango [0, 1]
    scaler = MinMaxScaler()
    
    # Aplica el escalado solo a las columnas seleccionadas
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
   
    
    scaler = StandardScaler()
    
    posiciones = df[['CB', 'CDM', 'CF', 'CM', 'LB', 'LM', 'LW', 'LWB', 'RB', 'RM', 'RW', 'RWB', 'ST']]
   
    datos_normalizados = scaler.fit_transform(posiciones)
    
    
    numero_clusters = len(posiciones.columns)
       
   
    modelo = KMeans(n_clusters=numero_clusters)
    
    modelo.fit(datos_normalizados)
    
   
    etiquetas_clusters = modelo.labels_
    
    
    df['Cluster'] = etiquetas_clusters
    
    return pd.DataFrame(df)



@app.route('/', methods=['GET'])
def cargar_archivo():
    # Renderizar el formulario para cargar el archivo CSV
    return render_template('upload_file.html')

@app.route('/predecir-archivo', methods=['POST'])
def predecir_overall():
    # Obtener el archivo csv enviado desde el formulario
    file = request.files['file']
    
    # Leer el archivo csv en un DataFrame de Pandas
    content = file.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(content))

    # Limpiar el DataFrame antes de las predicciones
    df = limpiar_datos(df)

    # Realizar las predicciones para el DataFrame limpiado
    predictions = model.predict(df.drop(['Name','Overall'], axis=1)) 

    # Redondear las predicciones a dos decimales
    predictions = [round(pred, 2) for pred in predictions]

    # Agregar las predicciones al DataFrame original
    df['Overall_Predicho'] = predictions

    # Redondear los valores de "Overall" a dos decimales
    df['Overall'] = df['Overall'].round(2)

    # Convertir el DataFrame de resultados a formato JSON
    results_json = df.to_dict(orient="records")

    # Devolver el resultado en formato JSON a la plantilla HTML
    return render_template('results.html', results=results_json)


if __name__ == '__main__':
    app.run(debug=True)
