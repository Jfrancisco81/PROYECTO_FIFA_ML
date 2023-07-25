import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import requests
import time
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from scipy import interpolate
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor,  GradientBoostingRegressor
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, PolynomialFeatures
from sklearn.decomposition import PCA, KernelPCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import joblib
from sklearn.model_selection import cross_val_score
import pickle
from xgboost import XGBRegressor
from sklearn import metrics



def set_display_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)



def eliminar_espacios(df, columna):
    """
    Función que elimina los espacios en blanco al principio y al final de la cadena de caracteres en una columna de un DataFrame.
    
    Parámetros:
        - df: DataFrame al que se le aplicará la limpieza.
        - columna: Nombre de la columna a limpiar.
        
    Retorna:
        - El DataFrame con la columna limpia.
    """
    df[columna] = df[columna].str.strip()
    return df


def analizar_columnas(df,col):
    resultados = {}
    media = np.mean(df[col])
    desv_est = np.std(df[col])
    minimo = np.min(df[col])
    maximo = np.max(df[col])
    mediana = np.median(df[col])
    p_25 =np.percentile(df[col],25)
    p_75 =np.percentile(df[col],75)
    resultados[col] = {'media': media, 'desviacion_estandar': desv_est, 
                       'minimo': minimo, 'maximo': maximo,'mediana': mediana,
                       'percentile_25': p_25, 'percentile_75': p_75 }
    
     
    
    sns.histplot(data=df, x=col, kde=True, color='green')
    
    return resultados




def jugadores_por_continent(df, continent):
    # Crear una lista de países en Europa o Sudamérica
    if continent == 'Europa':
        countries = ['Germany', 'Belgium', 'Croatia', 'Denmark', 'Spain', 'France', 'Ireland', 'Latvia', 'Luxembourg', 'Netherlands', 'Sweden', 
                     'Bulgaria', 'Slovakia', 'Estonia', 'Greece', 'Malta', 'Poland', 'Czech' 'Republic', 'Austria', 'Cyprus', 'Slovenia', 'Finland',
                     'Hungary', 'Italy', 'Lithuania', 'Portugal', 'Romania']
    elif continent == 'Latinoamérica':
        countries = ['Argentina', 'Bahamas', 'Barbados', 'Belice', 'Bolivia',' Brazil', 'Chile', 'Colombia', 'Costa Rica', 'Cuba',
                     'Ecuador', 'El Salvador', 'San Cristóbal y Nieves', 'Granada', 'Guyana', 'Haití', 'Honduras', 
                     'Jamaica', 'Mancomunidad de Dominica', 'México', 'Nicaragua', 'Panamá', 'Paraguay', 'Perú', 'República Dominicana',
                     'San Vicente y las Granadinas', 'Santa Lucía', 'Surinam', 'Trinidad y Tobago', 'Uruguay' , 'Venezuela']
    else:
        print("Continent not recognized")
        return None
    
    # Filtrar el dataframe para obtener solo los jugadores de países en la lista
    filtered_df = df[df['Nacionalidad'].isin(countries)]
    
    # Devolver el nuevo dataframe con los jugadores seleccionados
    return filtered_df






def prueba_ks(muestra1, muestra2, nivel_significancia):
    """
    Realiza una prueba de Kolmogorov-Smirnov para comparar la distribución de dos muestras.
    Devuelve el valor de D, el p-valor y una conclusión en función del nivel de significancia.
    """
    D, p_valor = stats.ks_2samp(muestra1, muestra2)
    if p_valor < nivel_significancia:
        conclusion = "Se rechaza la hipótesis nula."
    else:
        conclusion = "No se puede rechazar la hipótesis nula."
    return D, p_valor, conclusion




def t_test(df, col1, col2, alpha=0.05):
    """
    Función que realiza una prueba t de dos muestras independientes 
    para comparar las medias de dos columnas de un DataFrame dado.
    
    Parámetros:
        df (pandas.DataFrame): el DataFrame que contiene las dos columnas a comparar.
        col1 (str): el nombre de la primera columna a comparar.
        col2 (str): el nombre de la segunda columna a comparar.
        alpha (float): nivel de significancia (por defecto 0.05).
        
    Returns:
        None
        
    Imprime:
        El estadístico t, el p-Valor, y una conclusión sobre si se rechaza o no la hipótesis nula.
    """
    x = df[col1]
    y = df[col2]
    t_stat, p_valor = stats.ttest_ind(x, y, equal_var=False)
    print("Valor de t: {:.4f}".format(t_stat))
    print("Valor de p: {:.4f}".format(p_valor))
    
    if p_valor < alpha:
        print("Rechazar la hipótesis nula: {} y {} tienen una diferencia significativa (p-Valor: {:.4f})".format(col1, col2, p_valor))
    else:
        print("No se puede rechazar la hipótesis nula: no hay evidencia suficiente para afirmar que {} y {} tienen una diferencia significativa (p-Valor: {:.4f})".format(col1, col2, p_valor))





def ks_test(df, col1, col2, alpha=0.05):
    """
    Realiza la prueba de Kolmogorov-Smirnov para comparar las distribuciones de dos columnas en un DataFrame.
    
    Parámetros:
    - df: DataFrame con los datos.
    - col1: Nombre de la primera columna a comparar.
    - col2: Nombre de la segunda columna a comparar.
    - alpha: Nivel de significancia (por defecto 0.05).
    
    Retorna:
    - Un diccionario con los resultados de la prueba.
    """
    # Realizar la prueba de Kolmogorov-Smirnov
    ks_stat, p_val = ks_2samp(df[col1], df[col2])
    
    # Comprobar si se rechaza o no la hipótesis nula
    if p_val < alpha:
        result = print("Se rechaza la hipótesis nula: las distribuciones son diferentes (p-Valor: {:.4f})".format(p_val))
    else:
        result = print("No se puede rechazar la hipótesis nula: las distribuciones son iguales (p-Valor: {:.4f})".format(p_val))
    
    # Guardar los resultados en un diccionario
    results_dict = {"KS statistic": ks_stat,
                    "p-Valor": p_val,
                    "result": result}
    
    return results_dict

def scatterplot(df, x, y):
    plt.scatter(df[x], df[y])
    plt.xlabel('Variable ' + x)
    plt.ylabel('Variable ' + y)
    plt.title('Scatterplot de ' + x + ' vs ' + y)
    plt.show()




def corr_heatmap(df):
    plt.figure(figsize=(40,40))
    sns.heatmap(df.corr(),
                vmin=-1,
                vmax=1,
                cmap='RdBu_r',
                square=True,
                linewidths=.1,
                annot=True);




def scatterplot_bivariate(df, x_col, y_col):
    sns.set_palette("Set2")
    sns.set(style="dark")
    sns.lmplot(x=x_col, y=y_col, data=df)
    plt.title('Scatterplot de {} vs {}'.format(x_col, y_col))
    plt.show()
    
   

def chi_square_test(df, col1, col2):
    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2, p_Valor, dof, expected = chi2_contingency(contingency_table)
    if p_Valor < 0.05:
        print("Rechazar hipótesis nula: existe una relación significativa entre las variables {} y {}".format(col1, col2))
    else:
        print("No se puede rechazar hipótesis nula: no existe suficiente evidencia para afirmar que hay una relación significativa entre las variables {} y {}".format(col1, col2))
    return chi2, p_Valor








def plot_top_countries_by_column(df, column_name, agg_func='mean', top_n=10, agg_column='ALTURA'):
    """
    Grafica los países con los valores más altos para una columna determinada.
    
    Argumentos:
    - df: DataFrame que contiene los datos.
    - column_name: Nombre de la columna que se utilizará para la agrupación.
    - agg_func: función de agregación a aplicar sobre la columna (por defecto: 'mean').
    - top_n: número de países a mostrar en la gráfica (por defecto: 10).
    - agg_column: columna sobre la que se realizará la agregación (por defecto: 'ALTURA').
    """
    
    grouped = df.groupby(column_name).agg({ agg_column: agg_func, 'Edad': 'mean' }).reset_index()
    top_countries = grouped.nlargest(top_n, agg_column)
    palette = sns.color_palette("Paired", top_n)
    sns.barplot(data=top_countries, x=column_name, y=agg_column, palette=palette)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {top_n} países con la mayor {agg_column.lower()} y la Edad promedio')
    plt.xlabel(column_name.capitalize())
    plt.ylabel(f'{agg_column.capitalize()}')
    plt.show()


    
    


def plot_tallest_countries(df, col1, col2, n=10):
    """
    Crea un gráfico de barras que muestra los países con las alturas promedio más altas.

    Argumentos:
    - df: DataFrame con los datos.
    - col1: Nombre de la columna con los países.
    - col2: Nombre de la columna con las alturas.
    - n: número de países a mostrar.
    """

    # Agrupar por país y calcular el promedio
    country_heights = df.groupby(col1)[col2].mean().reset_index()

    # Ordenar de manera ascendente y tomar los primeros n países
    tallest_countries = country_heights.sort_Valors(col2, ascending=False).head(n)

    # Crear gráfico de barras
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data=tallest_countries, x=col1, y=col2)
    plt.title(f"Los {n} países con  {col2} promedio más altas")
    plt.xlabel("País")
    plt.ylabel(f"{col2}")

    # Rotar etiquetas del eje x para que no se superpongan
    plt.xticks(rotation=45, ha='right')

    # Añadir valores numéricos encima de las barras
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='gray', xytext=(0, 5),
                    textcoords='offset points')

    plt.show()

def plot_top_players(df, col, n=20):
    """
    Crea un gráfico de barras que muestra los jugadores con los valores más altos en la columna especificada.

    Argumentos:
    - df: DataFrame con los datos.
    - col: Nombre de la columna a utilizar.
    - n: número de jugadores a mostrar.
    """

    # Tomar los n jugadores con los valores más altos en la columna especificada
    top_players = df.nlargest(n, col)

    # Crear gráfico de barras
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data=top_players, x=col, y='Nombre')
    plt.title(f"Los {n} jugadores con los valores más altos en {col}")
    plt.xlabel(col)
    plt.ylabel("Jugador")

    # Rotar etiquetas del eje y para que no se superpongan
    plt.xticks(rotation=45, ha='right')

    # Añadir valores numéricos encima de las barras
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.2f}", (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=11, color='gray', xytext=(5, 0),
                    textcoords='offset points')

    plt.show()


def plot_regression(df, x_col, y_col, title='', xlabel='', ylabel='', color='darkblue', figsize=(10, 6)):
    """
    Crea un gráfico de dispersión con una línea de regresión lineal.

    Argumentos:
    - df: DataFrame con los datos.
    - x_col: Nombre de la columna con los datos del eje x.
    - y_col: Nombre de la columna con los datos del eje y.
    - title: título del gráfico.
    - xlabel: título del eje x.
    - ylabel: título del eje y.
    - color: color de los puntos del gráfico.
    - figsize: tamaño de la figura.

    Retorna:
    - None
    """
    # Obtener los datos de las columnas especificadas
    x = df[x_col]
    y = df[y_col]

    # Calcular la línea de regresión lineal
    fit = np.polyfit(x, y, deg=1)

    # Crear gráfico de dispersión con una línea de regresión lineal
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.figure(figsize=figsize)
    ax = sns.scatterplot(data=df, x=x_col, y=y_col, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, fit[0] * x + fit[1], color='red')
    plt.annotate('y = {:.2f}x + {:.2f}'.format(fit[0], fit[1]), xy=(min(x)+5, max(y)-1000), color='red')
    plt.show()

def get_top_economic_players(df, n=20):
    """
    Retorna los n jugadores más económicos con el mejor promedio de General.

    Argumentos:
    - df: DataFrame con los datos.
    - n: número de jugadores a mostrar (por defecto es 20).
    """
    # Filtrar el DataFrame para incluir solo las columnas necesarias
    df = df[['Nombre', 'Edad', 'Club', 'Valor', 'General']]

    # Limpiar la columna 'Valor' y convertirla en millones des
    # df['Valor'] = df['Valor'].apply(lambda x: float( * 

    # Calcular el valor por unidad de General
    df['Valor_General'] = df['Valor'] / df['General']
    df= df.round({'Valor_General': 2})


    # Ordenar por valor por unidad de General en orden ascendente
    df = df.sort_values('Valor_General',ascending=True)

    # Seleccionar los primeros n jugadores con el valor por unidad de General más bajo
    top_players = df.head(n)

    return top_players




def top_economic_players(df):
    """
    Esta función devuelve un gráfico de barras que muestra los 10 jugadores más económicos
    y con mejor promedio de General.
    
    Argumentos:
    - df: DataFrame con los datos.
    """
    
    # Ordenar jugadores por valor ens y seleccionar los 10 primeros con mejor promedio de General
    top_players = df.sort_values(by=['Valor_euro', 'General'], ascending=[True, False]).head(10)
    
    # Crear gráfico de barras
    plt.figure(figsize=(12, 8))
    sns.set_style('white')
    sns.set_palette("dark")
    ax = sns.barplot(data=top_players, x='Nombre', y='General', hue='Valor_euro')
    
    # Añadir etiquetas y títulos
    plt.title('Los 10 jugadores más económicos y con mejor promedio de General')
    plt.xlabel('Jugador')
    plt.ylabel('Promedio de General')
    plt.xticks(rotation=45, ha='right', va='top')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, ncol=1)

    
    # Mostrar gráfico
    plt.show()
    
    


def plot_histograms(data, target_column):
    numeric_columns = data.select_dtypes(include='number').columns.sort_values(ascending=False)
    
    for column in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=data, x=column, hue=target_column, kde=True, palette='coolwarm')
        plt.title(f'Histograma de {column} ({target_column})')
        plt.xlabel(column)
        plt.ylabel('Frecuencia')
        plt.show()


def analizar_todas_columnas(df):
    resultados = {}
    numeric_columns = df.select_dtypes(include='number').columns
    
    for col in numeric_columns:
        media = np.mean(df[col])
        desv_est = np.std(df[col])
        minimo = np.min(df[col])
        maximo = np.max(df[col])
        mediana = np.median(df[col])
        p_25 = np.percentile(df[col], 25)
        p_75 = np.percentile(df[col], 75)
        resultados[col] = {'media': media, 'desviacion_estandar': desv_est,
                           'minimo': minimo, 'maximo': maximo, 'mediana': mediana,
                           'percentile_25': p_25, 'percentile_75': p_75}
    
    return resultados





def plot_histograms_with_analysis(data, target_column):
    numeric_columns = data.select_dtypes(include='number').columns.sort_values(ascending=False)

    for column in numeric_columns:
        plt.figure(figsize=(10, 8))
        sns.histplot(data=data, x=column, hue=target_column, kde=True, palette='coolwarm')
        plt.title(f'Histograma de {column} ({target_column})')
        plt.xlabel(column)
        plt.ylabel('Frecuencia')
        plt.show()

        media = np.mean(data[column])
        desv_est = np.std(data[column])
        minimo = np.min(data[column])
        maximo = np.max(data[column])
        mediana = np.median(data[column])
        p_25 = np.percentile(data[column], 25)
        p_75 = np.percentile(data[column], 75)

        print(f"Análisis estadístico de {column}:")
        print(f"Media: {media}")
        print(f"Desviación estándar: {desv_est}")
        print(f"Mínimo: {minimo}")
        print(f"Máximo: {maximo}")
        print(f"Mediana: {mediana}")
        print(f"Percentil 25: {p_25}")
        print(f"Percentil 75: {p_75}")
        print("-------------------------")

def data_report(df):
    # Sacamos los NOMBRES
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    # Sacamos los TIPOS
    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    # Sacamos los MISSINGS
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    # Sacamos los VALORES UNICOS
    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)


    return concatenado.T