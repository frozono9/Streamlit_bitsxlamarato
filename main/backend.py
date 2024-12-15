import numpy as np
import joblib
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import pandas as pd
import streamlit as st


 
# Cargar los modelos guardados
model_death = load_model("../local/models/model_death.h5")
model_transplant = load_model("../local/models/model_transplant.h5")
model_progressive = load_model("../local/models/model_progressive.h5")
 
# Cargar los escaladores guardados
death_scaler = joblib.load("../local/scalers/death_scaler.pkl")
transplant_scaler = joblib.load("../local/scalers/transplant_scaler.pkl")
progressive_scaler = joblib.load("../local/scalers/progressive_scaler.pkl")
 
# Mapear tareas a sus modelos y escaladores
task_map = {
    "death": (model_death, death_scaler),
    "transplant": (model_transplant, transplant_scaler),
    "progressive": (model_progressive, progressive_scaler)
}
 
# Función para predecir solo la probabilidad correspondiente a cada entrada
def predict_single_probabilities(input_data, tasks):
    """
    Recibe un conjunto de entradas (input_data) y una lista de tareas (tasks) que define
    qué modelo usar por cada entrada.
    - input_data: Lista o array con las características (de tamaño 15).
    - tasks: Lista de tareas que identifica el modelo a usar -> "death", "transplant", "progressive".
 
    Retorna las probabilidades correspondientes, una por tarea.
    """
    if len(input_data) != 15:
        raise ValueError(f"La entrada debe tener 15 características, pero tiene {len(input_data)}.")
 
    input_data = np.array(input_data).reshape(1, -1)  # Convertir a numpy array 2D
    probabilities = []
 
    for task in tasks:
        if task not in task_map:
            raise ValueError(f"Tarea desconocida: {task}. Debe ser 'death', 'transplant', o 'progressive'.")
 
        model, scaler = task_map[task]  # Obtener el modelo y el escalador correspondientes
        scaled_data = scaler.transform(input_data)  # Escalar la entrada
        prob = model.predict(scaled_data).flatten()[0]  # Obtener probabilidad
        probabilities.append(prob)
    
    return probabilities

def process_input_data(input_data):
    processed_data = []
    # q1, q2, q3, q4, q5
    q1 = input_data.get('q1', 0) # q1 FVC(%) al diagnosticar
    q2 = input_data.get('q2', 0) # q2 FVC(%) 1 año después
    q3 = input_data.get('q3', 0) # q3 DLCO(%) al diagnosticar
    q4 = input_data.get('q4', 0) # q4 DLCO(%) 1 año después
    q5 = input_data.get('q5', 0) # q5 Empitjorament radiologic
    
    absFVC = q2 - q1
    absDLCO = q4 - q3

    if absDLCO <= -10 or absFVC <= -5 or q5 == 'Si':
        fibrosis_progression = 1
    else:
        fibrosis_progression = 0

    # q6, q7
    q6 = input_data.get('q6', 0) # q6 Ha fumat alguna vegada?
    q7 = input_data.get('q7', 0) # q7 Segueix fumant actualment?
    
    if q6 == 'Si':
        if q7 == 'Si':
            smoking_status = 1
        else:
            smoking_status = 2
    else:
        smoking_status = 0

    # q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18
    q8 = 1 if input_data.get('q8', 'No') == 'Si' else 0  # q8 Agregacio familiar (Si/No)
    q9 = 1 if input_data.get('q9', 'No') == 'Si' else 0  # q9 Comorbiditat (Si/No)
    q10 = 1 if input_data.get('q10', 'Hombre') == 'Hombre' else 0  # q10 Sexe (Hombre/Mujer)
    q11_age = input_data.get('q11', 0)  # q11 Edat (0-100)
    if q11_age < 60:
        q11 = 0
    elif 60 <= q11_age < 70:
        q11 = 1
    elif 70 <= q11_age < 80:
        q11 = 2
    else:
        q11 = 3

    q12 = 1 if input_data.get('q12', 'No') == 'Si' else 0  # q12 Mutacio genetica (Si/No)

    # for uip: q21, q22, q23, q24, q25, q26

    q21 = input_data.get('q21', 'No hi ha biopsia')  # q21 Biopsia (No hi ha biopsia/Es criobiopsia endoscopica/Quirurgica)
    q22 = input_data.get('q22', 'No')  # q22 Radiologia (UIP/Probable UIP/Non UIP)
    q23 = input_data.get('q23', 'No')  # q23 Patologia CHP (UIP/Probable UIP/CHP)
    q24 = input_data.get('q24', 'No')  # q24 Patologia binaria (UIP/NON UIP)
    q25 = input_data.get('q25', 'No')  # q25 Detall no UIP (Fibrosing Organizing Pneumonia/CHP/Sarcoidosis IV)
    q26 = input_data.get('q26', 'No')  # q26 Patologia pattern (UIP/Probable UIP/CHP/NSIP/AFOP)

    uip = 0

    # Definir la presencia del patrón UIP basado en las reglas
    if (
        q21 in ['UIP', 'Probable UIP'] or 
        q23 in ['UIP', 'Probable UIP'] or 
        q24 == 'UIP' or 
        q26 in ['UIP', 'Probable UIP']
    ):
        uip = 1  # UIP presente
    elif (
        q21 == 'Non UIP' or 
        q23 == 'CHP' or 
        q24 == 'NON UIP' or 
        q22 in ['Fibrosing Organizing Pneumonia', 'CHP', 'Sarcoidosis IV'] or 
        q26 in ['CHP', 'NSIP', 'AFOP']
    ):
        uip = 0  # UIP ausente
    elif q25 in ["No hi ha biopsia"]:
        uip = 0  # Ausencia confirmada por biopsia negativa o no realizada
    elif q25 in ["Es criobiopsia endoscopica", "Quirurgica"]:
        uip = 1  # Ausencia confirmada por biopsia negativa o no realizada

    # for mpid type: q19, q20

    mutation_type = input_data.get('q19', '')  # q19 Tipo de mutación
    severity_of_telomere_shortening = int(input_data.get('q20', 0))  # q20 Severidad del acortamiento telomérico

    mpid = 0

    genetic_score = 0
    mutation_types = {
        'TERT': 5,
        'TERC': 5,
        'PARN': 5,
        'DKC1': 5,
        'RTEL1': 5
    }
    if mutation_type in mutation_types:
        genetic_score = mutation_types[mutation_type]

    telomere_score = 0
    if severity_of_telomere_shortening >= 5:
        telomere_score = 4
    elif 3 <= severity_of_telomere_shortening <= 4:
        telomere_score = 2
    elif severity_of_telomere_shortening <= 2:
        telomere_score = 1

    score = genetic_score + telomere_score

    if score >= 8:
        mpid = 2
    elif 4 <= score <= 7:
        mpid = 1
    else:
        mpid = 0

    q13 = 1 if input_data.get('q13', 'No') == 'Si' else 0  # q13 Aguditzacions (Si/No)
    q14 = 1 if input_data.get('q14', 'No') == 'Si' else 0  # q14 Tractament antifibrotic (Si/No)
    q15 = 1 if input_data.get('q15', 'No') == 'Si' else 0  # q15 Cancer de pulmo associat (Si/No)
    q16 = 1 if input_data.get('q16', 'No') == 'Si' else 0  # q16 Altres cancers (Si/No)
    q17 = 1 if input_data.get('q17', 'No') == 'Si' else 0  # q17 Malaltia hepatica (Si/No)
    q18 = 1 if input_data.get('q18', 'No') == 'Si' else 0  # q18 Malaltia hematologica (Si/No)

    processed_data.extend([fibrosis_progression, smoking_status, q8, q9, q10, q11, q12, uip, mpid, q13, q14, q15, q16, q17, q18])
    
    if len(processed_data) != 15:
        raise ValueError(f"El diccionario de entrada debe tener 15 características, pero tiene {len(processed_data)}.")
    
    #processed_data = np.array(processed_data).reshape(1, -1)
    return processed_data
 
 #shap
def get_importances(input_data, explainer_path):
    input_data = np.array(input_data).reshape(1, -1)
    explainer = joblib.load(explainer_path)
    shap_values = explainer.shap_values(input_data)
    mean_shap_values = np.mean(np.abs(shap_values), axis=0).flatten()
    
    return mean_shap_values

def plot_shap(mean_abs_shap_values, task):
    shap_importance_df = pd.DataFrame({
        'Features': ['Fibrosis_progression', 'TOBBACO', 'familiar_aggregation', 'Comorbidities', 'Sex', 'Age', 'Genetic_Mutation', 'UIP', 'MPID', 'Aguditzacions', 'Antifibrotic', 'Associated_Lung_Caner', 'Other Cancer', 'Liver Disease', 'Hematologic Disease' ],  # Exclude the target column
        'Mean Absolute SHAP Value': mean_abs_shap_values
    })

    # Sort the DataFrame by SHAP importance
    shap_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False, inplace=True)

    # Configurar una fuente profesional y minimalista
    font_path = font_manager.findfont(font_manager.FontProperties(family='Arial'))  # Puedes cambiar 'Arial' por otra fuente disponible
    title_font = font_manager.FontProperties(fname=font_path, size=16, weight='bold')
    label_font = font_manager.FontProperties(fname=font_path, size=12)
    annot_font = font_manager.FontProperties(fname=font_path, size=10)

    # Definir la paleta de colores (gradiente de rosas, lilas y azules)
    cmap = sns.light_palette((280, 80, 60), input="husl", as_cmap=True)
    print('mean_values', mean_abs_shap_values)
    # Crear la gráfica de calor
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        shap_importance_df.set_index('Features').T, 
        annot=True, 
        cmap=cmap, 
        annot_kws={'fontproperties': annot_font},
        fmt='.2f',
        cbar_kws={'label': 'SHAP Value'}
    )

    # Configurar el título y etiquetas con la fuente personalizada
    plt.title(f'Mean Absolute SHAP Values for {task.capitalize()} Model Features', fontproperties=title_font)
    plt.xlabel('Features', fontproperties=label_font)
    plt.ylabel('Samples', fontproperties=label_font)

    # Mostrar la gráfica
    plt.tight_layout()
    st.pyplot(plt)

    

# # Ejemplo de entrada: Un array con 15 características
# input_data = [1, 1, 0, 0, 1, 2, 0, 1, 1, 1, 0, 0, 0, 0, 0]  # Para predecir "death", "transplant" y "progressive"
 
# # Lista de tareas correspondientes a la entrada
# tasks = ["death", "transplant", "progressive"]
 
# # Predecir probabilidades para cada tarea
# try:
#     probabilities = predict_single_probabilities(input_data, tasks)
 
#     # Mostrar los resultados
#     for i, task in enumerate(tasks):
#         print(f"Probabilidad para '{task}': {probabilities[i]}")
 
# except Exception as e:
#     print(f"Error durante la predicción: {e}")
 