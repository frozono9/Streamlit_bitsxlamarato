%pip install joblib
%pip install streamlit
%pip install numpy
%pip install pandas
%pip install joblib
%pip install tensorflow
%pip install matplotlib
%pip install seaborn
%pip install scikit-learn
%pip install shap
%pip install json
%pip install toPDF

import streamlit as st
from backend import process_input_data, predict_single_probabilities, get_importances, plot_shap
import json
from toPDF import MedicalReport
import matplotlib.pyplot as plt
from predict_probabilites import process_dataframe
import pandas as pd


# Define the questions and possible options for dropdowns
questions = {
    "q1": "FVC(%) al diagnosticar?",
    "q2": "FVC(%) 1 año despues?",
    "q3": "DLCO(%) al diagnosticar?",
    "q4": "DLCO(%) 1 año despues?",
    "q5": "Empitjorament radiologic de la fibrosi?",

    "q6": "Ha fumado en un pasado?",
    "q7": "Sigue fumando?",

    "q8": "Algún familiar con enfermedad pulmonar intersticial?",

    "q9": "Tiene alguna comorbilidad?",

    "q10": "Cual es el sexo del paciente?",

    "q11": "Que edad tenías al diagnosticar?",

    "q12": "El paciente presenta alguna mutación genética?",

    "q13": "Tiene agudizaciones?",

    "q14": "Esta tomando un medicamento antifibrótico?",

    "q15": "Tiene cáncer de pulmón asociado?",

    "q16": "Tiene algún otro cáncer?",

    "q17": "Tiene alguna enfermedad hepática?",

    "q18": "Tiene enfermedad hematológica?",

    "q19": "Cuál es el tipo de mutación?",
    "q20": "Cuál es la gravedad del acortamiento de los telómeros?",
    
    "q21": "Cuál es el patrón radiológico?",
    "q22": "Detalle sobre NO UIP?",
    "q23": "Cuál es el patrón de patología UIP, probable o CHP?",
    "q24": "Cuál es el patrón de patología binario?",
    "q25": "Biopsia hecha?",
    "q26": "Cuál es el patrón de patología?"
}

# Predefined options for dropdown menus
options = {
    "SiNo": ["No", "Si"],
    "Sex": ["Hombre", "Mujer"],
    "Severity": list(range(0, 10)),  # 0 to 10 for percentages or severity
    "Mutations": ["None", "TERT", "TERC", "PARN", "DKC1", "RTEL1"],
    "UIP Patterns": ["Non UIP", "UIP", "Probable UIP", "CHP"],
    "Details": ["Fibrosing Organizing Pneumonia", "CHP", "Sarcoidosis IV", "NSIP", "AFOP"],
    "Biopsy": ["No hi ha biopsia", "Es criobiopsia endoscopica", "Quirurgica"],
    "Age": list(range(0, 101)),
    "Pathology Pattern": ['CHP', 'NSIP', 'AFOP']
}

# Set up the Streamlit app layout
st.set_page_config(page_title="Patient Data Collection", layout="wide")
st.title("Recopilación de Datos de Pacientes para Enfermedades Pulmonares Intersticiales Fibrosantes")
st.markdown("""
Esta herramienta ayuda a recopilar y analizar datos de pacientes para el modelado predictivo de **Enfermedades Pulmonares Intersticiales Fibrosantes**.
Complete el formulario a continuación y haga clic en **Enviar** para ver los resultados.
""")

# Responses list
responses = {}
# Collect responses from the user

# Create tabs for each section
tab1, tab2, tab3, tab4, tab5, importTab, exportTab = st.tabs(["Datos generales", "Estilo de vida y antecedentes familiares", "Condiciones médicas y comorbilidades", "Genética y tratamientos", "Resultados radiológicos y patológicos", "Importar database", "Exportar informe"])

# Sección 1: Datos generales
with tab1:
    st.subheader("📋 Datos generales del paciente")
    responses["q10"] = st.selectbox(questions["q10"], options["Sex"])
    responses["q11"] = st.number_input(questions["q11"], min_value=0, max_value=100, value=0)
    responses["q1"] = int(st.number_input(questions["q1"], min_value=0.0, max_value=200.0, value=0.0))
    responses["q2"] = int(st.number_input(questions["q2"], min_value=0.0, max_value=200.0, value=0.0))
    responses["q3"] = int(st.number_input(questions["q3"], min_value=0.0, max_value=200.0, value=0.0))
    responses["q4"] = int(st.number_input(questions["q4"], min_value=0.0, max_value=200.0, value=0.0))

# Sección 2: Estilo de vida y antecedentes familiares
with tab2:
    st.subheader("🚬 Estilo de vida y antecedentes familiares")
    responses["q6"] = st.selectbox(questions["q6"], options["SiNo"])
    responses["q7"] = st.selectbox(questions["q7"], options["SiNo"])
    responses["q8"] = st.selectbox(questions["q8"], options["SiNo"])

# Sección 3: Condiciones médicas y comorbilidades
with tab3:
    st.subheader("🏥 Condiciones médicas y comorbilidades")
    responses["q9"] = st.selectbox(questions["q9"], options["SiNo"])
    responses["q13"] = st.selectbox(questions["q13"], options["SiNo"])
    responses["q15"] = st.selectbox(questions["q15"], options["SiNo"])
    responses["q16"] = st.selectbox(questions["q16"], options["SiNo"])
    responses["q17"] = st.selectbox(questions["q17"], options["SiNo"])
    responses["q18"] = st.selectbox(questions["q18"], options["SiNo"])

# Sección 4: Genética y tratamientos
with tab4:
    st.subheader("🧬 Genética y tratamientos")
    responses["q12"] = st.selectbox(questions["q12"], options["SiNo"])
    responses["q19"] = st.selectbox(questions["q19"], options["Mutations"])
    responses["q20"] = st.selectbox(questions["q20"], options["Severity"])
    responses["q14"] = st.selectbox(questions["q14"], options["SiNo"])

# Sección 5: Resultados radiológicos y patológicos
with tab5:
    st.subheader("🔬 Resultados radiológicos y patológicos")
    responses["q5"] = st.selectbox(questions["q5"], options["SiNo"])
    responses["q21"] = st.selectbox(questions["q21"], options["UIP Patterns"])
    responses["q22"] = st.selectbox(questions["q22"], options["Details"])
    responses["q23"] = st.selectbox(questions["q23"], options["UIP Patterns"])
    responses["q24"] = st.selectbox(questions["q24"], options["SiNo"])
    responses["q25"] = st.selectbox(questions["q25"], options["Biopsy"])
    responses["q26"] = st.selectbox(questions["q26"], options["Pathology Pattern"])

# Sección 6: Importar base de datos
with importTab:
    st.subheader("🔬 Importar datos de la base de datos")

    # Cargar el archivo CSV
    import_file = st.file_uploader("Subir archivo CSV", type=["csv"])
    if import_file is not None:
        # Procesar los datos una sola vez y almacenarlos en session_state
        if "imported_data" not in st.session_state:
            st.session_state.import_file_name = import_file.name
            st.session_state.import_data = pd.read_csv(import_file)
            with st.spinner('Procesando datos...'):
                st.session_state.imported_data = process_dataframe(st.session_state.import_data)
            # Guardar el archivo procesado para su descarga
            st.session_state.imported_data.to_csv("imported_data.csv", index=False)

        # Mostrar los datos procesados
        st.write("Datos procesados:")
        st.dataframe(st.session_state.imported_data)

        # Botón para descargar los datos
        with open("imported_data.csv", "rb") as file:
            st.download_button(
                label="Descargar datos importados",
                data=file,
                file_name=f"imported_{st.session_state.import_file_name}",
                mime="text/csv",
            )


        # if st.button("Guardar datos importados como CSV"):
        #     importedData.to_csv("imported_data.csv", index=False)
        #     st.success("Datos importados guardados como imported_data.csv")
    

# Sección 7: Exportar informe
with exportTab:
    st.subheader("📄 Exportar informe")
    col1, col2 = st.columns(2)
    with col1:
        responses["nombre"] = st.text_input("Nombre del paciente")
    with col2:
        responses["apellidos"] = st.text_input("Apellidos del paciente")
    responses["identificador"] = st.text_input("Identificador del paciente")

    if st.button("Procesar informe"):
        # Generate PDF
        pdf = MedicalReport()
        pdf.add_page()

        # Data for the report
        infoPersonal = [
            ("Sexo", responses["q10"], "[Hombre/Mujer]", ""),
            ("Edad de Diagnosis", responses["q11"], "[0 - 100]", ""),
            ("Fumador pasado?", responses["q6"], "[Si/No]", ""),
            ("Fumador actual?", responses["q7"], "[Si/No]", ""),
            ("Familiar con IPF", responses["q8"], "[Si/No]", "")
        ]

        # Add patient information
        pdf.add_patient_info(name=(responses["nombre"] + " " + responses["apellidos"]), id = responses["identificador"])

        conMedicas = [
            ("Comorbilidad?", responses["q9"], "[Si/No]", ""),
            ("Cáncer Pulmonar?", responses["q15"], "[Si/No]", ""),
            ("Otros tipos de cáncer?", responses["q16"], "[Si/No]", ""),
            ("Enfermedad hepática?", responses["q17"], "[Si/No]", ""),
            ("Enfermedad hematológica?", responses["q18"], "[Si/No]", ""),
            ("Mutaciones genéticas?", responses["q12"], "[Si/No]", ""),
            ("Agudizaciones?", responses["q13"], "[Si/No]", "")
        ]

        processed_data = process_input_data(responses)

        severity_mapping = {0: "Leve", 1: "Moderada", 2: "Severa"}
        mpid = severity_mapping.get(processed_data[8], "Unknown")

        muerte = "Unknown"
        transpl = "Unknown"
        progresion = "Unknown"
        tasks = ["death", "transplant", "progressive"]
        
        try:
            probabilities = predict_single_probabilities(processed_data, tasks)
            for task in tasks:
                if task == "death":
                    muerte = f"{probabilities[tasks.index(task)]*100:.1f}%"
                elif task == "transplant":
                    transpl = f"{probabilities[tasks.index(task)]*100:.1f}%"
                elif task == "progressive":
                    progresion = f"{probabilities[tasks.index(task)]*100:.1f}%"
 
        except Exception as e:
            print(f"Error durante la predicción: {e}")

 
        calculistica = [
            ("Progresión de la Fibrosis", 'Si' if processed_data[0] == 1 else 'No', "[Si/No]", ""),
            ("Existe Patrón UIP?", 'Si' if processed_data[7] == 1 else 'No', "[Si/No]", ""),
            ("Tipo MPID", mpid, "[Severa, Moderada, Leve]", ""),
        ]

        predicciones = [
            ("Muerte", muerte, "[0.0 - 100.0 %]", ""),
            ("Transplante", transpl, "[0.0 - 100.0 %]", ""),
            ("Progresión IPF", progresion, "[0.0 - 100.0 %]", ""),
        ]
        

        # Add sections
        pdf.ln(5)  # Add some space
        pdf.add_section("INFORMACION PERSONAL", infoPersonal)
        pdf.ln(5)  # Add some space
        pdf.add_section("CONDICIONES MEDICAS", conMedicas)
        pdf.ln(5)  # Add some space
        pdf.add_section("CALCULISTICA", calculistica)
        pdf.ln(5)  # Add some space
        pdf.add_section("PREDICCIONES", predicciones)

        # Output PDF
        pdf_output = pdf.output(dest='S').encode('latin1')
        st.download_button(
            label="Descargar informe médico",
            data=pdf_output,
            file_name=f"{responses['identificador']}_medical_report.pdf",
            mime="application/pdf"
        )
        print("PDF generated successfully!")

# for key, question in questions.items():
#     if key in ["q1", "q2", "q3", "q4"]:  # These questions expect a percentage
#         responses[key] = int(st.number_input(question, min_value=0.0, max_value=200.0, value=0.0))
#     elif key in ["q5", "q6", "q7", "q8", "q9", "q12", "q13", "q14", "q15", "q16", "q17", "q18", "q24", "q25"]:  # Yes/No questions
#         responses[key] = st.selectbox(question, options["SiNo"])
#     elif key == "q10":  # Sex of the patient
#         responses[key] = st.selectbox(question, options["Sex"])
#     elif key == "q11":  # Age at diagnosis
#         responses[key] = st.number_input(question, min_value=0, max_value=100, value=0)
#     elif key == "q19":  # Type of mutation
#         responses[key] = st.selectbox(question, options["Mutations"])
#     elif key == "q20":  # Severity of telomere shortening
#         responses[key] = st.selectbox(question, options["Severity"])
#     elif key in ["q21", "q23"]:  # Radiological pattern
#         responses[key] = st.selectbox(question, options["UIP Patterns"])
#     elif key == "q22":  # Details about NO UIP
#         responses[key] = st.selectbox(question, options["Details"])
#     elif key == "q26":  # Pathology pattern
#         responses[key] = st.selectbox(question, options["Pathology Pattern"])

# Submit button with results display

# Lista de tareas correspondientes a la entrada
tasks = ["death", "transplant", "progressive"]
thresholds = {}
cols = st.columns(len(tasks))
for idx, task in enumerate(tasks):
    with cols[idx]:
        thresholds[task] = st.slider(f"{task.capitalize()} Threshold (%)", 0, 100, 50, key=f"slider_{task}")

#st.markdown("---")
if st.button("Enviar"):
    st.markdown("---")
    st.subheader("📊 Resultados:")
    processed_data = process_input_data(responses)
    print(processed_data)
 
    # Predecir probabilidades para cada tarea
    try:
        probabilities = predict_single_probabilities(processed_data, tasks)
        colsResults = st.columns(len(tasks))
        # Mostrar los resultados
        for task in tasks:

            with colsResults[tasks.index(task)]:
                #st.write(f"Probabilidad para '{task}': {probabilities[tasks.index(task)]}")
                st.markdown(f"**{task.capitalize()}**")
                result_binary = "Yes" if probabilities[tasks.index(task)] >= thresholds[task] / 100 else "No"
                st.markdown(f"Result: **{result_binary}**")
                st.markdown(f"Probability: {probabilities[tasks.index(task)]*100:.2f}%")
                
                # Determine bar color based on threshold
                bar_color = '#d73027' if probabilities[tasks.index(task)] >= thresholds[task] / 100 else '#2c7fb8'
                
                # Create compact horizontal bar chart with percentage indicator
                fig, ax = plt.subplots(figsize=(4, 0.4))  # Compact size
                ax.barh([task], [probabilities[tasks.index(task)]], color=bar_color, height=0.3)
                # Fill the empty bar space with a border
                ax.barh([task], [1], color='none', edgecolor='black', height=0.3, linewidth=0.5)
                ax.set_xlim(0, 1)
                ax.axvline(thresholds[task] / 100, color='grey', linestyle='--', linewidth=1)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.text(probabilities[tasks.index(task)] + 0.02, 0, f"{probabilities[tasks.index(task)]*100:.2f}%", 
                        va='center', ha='left', fontsize=10, color=bar_color)
                st.pyplot(fig)

        st.markdown("---")
        st.subheader("🔭 Analiticas del modelo:")

        with st.spinner('Procesando datos...'):
            for task in tasks:
                if task == "death":
                    explainer_path = '../local/explainers/explainer_death.pkl'
                elif task == "transplant":
                    explainer_path = '../local/explainers/explainer_transplant.pkl'
                elif task == "progressive":
                    explainer_path = '../local/explainers/explainer_progressive.pkl'
                importances = get_importances(processed_data, explainer_path)
                plot_shap(importances, task)
            
    except Exception as e:
        print(f"Error durante la predicción: {e}")

# Add a footer
st.markdown("""
---
Made by Porks in Paris.
""")
