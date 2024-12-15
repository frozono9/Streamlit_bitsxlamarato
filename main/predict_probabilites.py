import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

def preprocess_input(df):
    print(type(df))
    # Preprocess input copying original columns that will be needed for the final prediction    
    # Ensure only existing columns are operated upon
    if 'COD NUMBER' in df.columns:
        df.drop(columns=['COD NUMBER'], inplace=True)

    for col in ['FVC (L) at diagnosis', 'FVC (L) 1 year after diagnosis']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    if 'FVC (%) 1 year after diagnosis' in df.columns and 'FVC (%) at diagnosis' in df.columns:
        df['FVC absolute'] = df['FVC (%) 1 year after diagnosis'] - df['FVC (%) at diagnosis']
        df['DLCO absolute'] = df['DLCO (%) 1 year after diagnosis'] - df['DLCO (%) at diagnosis']
        df.drop(columns=['FVC (%) 1 year after diagnosis', 'DLCO (%) 1 year after diagnosis', 'FVC (%) at diagnosis', 'DLCO (%) at diagnosis'], inplace=True)
    
    df['FVC binary'] = np.where(df.get('FVC absolute', 0) <= -5, 1, 0)
    df['DLCO binary'] = np.where(df.get('DLCO absolute', 0) <= -10, 1, 0)

    df.drop(columns=['FVC absolute', 'DLCO absolute'], inplace=True, errors='ignore')
    df['fibrosis progression'] = df[['FVC binary', 'DLCO binary', 'RadioWorsening2y']].any(axis=1).astype(int)

    drop_columns = ['FVC binary', 'DLCO binary', 'RadioWorsening2y', 'Detail', 'Severity of telomere shortening - Transform 4', 'Extra', 'Extras AP', 'Pedigree', '1st degree relative', '2nd degree relative', 'More than 1 relative', 'Type of family history']
    df.drop(columns=drop_columns, inplace=True, errors='ignore')
    
    if 'FamilialvsSporadic' in df.columns:
        df['familiar aggregation'] = df['FamilialvsSporadic'].apply(lambda x: 1 if x == 'Familial' else 0)
        df.drop(columns=['FamilialvsSporadic'], inplace=True)

    def determine_uip(row):
        radiological = row.get('Radiological Pattern', None)
        detail_non_uip = row.get('Detail on NON UIP', None)
        pathology_chp = row.get('Pathology pattern UIP, probable or CHP', None)
        pathology_binary = row.get('Pathology Pattern Binary', None)
        biopsy = row.get('Biopsy', None)
        pathology_pattern = row.get('Pathology pattern', None)
        
        if (
            radiological in ['UIP', 'Probable UIP'] or 
            pathology_chp in ['UIP', 'Probable UIP'] or 
            pathology_binary == 'UIP' or 
            pathology_pattern in ['UIP', 'Probable UIP']
        ):
            return 1
        elif (
            radiological == 'Non UIP' or 
            pathology_chp == 'CHP' or 
            pathology_binary == 'NON UIP' or 
            detail_non_uip in ['Fibrosing Organizing Pneumonia', 'CHP', 'Sarcoidosis IV'] or 
            pathology_pattern in ['CHP', 'NSIP', 'AFOP']
        ):
            return 0
        elif biopsy in [0, 2]:
            return 0
        return None

    df['UIP_Pattern'] = df.apply(determine_uip, axis=1)

    def create_mpid_score(df):
        df['mpid_score'] = 0
        df['genetic_score'] = 0
        mutation_types = {
            'TERT': 5,
            'TERC': 5,
            'PARN': 5,
            'DKC1': 5,
            'RTEL1': 5
        }
        for mutation, score in mutation_types.items():
            if 'Mutation Type' in df.columns:
                mask = df['Mutation Type'].str.contains(mutation, na=False)
                df.loc[mask, 'genetic_score'] = score

        df['extrapulmonary_score'] = 0
        if 'Extrapulmonary affectation' in df.columns:
            df.loc[df['Extrapulmonary affectation'] == 1, 'extrapulmonary_score'] = 3

        df['hematological_score'] = 0
        hematological_conditions = [
            'Anemia', 'Thrombocytopenia', 'Thrombocytosis',
            'Lymphocytosis', 'Lymphopenia', 'Neutrophilia',
            'Neutropenia', 'Leukocytosis', 'Leukopenia'
        ]
        for condition in hematological_conditions:
            if condition in df.columns:
                df.loc[df[condition] == 1, 'hematological_score'] += 2
        
        df['liver_score'] = 0
        if 'Liver abnormality' in df.columns:
            df.loc[df['Liver abnormality'] == 1, 'liver_score'] = 2

        df['telomere_score'] = 0        
        if 'Severity of telomere shortening' in df.columns:
            df.loc[df['Severity of telomere shortening'] >= 5, 'telomere_score'] = 4
            df.loc[df['Severity of telomere shortening'].between(3, 4), 'telomere_score'] = 2
            df.loc[df['Severity of telomere shortening'] <= 2, 'telomere_score'] = 1

        df['mpid_score'] = df['genetic_score'] + df['telomere_score']
    
        df['MPID_type'] = 'No MPID'
        df.loc[df['mpid_score'] >= 10, 'MPID_type'] = 'Severe MPID'
        df.loc[df['mpid_score'].between(5, 9), 'MPID_type'] = 'Moderate MPID'
        df.loc[df['mpid_score'].between(1, 4), 'MPID_type'] = 'Mild MPID'
    
        return df

    df = create_mpid_score(df)

    if 'Cause of death' in df.columns:
        df['aguditzacions'] = df['Cause of death'].apply(lambda x: 0 if x == 'No death' or pd.isna(x) else 1)
    df['Associated lung cancer'].fillna(0, inplace=True)
    df['Other cancer'].fillna(0, inplace=True)
    if 'Progressive disease' in df.columns:
        df['Progressive disease'].fillna(0, inplace=True)
    if 'ProgressiveDisease' in df.columns:
        df['ProgressiveDisease'].fillna(0, inplace=True)
    df['Genetic mutation studied in patient'].fillna(0, inplace=True)
    df['Mutation Type'].replace(-9, 0, inplace=True)
    if 'Death' in df.columns:
        df['Death'].fillna('No', inplace=True)
    
    if 'Liver disease' in df.columns:
        df['Liver disease'] = df['Liver disease'].apply(lambda x: 0 if x == 'No' else 1).fillna(0).astype(int)

    if 'Hematologic Disease' in df.columns:
        df['Hematologic Disease'] = df['Hematologic Disease'].apply(lambda x: 0 if x == 'No' else 1).fillna(0).astype(int)
    
    new_df = df[['fibrosis progression', 'TOBACCO', 'familiar aggregation', 'Comorbidities', 'Sex', 'Age at diagnosis', 
                 'Genetic mutation studied in patient',
                 'UIP_Pattern', 'MPID_type', 'aguditzacions', 'Antifibrotic Drug', 'Associated lung cancer', 
                 'Other cancer', 'Liver disease', 'Hematologic Disease']]
    
    if 'Age at diagnosis' in new_df.columns:
        new_df.dropna(subset=['Age at diagnosis'], inplace=True)

    label_encoder = LabelEncoder()
    columns_to_encode = new_df.columns.difference(['Age at diagnosis'])
    for column in columns_to_encode:
        new_df[column] = label_encoder.fit_transform(new_df[column])

    if 'Age at diagnosis' in new_df.columns:
        bins = [0, 60, 70, 80, 100]
        labels = [0, 1, 2, 3]
        new_df['Age at diagnosis'] = pd.cut(new_df['Age at diagnosis'], bins=bins, labels=labels, right=False)

    return new_df

# Cargar los modelos y los escaladores guardados
model_death = load_model("local/models/model_death.h5")
model_transplant = load_model("local/models/model_transplant.h5")
model_progressive = load_model("local/models/model_progressive.h5")

death_scaler = joblib.load("local/scalers/death_scaler.pkl")
transplant_scaler = joblib.load("local/scalers/transplant_scaler.pkl")
progressive_scaler = joblib.load("local/scalers/progressive_scaler.pkl")

tasks = ["death", "transplant", "progressive"]

def predict_single_probabilities(input_array, tasks):
    probabilities = []
    
    for input_data in input_array:
        row_probabilities = []
        for task in tasks:
            if task == "death":
                if len(input_data) != 15:
                    raise ValueError(f"Entrada para 'death' debe tener 15 características, pero tiene {len(input_data)}.")
                data = np.array(input_data).reshape(1, -1)
                data = death_scaler.transform(data)
                prob = model_death.predict(data).flatten()[0]
                row_probabilities.append(prob)
            elif task == "transplant":
                if len(input_data) != 15:
                    raise ValueError(f"Entrada para 'transplant' debe tener 15 características, pero tiene {len(input_data)}.")
                data = np.array(input_data).reshape(1, -1)
                data = transplant_scaler.transform(data)
                prob = model_transplant.predict(data).flatten()[0]
                row_probabilities.append(prob)
            elif task == "progressive":
                if len(input_data) != 15:
                    raise ValueError(f"Entrada para 'progressive' debe tener 15 características, pero tiene {len(input_data)}.")
                data = np.array(input_data).reshape(1, -1)
                data = progressive_scaler.transform(data)
                prob = model_progressive.predict(data).flatten()[0]
                row_probabilities.append(prob)
            else:
                raise ValueError(f"Tarea desconocida: {task}. Debe ser 'death', 'transplant', o 'progressive'.")
        probabilities.append(row_probabilities)
    return probabilities

def process_dataframe(df):
    # Create a copy of the DataFrame
    original_df = df.copy()
    #borrar target variables
    

    # Preprocess the entire DataFrame
    processed_df = preprocess_input(df)
    input_array = processed_df.values.tolist()

    # Get probabilities
    probabilities = predict_single_probabilities(input_array, tasks)
    # Append probabilities to the original DataFrame
    probabilities_df = pd.DataFrame(probabilities, columns=[f"{task}_probability" for task in tasks])
    result_df = original_df.reset_index(drop=True).join(probabilities_df)
    if 'Death' in result_df.columns:
        result_df.drop(columns=['Death'], inplace=True)
    if 'Progressive disease' in result_df.columns:
        result_df.drop(columns=['Progressive disease'], inplace=True)
    if 'ProgressiveDisease' in result_df.columns:
        result_df.drop(columns=['ProgressiveDisease'], inplace=True)
    if 'Necessity of transplantation' in result_df.columns:
        result_df.drop(columns=['Necessity of transplantation'], inplace=True)
    return result_df


## Ejemplo de worflow

# database_con_probabilidades = process_dataframe(raw_database_sin_probabilidades)
