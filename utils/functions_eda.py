"""Modulo de Utils"""
import pandas as pd

def resumen_dataframe(df: pd.DataFrame, sample: bool = False) -> pd.DataFrame:
    """Generación de resumen descriptivo de las variables del DataFrame

    Args:
        df (pd.DataFrame): DataFrame original

    Returns:
        pd.DataFrame: DataFrame descriptivo
    """
    df_resume = pd.DataFrame()

    for variable in df.columns:
        variable_name = variable
        tipo_dato = df[variable].dtype
        registros_esperados = len(df)
        valores_nulos = int(df[variable].isnull().sum())
        unicos = len(df[variable].unique())
        ejemplos = list(df[variable].sample(3))

        row_resume = pd.DataFrame({
            'Variable': [variable_name],
            'Tipo_Dato': [tipo_dato],
            'Registros_Esperados': [registros_esperados],
            'Valores_Unicos': [unicos],
            'Valores_Nulos': [valores_nulos],
            '%Valores_Nulos': [round(valores_nulos / registros_esperados * 100, 2)],
            'Ejemplos': [ejemplos]
        })

        df_resume = pd.concat([df_resume, row_resume])

    if sample is False:
        df_resume = df_resume.drop('Ejemplos', axis = 1)

    return df_resume.reset_index(drop=True)
