import pandas as pd
from IPython.core.display import display, HTML

def exibir_tabela_estilizada(df):
    estilo = """
    <style>
        table {
            width: 80%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 18px;
            text-align: left;
        }
        th, td {
            padding: 12px;
            border: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
    """
    html = df.to_html(index=False, escape=False)
    display(HTML(estilo + html))

# Criando um dataframe de exemplo
data = {
    "Nome": ["Ana", "Bruno", "Carlos"],
    "Idade": [25, 30, 22],
    "Cidade": ["São Paulo", "Rio de Janeiro", "Belo Horizonte"]
}
df = pd.DataFrame(data)

# Exibir a tabela estilizada
exibir_tabela_estilizada(df)