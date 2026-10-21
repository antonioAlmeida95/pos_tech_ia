"""Configurações compartilhadas pelo notebook e pelos scripts do projeto."""
from pathlib import Path

RANDOM_STATE = 42

# Caminhos sempre relativos à raiz do repositório, de onde quer que o código seja executado
RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_CSV = RAIZ / "data" / "diabetes_012_health_indicators_BRFSS2015.csv"
PASTA_FIGURAS = RAIZ / "reports" / "figures"
PASTA_RESULTADOS = RAIZ / "reports"

# Grupos de variáveis (ver justificativa no notebook)
BINARIAS = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
            "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
            "NoDocbcCost", "DiffWalk", "Sex"]
ORDINAIS = ["GenHlth", "Age", "Education", "Income"]
CONTINUAS = ["BMI", "MentHlth", "PhysHlth"]
FEATURES = BINARIAS + ORDINAIS + CONTINUAS

ALVO_ORIGINAL = "Diabetes_012"
ALVO = "Diabetes_binary"

IMC_MAXIMO = 60          # valores acima são limitados (winsorização)
RECALL_ALVO = 0.90       # critério de escolha do limiar de decisão
