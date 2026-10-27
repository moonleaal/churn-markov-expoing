# Churn Markov — ExpoIngenierías AD2026

Modelo de cadenas de Markov para cuantificar el **tiempo esperado hasta cancelación** y el **CLV (Customer Lifetime Value) probabilístico** de clientes de suscripción, con simulación de impacto económico de estrategias de retención.

Proyecto de MA2004B — Optimización estocástica, Tecnológico de Monterrey, Campus Querétaro.

## Equipo
- Montserrat Carrera Leal
- Celina Medina Bucio
- Daniela Iliana Rivera García
- Frida Azenette Hernández Illescas

## Estructura del repo

```
churn-markov-expoing/
├── data/
│   └── raw/              # dataset Telco Customer Churn (IBM)
├── notebooks/
│   ├── 01_eda.ipynb              # exploración de datos y definición de estados
│   ├── 02_markov_model.ipynb     # matriz de transición, clasificación de estados, probabilidades límite
│   └── 03_clv_simulation.ipynb   # CLV probabilístico y simulación de intervención
├── app/
│   └── dashboard.py       # demo interactiva en Streamlit
├── report/
│   └── reporte.tex         # reporte siguiendo la estructura de la guía del curso
└── video/
    └── guion.md             # guion del video de 5 min para ExpoIngenierías
```

## Cómo correr el proyecto

1. Instalar dependencias: `pip install -r requirements.txt`
2. Correr los notebooks en orden (01 → 02 → 03)
3. Levantar el dashboard: `streamlit run app/dashboard.py`

## Dataset

[Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 clientes de una empresa de telecomunicaciones, con variables de contrato, servicios, antigüedad (`tenure`) y cancelación (`Churn`).
