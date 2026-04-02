# 🏥 Sistema Predictivo de Inasistencias 

Esta aplicación es una herramienta de gestión clínica diseñada para reducir el ausentismo en la atención primaria. Utiliza un modelo de **Machine Learning** para analizar agendas médicas en tiempo real y priorizar el contacto con los pacientes que tienen mayor probabilidad de no asistir.

## 🛠️ Capacidades Técnicas

- **Carga de Agendas:** Procesamiento de archivos CSV extraídos directamente de los sistemas de registro clínico.
- **Motor de Predicción:** Implementación de un modelo *Random Forest* optimizado para detectar patrones de inasistencia sociodemográficos.
- **Categorización de Riesgo:** 
  - 🔴 **Extremo:** Intervención inmediata vía llamada telefónica.
  - 🟡 **Alto:** Gestión preventiva vía mensajería.
  - 🟢 **Bajo:** Flujo normal de atención.
- **Generación de Reportes:** Exportación de listas de trabajo en formato Excel con formato visual para el equipo evaluador.

## 📁 Estructura del Proyecto

- `app_cholchol.py`: Aplicación web interactiva (Streamlit).
- `modelo_rf_agenda_cholchol.pkl`: Modelo entrenado (necesario para el funcionamiento de la app).
- `requirements.txt`: Dependencias del proyecto.
- `image.png`: Icono de la aplicación.

## 🚀 Instalación Local

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/antinaosDev/machinelearning_agend.git
   cd machinelearning_agend
   ```

2. Crear un entorno virtual e instalar dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Ejecutar la aplicación:
   ```bash
   streamlit run app_cholchol.py
   ```

## ☁️ Despliegue en Streamlit Cloud

1. Sube este repositorio a GitHub.
2. Ve a [Streamlit Cloud](https://streamlit.io/cloud).
3. Conecta tu repositorio de GitHub.
4. Configura los secrets si es necesario (no requiere secrets para esta app).
5. Deploy.

## 📊 Uso de la Aplicación

1. **Carga Masiva**: Sube un archivo CSV o Excel con la agenda de pacientes.
2. El sistema analizará cada registro y asignará una probabilidad de inasistencia.
3. Descarga la lista de rescate con las recomendaciones de acción.

## ⚠️ Notas

- Los datos se procesan localmente; no se almacenan en la nube.
- El modelo fue entrenado con datos del CESFAM Cholchol.
- Precisión del modelo: ~85.45%