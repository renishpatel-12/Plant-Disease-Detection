# Importing Necessary Libraries
import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results
import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import pandas as pd

st.set_page_config(
    page_title='Plant Disease Detection',
    page_icon='🍃',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ============ LANGUAGE DICTIONARY ============
LANGUAGES = {
    'en': {
        'title': '🍃 Plant Disease Detection',
        'subtitle': 'Upload a leaf image and get instant disease prediction powered by an ensemble of deep learning models. Get confidence scores, disease details, and treatment recommendations.',
        'badge': '⚡ Fast • 🎯 Accurate • 📊 Detailed',
        'model_type': 'Model Type',
        'architecture': 'Architecture',
        'classes': 'Classes',
        'upload_image': '📸 Upload Your Leaf Image',
        'batch_upload': '📁 Batch Upload (Multiple Images)',
        'upload_single': 'Choose a plant leaf image',
        'upload_batch': 'Upload multiple leaf images at once',
        'analyze': '🔍 Analyze Image',
        'analyze_batch': '🔍 Analyze All',
        'reset': '🔄 Reset',
        'how_it_works': '⚙️ How It Works',
        'disease_classes': '🌿 Disease Classes',
        'pro_tips': '💡 Pro Tips',
        'upload_placeholder': '📌 Choose a clear leaf image to begin the prediction.',
        'batch_placeholder': '📌 Upload multiple images to analyze them all at once.',
        'supported': 'Supported formats: PNG, JPG, JPEG',
        'recommended': 'Recommended: Well-lit, clear photos of individual leaves',
        'analyzing': '🤖 Analyzing your leaf...',
        'analyzing_batch': '🤖 Analyzing all images...',
        'low_confidence': '⚠️ Low Confidence',
        'high_confidence': '✅ High Confidence',
        'symptoms': '🔍 Symptoms',
        'treatment': '💊 Recommended Treatment',
        'all_probabilities': '📊 All Probabilities',
        'learn_more': '📚 Learn More About Plant Diseases',
        'healthy_desc': 'No disease symptoms. Green, uniform coloring.',
        'rust_desc': 'Fungal infection with orange-brown pustules.',
        'scab_desc': 'Dark, corky lesions on leaf surface.',
        'multiple_desc': 'Co-infection with multiple disease types.',
        'gallery': '📷 Recent Uploads Gallery',
        'download_report': '📥 Download Report as PDF',
        'generate_report': '📄 Generate Report',
        'analytics_dashboard': '📊 Analytics Dashboard',
        'export_csv': '📥 Export Predictions to CSV',
        'disease_prevalence': 'Disease Prevalence',
        'prediction_trends': 'Prediction Trends',
        'performance_metrics': 'Performance Metrics',
        'user_feedback': 'User Feedback',
        'thumbs_up': '👍 Helpful',
        'thumbs_down': '👎 Incorrect',
        'feedback_saved': 'Thank you — feedback recorded.',
        'analysis': 'Analysis',
    },
    'es': {
        'title': '🍃 Detección de Enfermedades de Plantas',
        'subtitle': 'Cargue una imagen de hoja y obtenga una predicción instantánea de enfermedades potenciada por modelos de aprendizaje profundo.',
        'badge': '⚡ Rápido • 🎯 Preciso • 📊 Detallado',
        'model_type': 'Tipo de Modelo',
        'architecture': 'Arquitectura',
        'classes': 'Clases',
        'upload_image': '📸 Cargue su Imagen de Hoja',
        'batch_upload': '📁 Carga por Lotes',
        'upload_single': 'Elija una imagen de hoja',
        'upload_batch': 'Cargue múltiples imágenes',
        'analyze': '🔍 Analizar',
        'analyze_batch': '🔍 Analizar Todo',
        'reset': '🔄 Restablecer',
        'how_it_works': '⚙️ Cómo Funciona',
        'disease_classes': '🌿 Clases de Enfermedades',
        'pro_tips': '💡 Consejos Pro',
        'upload_placeholder': '📌 Elija una imagen clara de hoja',
        'batch_placeholder': '📌 Cargue múltiples imágenes',
        'supported': 'Formatos: PNG, JPG, JPEG',
        'recommended': 'Recomendado: Fotos claras y bien iluminadas',
        'analyzing': '🤖 Analizando...',
        'analyzing_batch': '🤖 Analizando todas...',
        'low_confidence': '⚠️ Confianza Baja',
        'high_confidence': '✅ Confianza Alta',
        'symptoms': '🔍 Síntomas',
        'treatment': '💊 Tratamiento',
        'all_probabilities': '📊 Probabilidades',
        'learn_more': '📚 Aprendre Más',
        'healthy_desc': 'Sin síntomas de enfermedad.',
        'rust_desc': 'Infección fúngica con pústulas.',
        'scab_desc': 'Lesiones oscuras en la hoja.',
        'multiple_desc': 'Co-infección con múltiples tipos.',
        'gallery': '📷 Descargas Recientes',
        'download_report': '📥 Descargar Informe',
        'generate_report': '📄 Generar Informe',
        'analytics_dashboard': '📊 Panel de Análisis',
        'export_csv': '📥 Exportar Predicciones CSV',
        'disease_prevalence': 'Prevalencia de Enfermedades',
        'prediction_trends': 'Tendencias de Predicción',
        'performance_metrics': 'Métricas de Rendimiento',
        'user_feedback': 'Comentarios de Usuarios',
        'thumbs_up': '👍 Útil',
        'thumbs_down': '👎 Incorrecto',
        'feedback_saved': 'Gracias — comentario registrado.',
        'analysis': 'Análisis',
    },
    'fr': {
        'title': '🍃 Détection des Maladies des Plantes',
        'subtitle': 'Téléchargez une image de feuille et obtenez une prédiction instantanée des maladies.',
        'badge': '⚡ Rapide • 🎯 Précis • 📊 Détaillé',
        'model_type': 'Type de Modèle',
        'architecture': 'Architecture',
        'classes': 'Classes',
        'upload_image': '📸 Télécharger Votre Image',
        'batch_upload': '📁 Téléchargement par Lot',
        'upload_single': 'Choisissez une image de feuille',
        'upload_batch': 'Téléchargez plusieurs images',
        'analyze': '🔍 Analyser',
        'analyze_batch': '🔍 Analyser Tout',
        'reset': '🔄 Réinitialiser',
        'how_it_works': '⚙️ Comment Ça Marche',
        'disease_classes': '🌿 Classes de Maladies',
        'pro_tips': '💡 Conseils Pro',
        'upload_placeholder': '📌 Choisissez une image claire',
        'batch_placeholder': '📌 Téléchargez plusieurs images',
        'supported': 'Formats: PNG, JPG, JPEG',
        'recommended': 'Recommandé: Photos claires et bien éclairées',
        'analyzing': '🤖 Analyse en cours...',
        'analyzing_batch': '🤖 Analyse de toutes...',
        'low_confidence': '⚠️ Faible Confiance',
        'high_confidence': '✅ Confiance Élevée',
        'symptoms': '🔍 Symptômes',
        'treatment': '💊 Traitement',
        'all_probabilities': '📊 Probabilités',
        'learn_more': '📚 En Savoir Plus',
        'healthy_desc': 'Aucun symptôme de maladie.',
        'rust_desc': 'Infection fongique avec pustules.',
        'scab_desc': 'Lésions sombres sur la feuille.',
        'multiple_desc': 'Co-infection avec plusieurs types.',
        'gallery': '📷 Téléchargements Récents',
        'download_report': '📥 Télécharger le Rapport',
        'generate_report': '📄 Générer un Rapport',
        'analytics_dashboard': '📊 Tableau de Bord',
        'export_csv': '📥 Exporter les Prédictions en CSV',
        'disease_prevalence': 'Prévalence des Maladies',
        'prediction_trends': 'Tendances des Prédictions',
        'performance_metrics': 'Métriques de Performance',
        'user_feedback': 'Retour Utilisateur',
        'thumbs_up': '👍 Utile',
        'thumbs_down': '👎 Incorrect',
        'feedback_saved': 'Merci — retour enregistré.',
        'analysis': 'Analyse',
    }
}

# ============ INITIALIZE SESSION STATE ============
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'show_result' not in st.session_state:
    st.session_state.show_result = False
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'compare_prediction' not in st.session_state:
    st.session_state.compare_prediction = None
if 'upload_history' not in st.session_state:
    st.session_state.upload_history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []
if 'prediction_log' not in st.session_state:
    st.session_state.prediction_log = []
if 'last_feedback' not in st.session_state:
    st.session_state.last_feedback = None

def t(key):
    """Get translated text"""
    return LANGUAGES[st.session_state.language].get(key, key)


def export_predictions_csv():
    if not st.session_state.prediction_log:
        return None
    df = pd.DataFrame(st.session_state.prediction_log)
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    return csv_bytes


def summarize_feedback():
    if not st.session_state.prediction_log:
        return None
    df = pd.DataFrame(st.session_state.prediction_log)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    feedback_df = df[df['Feedback'].notnull()]
    summary = {
        'total_predictions': len(df),
        'average_confidence': df['Confidence'].mean(),
        'feedback_rate': len(feedback_df) / len(df) if len(df) else 0,
        'positive_feedback_rate': feedback_df['Feedback_Score'].mean() if not feedback_df.empty else None,
        'prevalence': df['Disease'].value_counts().to_dict(),
        'trend': df.set_index('Timestamp').resample('D').size().rename('Count').reset_index()
    }
    return summary

# ============ DISEASE INFORMATION ============
DISEASE_INFO = {
    0: {'name': 'Healthy', 'emoji': '✅', 'color': '#22c55e', 'threshold': 0.85},
    1: {'name': 'Multiple Diseases', 'emoji': '⚠️', 'color': '#f59e0b', 'threshold': 0.70},
    2: {'name': 'Rust', 'emoji': '🟠', 'color': '#fb923c', 'threshold': 0.75},
    3: {'name': 'Scab', 'emoji': '🔴', 'color': '#ef4444', 'threshold': 0.72}
}

def get_theme_css():
    """Generate CSS based on current theme"""
    if st.session_state.theme == 'dark':
        return """
        <style>
            :root { color-scheme: dark; }
            body { background: linear-gradient(135deg, #0f172a 0%, #111827 100%); color: #e2e8f0; }
            .stApp { background: transparent; }
            .title-text { font-size: clamp(2.4rem, 3vw, 4rem); font-weight: 800; letter-spacing: -0.03em; 
                         background: linear-gradient(135deg, #a7f3d0 0%, #34d399 100%); 
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
            .hero-card, .panel-card, .result-card, .upload-card, .info-card {
                         background: rgba(15, 23, 42, 0.92); border: 1px solid rgba(148, 163, 184, 0.16); 
                         border-radius: 30px; box-shadow: 0 40px 90px rgba(15, 23, 42, 0.35); padding: 32px; }
            .stButton>button { background: linear-gradient(135deg, #22c55e 0%, #06b6d4 100%); color: white;
                              border: none; border-radius: 999px; height: 48px; font-weight: 700; transition: all 0.3s; }
            .prediction-label { color: #34d399; font-size: 1.25rem; font-weight: 800; }
            .confidence-badge { background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.3);
                               padding: 8px 16px; border-radius: 999px; color: #34d399; font-weight: 700; }
            .warning-box { background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 16px; border-radius: 12px; color: #fca5a5; }
            .success-box { background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22c55e; padding: 16px; border-radius: 12px; color: #86efac; }
            .preview-box { border: 1px solid rgba(148, 163, 184, 0.2); padding: 12px; border-radius: 20px; background: rgba(15, 23, 42, 0.75); }
        </style>
        """
    else:
        return """
        <style>
            :root { color-scheme: light; }
            body { background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); color: #0f172a; }
            .stApp { background: transparent; }
            .title-text { font-size: clamp(2.4rem, 3vw, 4rem); font-weight: 800; letter-spacing: -0.03em; 
                         background: linear-gradient(135deg, #059669 0%, #10b981 100%); 
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
            .hero-card, .panel-card, .result-card, .upload-card, .info-card {
                         background: rgba(241, 245, 249, 0.95); border: 1px solid rgba(100, 116, 139, 0.2); 
                         border-radius: 30px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1); padding: 32px; }
            .stButton>button { background: linear-gradient(135deg, #22c55e 0%, #06b6d4 100%); color: white;
                              border: none; border-radius: 999px; height: 48px; font-weight: 700; transition: all 0.3s; }
            .prediction-label { color: #059669; font-size: 1.25rem; font-weight: 800; }
            .confidence-badge { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3);
                               padding: 8px 16px; border-radius: 999px; color: #059669; font-weight: 700; }
            .warning-box { background: rgba(239, 68, 68, 0.1); border-left: 4px solid #dc2626; padding: 16px; border-radius: 12px; color: #7f1d1d; }
            .success-box { background: rgba(34, 197, 94, 0.1); border-left: 4px solid #059669; padding: 16px; border-radius: 12px; color: #065f46; }
            .preview-box { border: 1px solid rgba(148, 163, 184, 0.2); padding: 12px; border-radius: 20px; background: rgba(255, 255, 255, 0.8); }
        </style>
        """


def preview_images(original_image):
    """Return original and resized input preview images."""
    resized_array = clean_image(original_image)
    resized_image = Image.fromarray(resized_array[0].astype(np.uint8))
    edge_image = ImageOps.grayscale(resized_image).filter(ImageFilter.FIND_EDGES)
    return original_image, resized_image, edge_image

@st.cache_resource
def load_model(path):
    xception = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    densenet = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    inputs = tf.keras.Input(shape=(512, 512, 3))
    outputs = tf.keras.layers.average([densenet(inputs), xception(inputs)])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.load_weights(path)
    return model

def generate_pdf_report(image_name, disease_name, confidence, probabilities):
    """Generate PDF report"""
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#22c55e'), spaceAfter=30, alignment=1)
    story.append(Paragraph('🍃 Plant Disease Detection Report', title_style))
    story.append(Spacer(1, 0.3*inch))
    
    info_data = [
        ['Report Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Image', image_name],
        ['Disease', disease_name],
        ['Confidence', confidence]
    ]
    
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph('<b>Probabilities</b>', styles['Heading2']))
    prob_data = [['Class', 'Probability']]
    for name, prob in zip(['Healthy', 'Multiple', 'Rust', 'Scab'], probabilities):
        prob_data.append([name, f'{prob*100:.1f}%'])
    
    prob_table = Table(prob_data, colWidths=[3*inch, 2*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#22c55e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.beige)
    ]))
    story.append(prob_table)
    
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

hide_streamlit = "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"
st.markdown(hide_streamlit, unsafe_allow_html=True)

# Apply dynamic theme CSS BEFORE loading model
st.markdown(get_theme_css(), unsafe_allow_html=True)

model = load_model('model.h5')

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown('### ⚙️ Settings')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('🇬🇧 EN', use_container_width=True):
            st.session_state.language = 'en'
            st.rerun()
    with col2:
        if st.button('🇪🇸 ES', use_container_width=True):
            st.session_state.language = 'es'
            st.rerun()
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button('🇫🇷 FR', use_container_width=True):
            st.session_state.language = 'fr'
            st.rerun()
    with col4:
        if st.button('🌙 Dark/Light', use_container_width=True):
            st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
            st.rerun()
    
    st.divider()
    
    if st.session_state.upload_history:
        st.markdown('### 📷 Recent Uploads')
        with st.expander('View Gallery', expanded=False):
            cols = st.columns(3)
            for idx, img_data in enumerate(st.session_state.upload_history[-9:]):
                with cols[idx % 3]:
                    st.image(img_data['image'], use_column_width=True)

# Apply dynamic theme CSS after sidebar (so it updates on theme change)
st.markdown(get_theme_css(), unsafe_allow_html=True)

# ============ HERO SECTION ============
with st.container():
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    left, right = st.columns([3, 1])
    with left:
        st.markdown(f'<div class="title-text">{t("title")}</div>', unsafe_allow_html=True)
        subtitle_color = '#cbd5e1' if st.session_state.theme == 'dark' else '#334155'
        st.markdown(f'<p style="color:{subtitle_color}; font-size:1.1rem;">{t("subtitle")}</p>', unsafe_allow_html=True)
    with right:
        card_bg = 'rgba(30, 41, 59, 0.9)' if st.session_state.theme == 'dark' else 'rgba(226, 232, 240, 0.9)'
        text_color = '#a7f3d0' if st.session_state.theme == 'dark' else '#059669'
        content_color = '#cbd5e1' if st.session_state.theme == 'dark' else '#64748b'
        st.markdown(f'<div style="background: {card_bg}; border-radius: 16px; padding: 16px; margin-bottom: 12px;"><h4 style="margin:0;color:{text_color};">Model</h4><p style="margin:4px 0 0; color:{content_color};">Ensemble</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background: {card_bg}; border-radius: 16px; padding: 16px;"><h4 style="margin:0;color:{text_color};">Languages</h4><p style="margin:4px 0 0; color:{content_color};">3 (EN/ES/FR)</p></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.write('')

# ============ TABS ============
tab1, tab2, tab3 = st.tabs(['📸 Single Image', '📁 Batch Upload', '📊 Analytics'])

# Tab 1: Single Image
with tab1:
    left_col, right_col = st.columns([2.2, 1])
    
    with left_col:
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.subheader(t('upload_image'))
        
        uploaded_file = st.file_uploader(t('upload_single'), type=['png', 'jpg', 'jpeg'], label_visibility='collapsed')
        camera_capture = st.camera_input('📷 Capture from camera', label_visibility='collapsed')

        image = None
        image_source = None
        if uploaded_file is not None:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            image_source = uploaded_file.name
        elif camera_capture is not None:
            image = Image.open(io.BytesIO(camera_capture.getvalue()))
            image_source = 'Camera capture'

        if image is not None:
            if image.size[0] < 100 or image.size[1] < 100:
                st.error('❌ Image too small (min 100x100)')
            else:
                st.markdown('### Preview')
                original_img, resized_img, edge_img = preview_images(image)
                preview_col1, preview_col2, preview_col3 = st.columns(3)
                with preview_col1:
                    st.image(original_img, caption='Original image', use_column_width=True)
                with preview_col2:
                    st.image(resized_img, caption='Resized input', use_column_width=True)
                with preview_col3:
                    st.image(edge_img, caption='Model feature view', use_column_width=True)

                st.info('The left image is your original upload. The middle image is the exact resized input the model uses, and the right image highlights edge detail the model pays attention to.')

                comparison_mode = st.checkbox('Enable comparison mode', key='comparison_mode')
                compare_result = None
                compare_image = None
                if comparison_mode:
                    st.markdown('### Comparison Mode')
                    compare_image_file = st.file_uploader('Upload comparison leaf image', type=['png', 'jpg', 'jpeg'], key='compare_file', label_visibility='collapsed')
                    if compare_image_file is not None:
                        compare_image = Image.open(io.BytesIO(compare_image_file.read()))
                        st.image(compare_image, caption='Comparison image', use_column_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    analyze_btn = st.button(t('analyze'), key='analyze_single', use_container_width=True)
                with col2:
                    reset_btn = st.button(t('reset'), key='reset_single', use_container_width=True)

                if reset_btn:
                    st.session_state.show_result = False
                    st.session_state.current_prediction = None
                    st.session_state.compare_prediction = None
                    st.rerun()

                if analyze_btn:
                    with st.spinner(t('analyzing')):
                        img_array = clean_image(image)
                        preds, preds_arr = get_prediction(model, img_array)
                        st.session_state.current_prediction = (preds, preds_arr)
                        st.session_state.current_image = image
                        st.session_state.show_result = True
                        disease_name = DISEASE_INFO[int(preds_arr)]['name']
                        confidence_pct = preds[0][int(preds_arr)] * 100
                        log_entry = {
                            'Timestamp': datetime.now(),
                            'Image': image_source,
                            'Disease': disease_name,
                            'Confidence': confidence_pct,
                            'Probability_Healthy': preds[0][0],
                            'Probability_Multiple': preds[0][1],
                            'Probability_Rust': preds[0][2],
                            'Probability_Scab': preds[0][3],
                            'Feedback': None,
                            'Feedback_Score': None
                        }
                        st.session_state.prediction_log.append(log_entry)
                        st.session_state.upload_history.append({
                            'image': image,
                            'name': image_source,
                            'timestamp': log_entry['Timestamp'],
                            'disease': disease_name,
                            'confidence': f'{confidence_pct:.1f}%'
                        })

                        if comparison_mode and compare_image is not None:
                            compare_array = clean_image(compare_image)
                            cmp_preds, cmp_preds_arr = get_prediction(model, compare_array)
                            st.session_state.compare_prediction = {
                                'preds': cmp_preds,
                                'preds_arr': cmp_preds_arr,
                                'image': compare_image,
                                'source': getattr(compare_image_file, 'name', 'Comparison image')
                            }

                if st.session_state.show_result and st.session_state.current_prediction:
                    preds, preds_arr = st.session_state.current_prediction
                    result = make_results(preds, preds_arr)
                    disease_class = int(preds_arr)
                    disease_info = DISEASE_INFO[disease_class]
                    confidence = preds[0][disease_class]

                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f'<p class="prediction-label">{disease_info["emoji"]} {t("analysis")}: {disease_info["name"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p>Confidence: <span class="confidence-badge">{result["prediction"]}</span></p>', unsafe_allow_html=True)

                    if confidence < disease_info['threshold']:
                        st.markdown(f'<div class="warning-box"><strong>{t("low_confidence")}</strong></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="success-box"><strong>{t("high_confidence")}</strong></div>', unsafe_allow_html=True)

                    with st.expander(t('symptoms'), expanded=True):
                        st.write('Disease detected with visible symptoms')

                    with st.expander(t('treatment'), expanded=True):
                        st.write('Consult a plant pathologist for proper treatment')

                    pdf = generate_pdf_report(image_source or 'Uploaded image', disease_info['name'], result['prediction'], preds[0])
                    st.download_button(label=t('download_report'), data=pdf, file_name=f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf', mime='application/pdf', use_container_width=True, key='download_pdf')

                    st.write('---')
                    st.subheader(t('all_probabilities'))
                    for label, prob in zip(['Healthy', 'Multiple', 'Rust', 'Scab'], preds[0]):
                        pct = prob * 100
                        st.write(f'**{label}** — {pct:.1f}%')
                        st.progress(float(prob))

                    st.write('---')
                    st.subheader(t('user_feedback'))
                    fb_col1, fb_col2 = st.columns(2)
                    if fb_col1.button(t('thumbs_up'), key='feedback_up'):
                        if st.session_state.prediction_log:
                            st.session_state.prediction_log[-1]['Feedback'] = t('thumbs_up')
                            st.session_state.prediction_log[-1]['Feedback_Score'] = 1
                            st.session_state.last_feedback = t('feedback_saved')
                    if fb_col2.button(t('thumbs_down'), key='feedback_down'):
                        if st.session_state.prediction_log:
                            st.session_state.prediction_log[-1]['Feedback'] = t('thumbs_down')
                            st.session_state.prediction_log[-1]['Feedback_Score'] = 0
                            st.session_state.last_feedback = t('feedback_saved')
                    if st.session_state.last_feedback:
                        st.success(st.session_state.last_feedback)

                    if comparison_mode and st.session_state.compare_prediction is not None:
                        st.write('---')
                        st.subheader('Comparison Results')
                        comp = st.session_state.compare_prediction
                        comp_preds = comp['preds']
                        comp_preds_arr = comp['preds_arr']
                        comp_disease = DISEASE_INFO[int(comp_preds_arr)]
                        comp_cols = st.columns(2)
                        with comp_cols[0]:
                            st.markdown('**Current image**')
                            st.image(image, use_column_width=True)
                            st.write(f'Prediction: {disease_info["name"]}')
                            st.write(f'Confidence: {result["prediction"]}')
                        with comp_cols[1]:
                            st.markdown('**Comparison image**')
                            st.image(comp['image'], use_column_width=True)
                            st.write(f'Prediction: {comp_disease["name"]}')
                            st.write(f'Confidence: {comp_preds[0][int(comp_preds_arr)]*100:.1f}%')

                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info(t('upload_placeholder'))
            st.write(t('supported'))
            st.write(t('recommended'))

        st.markdown('</div>', unsafe_allow_html=True)
    
    with right_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader(t('how_it_works'))
        st.write('1️⃣ Upload leaf image')
        st.write('2️⃣ Resize to 512×512')
        st.write('3️⃣ Ensemble analysis')
        st.write('4️⃣ Get probabilities')
        
        st.write('---')
        st.subheader(t('disease_classes'))
        for i, info in DISEASE_INFO.items():
            st.write(f'{info["emoji"]} {info["name"]}')
        
        st.write('---')
        st.subheader(t('pro_tips'))
        st.write('✅ Sharp, well-lit photos')
        st.write('✅ No shadows/reflections')
        st.write('✅ Center the leaf')
        st.write('✅ Minimal background')

        if st.session_state.upload_history:
            st.write('---')
            st.subheader('📜 Prediction History')
            history_items = [
                {'Timestamp': item['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                 'Image': item['name'],
                 'Disease': item['disease'],
                 'Confidence': item['confidence']}
                for item in st.session_state.upload_history[-8:]
            ]
            st.dataframe(pd.DataFrame(history_items), use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Batch Upload
with tab2:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.subheader(t('batch_upload'))
    
    batch_files = st.file_uploader(t('upload_batch'), type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    if batch_files:
        st.info(f'📊 {len(batch_files)} image(s) ready')
        
        if st.button(t('analyze_batch'), use_container_width=True):
            progress = st.progress(0)
            results = []
            
            for idx, file in enumerate(batch_files):
                try:
                    img = Image.open(io.BytesIO(file.read()))
                    if img.size[0] >= 100 and img.size[1] >= 100:
                        with st.spinner(f'🤖 {idx+1}/{len(batch_files)}...'):
                            arr = clean_image(img)
                            preds, preds_arr = get_prediction(model, arr)
                            disease_class = int(preds_arr)
                            disease_name = DISEASE_INFO[disease_class]['name']
                            confidence_pct = preds[0][disease_class] * 100
                            results.append({'Image': file.name, 'Disease': disease_name, 'Confidence': f'{confidence_pct:.1f}%'})
                            st.session_state.upload_history.append({'image': img, 'name': file.name, 'timestamp': datetime.now(), 'disease': disease_name, 'confidence': f'{confidence_pct:.1f}%'})
                            st.session_state.prediction_log.append({
                                'Timestamp': datetime.now(),
                                'Image': file.name,
                                'Disease': disease_name,
                                'Confidence': confidence_pct,
                                'Probability_Healthy': preds[0][0],
                                'Probability_Multiple': preds[0][1],
                                'Probability_Rust': preds[0][2],
                                'Probability_Scab': preds[0][3],
                                'Feedback': None,
                                'Feedback_Score': None
                            })
                    progress.progress((idx + 1) / len(batch_files))
                except:
                    pass
            
            if results:
                st.success(f'✅ {len(results)}/{ len(batch_files)} analyzed')
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                cols = st.columns(min(3, len(results)))
                for idx, r in enumerate(results):
                    with cols[idx % 3]:
                        st.metric('Disease', r['Disease'])
    else:
        st.info(t('batch_placeholder'))
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Analytics
with tab3:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader(t('analytics_dashboard'))

    if st.session_state.prediction_log:
        analytics = summarize_feedback()
        df = pd.DataFrame(st.session_state.prediction_log)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        st.markdown('### ' + t('disease_prevalence'))
        prevalence = pd.DataFrame(list(analytics['prevalence'].items()), columns=['Disease', 'Count'])
        st.bar_chart(prevalence.rename(columns={'Disease': 'index'}).set_index('index'))

        st.markdown('### ' + t('prediction_trends'))
        trend_df = analytics['trend']
        trend_df.columns = ['Date', 'Predictions']
        trend_df = trend_df.sort_values('Date')
        st.line_chart(trend_df.rename(columns={'Date': 'index'}).set_index('index'))

        avg_confidence = analytics['average_confidence']
        positive_feedback_rate = analytics['positive_feedback_rate']
        feedback_rate = analytics['feedback_rate'] * 100

        st.markdown('### ' + t('performance_metrics'))
        perf_cols = st.columns(3)
        perf_cols[0].metric('Total Predictions', analytics['total_predictions'])
        perf_cols[1].metric('Avg. Confidence', f'{avg_confidence:.1f}%')
        perf_cols[2].metric('Feedback Coverage', f'{feedback_rate:.1f}%')

        if positive_feedback_rate is not None:
            st.metric('Positive Feedback Rate', f'{positive_feedback_rate * 100:.1f}%')
        else:
            st.info('No feedback yet. Use thumbs up/down in prediction results to improve this score.')

        st.write('---')
        st.subheader('Disease-by-Disease Metrics')
        metrics = []
        for disease, group in df.groupby('Disease'):
            feedback_group = group[group['Feedback_Score'].notnull()]
            metrics.append({
                'Disease': disease,
                'Predictions': len(group),
                'Avg Confidence': f"{group['Confidence'].mean():.1f}%",
                'Positive Feedback': f"{feedback_group['Feedback_Score'].mean() * 100:.1f}%" if not feedback_group.empty else 'N/A'
            })
        st.table(pd.DataFrame(metrics))

        st.write('---')
        st.subheader(t('export_csv'))
        csv_data = export_predictions_csv()
        if csv_data is not None:
            st.download_button(label=t('export_csv'), data=csv_data, file_name='prediction_history.csv', mime='text/csv', use_container_width=True)
    else:
        st.info('Upload and analyze at least one image to view analytics.')

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.write('---')
st.markdown('<div style="text-align:center; color:#64748b; padding:24px 0;">🔧 Streamlit • 🧠 TensorFlow • 📊 Ensemble Learning</div>', unsafe_allow_html=True)
