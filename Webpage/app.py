import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file
from io import BytesIO
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
import cloudpickle

from tamil_poem_analyzer import TamilPoemAnalyzer

app = Flask(__name__)

# Define model directory
model_dir = "models"

def load_model(path):
    """Load model using joblib"""
    with open(path, 'rb') as f:
        return joblib.load(f)

def load_pkl(path):
    """Load model using cloudpickle"""
    with open(path, 'rb') as f:
        return cloudpickle.load(f)

# Load models
try:
    aif_model = load_model(r"H:\Text Mining\Tamil poems\Webpage\new_models\AIF\jl\aif_rf_model.joblib")
    aif_scaler = load_model(r"H:\Text Mining\Tamil poems\Webpage\new_models\AIF\jl\aif_rf_scaler.joblib")
    aif_label_enc = load_model(r"H:\Text Mining\Tamil poems\Webpage\new_models\AIF\jl\aif_rf_le.joblib")

    sent_model = load_model(r"H:\Text Mining\Tamil poems\Webpage\new_models\Sentiment\jl\sent_rf_model.joblib")
    sent_scaler = load_model(r"H:\Text Mining\Tamil poems\Webpage\new_models\Sentiment\jl\sent_rf_scaler.joblib")
    sent_label_enc = load_model(r"H:\Text Mining\Tamil poems\Webpage\new_models\Sentiment\jl\sent_rf_le.joblib")

    emo_model = load_model(r"H:\Text Mining\Tamil poems\Webpage\new_models\Emotion\jl\emot_dt_model.joblib")
    #emo_scaler = load_model(os.path.join(model_dir, "emot_gbc_scaler.pkl"))
    emo_label_enc = load_model(r"H:\Text Mining\Tamil poems\Webpage\new_models\Emotion\jl\emot_dt_le.joblib")

    analyzer = TamilPoemAnalyzer()
    print("All models loaded successfully!")
    
except Exception as e:
    print(f"Error loading models: {e}")
    analyzer = None

# Feature name mapping from lowercase to uppercase (same as in analyzer)
FEATURE_NAME_MAPPING = {
    # Lexical Features (8 features)
    'total_word_count': 'TOTAL_WORD_COUNT',
    'unique_word_count': 'UNIQUE_WORD_COUNT',
    'type_token_ratio': 'TYPE_TOKEN_RATIO',
    'avg_word_length': 'AVG_WORD_LENGTH',
    'std_word_length': 'STD_WORD_LENGTH',
    'hapax_legomena': 'HAPAX_LEGOMENA',
    'function_word_freq': 'FUNCTION_WORD_FREQ',
    'stop_word_ratio': 'STOP_WORD_RATIO',
    
    # Character-Level Features (6 features)
    'char_unigram_entropy': 'CHAR_UNIGRAM_ENTROPY',
    'char_bigram_entropy': 'CHAR_BIGRAM_ENTROPY',
    'char_trigram_entropy': 'CHAR_TRIGRAM_ENTROPY',
    'char_diversity': 'CHAR_DIVERSITY',
    'rare_char_ratio': 'RARE_CHAR_RATIO',
    'avg_syllables_per_word': 'AVG_SYLLABLES_PER_WORD',
    
    # Structural Features (6 features)
    'avg_line_length_words': 'AVG_LINE_LENGTH_WORDS',
    'avg_stanza_length': 'AVG_STANZA_LENGTH',
    'total_lines': 'TOTAL_LINES',
    'anaphora_score': 'ANAPHORA_SCORE',
    'epiphora_score': 'EPIPHORA_SCORE',
    'enjambment_ratio': 'ENJAMBMENT_RATIO',
    
    # Syntactic Features (3 features)
    'estimated_noun_count': 'ESTIMATED_NOUN_COUNT',
    'estimated_verb_count': 'ESTIMATED_VERB_COUNT',
    'noun_verb_ratio': 'NOUN_VERB_RATIO',
    
    # Stylometric Features (1 feature)
    'readability_score': 'READABILITY_SCORE'
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results", methods=["POST"])
def results():
    try:
        text = request.form["poem"]
        analysis_type = request.form["analysis"]
        
        if not text.strip():
            return render_template("results.html", result="Please enter some text to analyze.")
        
        if analysis_type == "word":
            # Generate word cloud
            """
            img = BytesIO()
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(img, format="PNG", bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            img.seek(0)
            return send_file(img, mimetype="image/png")
            """
            
            return render_template("word_cloud.html", poem_text=text)
        
        elif analysis_type in ["aif", "sent", "emo"]:
            if analyzer is None:
                return render_template("results.html", result="Models not loaded. Please check server logs.")
            
            # Extract features from the poem
            lexical = analyzer.extract_lexical_features(text)
            char = analyzer.extract_character_features(text)
            structural = analyzer.extract_structural_features(text)
            syntactic = analyzer.extract_syntactic_features(text)
            stylometric = analyzer.extract_stylometric_features(text)

            # Combine all features
            combined_features = {}
            combined_features.update(lexical)
            combined_features.update(char)
            combined_features.update(structural)
            combined_features.update(syntactic)
            combined_features.update(stylometric)

            # Create DataFrame and rename columns to match training data
            df = pd.DataFrame([combined_features])
            
            # Rename columns to uppercase format (same as training data)
            df = df.rename(columns=FEATURE_NAME_MAPPING)
            
            # Select only the main 24 features that models expect
            expected_features = list(FEATURE_NAME_MAPPING.values())
            df = df[[col for col in expected_features if col in df.columns]]
            
            print(f"Feature extraction completed. Shape: {df.shape}")
            print(f"Features: {list(df.columns)}")
            
            if analysis_type == "aif":
                # Author identification
                features_scaled = aif_scaler.transform(df)
                pred = aif_model.predict(features_scaled)
                author = aif_label_enc.inverse_transform(pred)[0]
                return render_template("results.html", result=f"Predicted Author: {author}")

            elif analysis_type == "sent":
                # Sentiment analysis
                features_scaled = sent_scaler.transform(df)
                pred = sent_model.predict(features_scaled)
                sentiment = sent_label_enc.inverse_transform(pred)[0]
                return render_template("results.html", result=f"Predicted Sentiment: {sentiment}")

            elif analysis_type == "emo":
                # Emotion analysis
                #features_scaled = emo_scaler.transform(df)
                pred = emo_model.predict(df)
                emotion = emo_label_enc.inverse_transform(pred)[0]
                return render_template("results.html", result=f"Predicted Emotion: {emotion}")
        
        else:
            return render_template("results.html", result="Invalid analysis type selected.")
    
    except Exception as e:
        print(f"Error in results route: {e}")
        return render_template("results.html", result=f"An error occurred during analysis: {str(e)}")

@app.errorhandler(404)
def page_not_found(e):
    return render_template("error.html", error="Page not found"), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template("error.html", error="Internal server error"), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)