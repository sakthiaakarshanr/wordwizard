import os
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import math
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TamilPoemAnalyzer:
    def __init__(self):  # Fixed constructor name
        # Tamil Unicode ranges and common characters
        self.tamil_chars = set('அஆஇஈஉஊஎஏஐஒஓஔகஙசஜஞடணதநனபமயரறலளழவஶஷஸஹ்ாிீுூெேைொோௌ்ௐ')
        self.rare_tamil_chars = set('ற்ழ்ஞ்ஶ்ஷ்ஸ்ஹ்')
        self.tamil_function_words = {'என்று', 'ஆனால்', 'மற்றும்', 'அல்லது', 'இது', 'அது', 'இந்த', 'அந்த', 'ஒரு', 'அவர்', 'அவள்', 'நான்', 'நீ', 'நாம்', 'நீங்கள்'}
        self.tamil_stop_words = {'ஒரு', 'இந்த', 'அந்த', 'இது', 'அது', 'என்று', 'அல்லது', 'மற்றும்', 'ஆனால்', 'எனவே', 'மேலும்', 'தான்', 'கூட', 'மட்டும்', 'போல்', 'வரை', 'பிறகு', 'முன்', 'பின்', 'மேல்', 'கீழ்'}
        
        # Tamil consonants and vowels for better word segmentation
        self.tamil_consonants = 'கஙசஞடணதநபமயரறலளழவஶஷஸஹ'
        self.tamil_vowels = 'அஆஇஈஉஊஎஏஐஒஓஔ'
        self.tamil_vowel_signs = 'ாிீுூெேைொோௌ்'
        
        # Enhanced verb endings for better grammatical analysis
        self.verb_endings = [
            'கிறார்', 'கிறாள்', 'கிறான்', 'கிறது', 'கிறேன்', 'கிறோம்', 'கிறீர்', 'கிறீர்கள்',
            'ந்தார்', 'ந்தாள்', 'ந்தான்', 'ந்தது', 'ந்தேன்', 'ந்தோம்', 'ந்தீர்', 'ந்தீர்கள்',
            'வார்', 'வாள்', 'வான்', 'வது', 'வேன்', 'வோம்', 'வீர்', 'வீர்கள்',
            'கின்றார்', 'கின்றாள்', 'கின்றான்', 'கின்றது', 'கின்றேன்', 'கின்றோம்',
            'ட்டார்', 'ட்டாள்', 'ட்டான்', 'ட்டது', 'ட்டேன்', 'ட்டோம்'
        ]
        
        # Enhanced noun endings
        self.noun_endings = ['கள்', 'அன்', 'இன்', 'உன்', 'என்', 'ஆன்', 'ீன்', 'ூன்', 'ான்', 'ின்']
        
    def is_tamil_text(self, text):
        """Check if text contains significant Tamil content"""
        tamil_char_count = sum(1 for char in text if '\u0B80' <= char <= '\u0BFF')
        return tamil_char_count / len(text) > 0.3 if text else False
    
    def tamil_word_tokenize(self, text):
        """
        Improved Tamil word tokenization that handles Tamil script properly
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split by spaces first (this handles most cases correctly)
        potential_words = text.split()
        
        # Filter out empty strings and clean each word
        words = []
        for word in potential_words:
            # Remove punctuation from the word but keep the core Tamil text
            cleaned_word = re.sub(r'[^\u0B80-\u0BFF\w]', '', word)  # Keep only Tamil Unicode and word characters
            if cleaned_word and len(cleaned_word) > 0:
                # Check if it's a meaningful Tamil word (contains Tamil characters)
                if any(char in self.tamil_chars for char in cleaned_word):
                    words.append(cleaned_word)
        
        return words
    
    def advanced_tamil_word_tokenize(self, text):
        """
        More sophisticated Tamil word tokenization with better error handling
        """
        if not text or not isinstance(text, str):
            return []
            
        # First, clean the text
        text = text.replace('\n', ' ').replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split by natural word boundaries (spaces and common punctuation)
        words = re.split(r'[\s\u0020\u00A0\u2000-\u200F\u2028\u2029.,;:!?()[\]{}"\'-]+', text)
        
        # Filter and clean words
        cleaned_words = []
        for word in words:
            word = word.strip()
            if word:
                # Remove any remaining non-Tamil characters at word boundaries
                word = re.sub(r'^[^\u0B80-\u0BFF]+|[^\u0B80-\u0BFF]+$', '', word)
                if word and len(word) > 0:
                    # Ensure the word contains at least one Tamil character
                    if any('\u0B80' <= char <= '\u0BFF' for char in word):
                        cleaned_words.append(word)
        
        return cleaned_words
        
    def read_poem_files(self, directory_path, author):
        """Read all poem files from directory with enhanced error handling"""
        poems_data = []
        directory = Path(directory_path)
        
        if not directory.exists():
            print(f"Directory {directory_path} does not exist!")
            return []
        
        # Support multiple file extensions
        file_extensions = ['*.txt', '*.csv', '*.tsv']
        files_found = []
        
        for ext in file_extensions:
            files_found.extend(directory.glob(ext))
        
        if not files_found:
            print(f"No text files found in {directory_path}")
            return []
        
        print(f"Found {len(files_found)} files to process")
        
        for file_path in files_found:
            try:
                # Try multiple encodings to properly read Tamil text
                content = None
                for encoding in ['utf-8', 'utf-8-sig', 'utf-16', 'cp1252', 'latin-1']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            content = file.read().strip()
                            break
                    except UnicodeDecodeError:
                        continue
                
                if content is None:
                    print(f"Could not read {file_path} with any encoding")
                    continue
                
                # Check if content contains Tamil text
                if not self.is_tamil_text(content):
                    print(f"Warning: {file_path} doesn't seem to contain Tamil text")
                    
                # Extract author from filename or first line
                filename = file_path.stem
                
                # Try to extract poem name from filename
                if '_' in filename:
                    poem_name = ''.join(filename.split('_')[0:])
                else:
                    poem_name = filename
                
                # If first line contains "Author:", remove it from content
                lines = content.split('\n')
                if lines and lines[0].strip().startswith('Author:'):
                    content = '\n'.join(lines[1:])
                
                # Skip empty files
                if not content.strip():
                    print(f"Skipping empty file: {file_path}")
                    continue
                
                poems_data.append({
                    'author': author,
                    'poem_name': poem_name,
                    'content': content,
                    'filepath': str(file_path)
                })
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
        return poems_data
    
    def extract_lexical_features(self, text):
        """Extract word-level usage patterns with improved Tamil tokenization"""
        # Use improved Tamil tokenization
        words = self.advanced_tamil_word_tokenize(text)
        
        # Also get English regex words for comparison/debugging
        english_regex_words = re.findall(r'\b\w+\b', text)
        
        word_lengths = [len(word) for word in words]
        word_freq = Counter(words)
        
        features = {}
        
        # Basic word statistics
        features['total_word_count'] = len(words)
        features['unique_word_count'] = len(set(words))
        features['type_token_ratio'] = len(set(words)) / len(words) if words else 0
        features['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0
        features['std_word_length'] = np.std(word_lengths) if word_lengths else 0
        
        # Vocabulary richness
        hapax = sum(1 for count in word_freq.values() if count == 1)
        features['hapax_legomena'] = hapax
        features['hapax_ratio'] = hapax / len(words) if words else 0
        
        # Dis legomena (words appearing exactly twice)
        dis_legomena = sum(1 for count in word_freq.values() if count == 2)
        features['dis_legomena'] = dis_legomena
        features['dis_legomena_ratio'] = dis_legomena / len(words) if words else 0
        
        # Function and stop word analysis
        function_word_count = sum(1 for word in words if word in self.tamil_function_words)
        features['function_word_freq'] = function_word_count
        features['function_word_ratio'] = function_word_count / len(words) if words else 0
        
        stop_word_count = sum(1 for word in words if word in self.tamil_stop_words)
        features['stop_word_ratio'] = stop_word_count / len(words) if words else 0
        
        # Word frequency distribution
        if word_freq:
            features['most_frequent_word_ratio'] = max(word_freq.values()) / len(words)
            features['vocabulary_concentration'] = sum(count**2 for count in word_freq.values()) / len(words)**2
        else:
            features['most_frequent_word_ratio'] = 0
            features['vocabulary_concentration'] = 0
        
        # Debug information
        features['debug_english_regex_count'] = len(english_regex_words)
        features['debug_word_count_difference'] = len(english_regex_words) - len(words)
        
        return features
    
    def extract_character_features(self, text):
        """Extract character-level features"""
        features = {}
        chars = list(text)
        
        if not chars:
            return {key: 0 for key in ['char_unigram_entropy', 'char_bigram_entropy', 'char_trigram_entropy', 
                                      'char_diversity', 'char_diversity_ratio', 'rare_char_count', 'rare_char_ratio', 
                                      'avg_syllables_per_word']}
        
        # Character n-grams
        unigrams = Counter(chars)
        bigrams = Counter([text[i:i+2] for i in range(len(text)-1)])
        trigrams = Counter([text[i:i+3] for i in range(len(text)-2)])
        
        features['char_unigram_entropy'] = self.calculate_entropy(unigrams)
        features['char_bigram_entropy'] = self.calculate_entropy(bigrams)
        features['char_trigram_entropy'] = self.calculate_entropy(trigrams)
        
        # Character diversity
        tamil_chars_in_text = set(char for char in chars if char in self.tamil_chars)
        features['char_diversity'] = len(tamil_chars_in_text)
        features['char_diversity_ratio'] = len(tamil_chars_in_text) / len(self.tamil_chars)
        
        # Frequency of rare Tamil letters
        rare_char_count = sum(1 for char in chars if char in self.rare_tamil_chars)
        features['rare_char_count'] = rare_char_count
        features['rare_char_ratio'] = rare_char_count / len(chars) if chars else 0
        
        # Improved syllable counting
        words = self.advanced_tamil_word_tokenize(text)
        syllable_counts = []
        for word in words:
            syllables = self.count_tamil_syllables(word)
            syllable_counts.append(syllables)
        
        features['avg_syllables_per_word'] = np.mean(syllable_counts) if syllable_counts else 0
        features['syllable_variance'] = np.var(syllable_counts) if syllable_counts else 0
        
        return features
    
    def count_tamil_syllables(self, word):
        """Improved Tamil syllable counting"""
        if not word:
            return 0
        
        # Count vowels and vowel signs
        vowel_count = 0
        consonant_vowel_count = 0
        
        for i, char in enumerate(word):
            if char in self.tamil_vowels:
                vowel_count += 1
            elif char in self.tamil_vowel_signs:
                consonant_vowel_count += 1
        
        # Each consonant without a vowel sign gets an inherent 'a' sound
        consonant_count = sum(1 for char in word if char in self.tamil_consonants)
        inherent_vowels = max(0, consonant_count - consonant_vowel_count)
        
        total_syllables = vowel_count + consonant_vowel_count + inherent_vowels
        return max(1, total_syllables)  # At least 1 syllable per word
    
    def extract_structural_features(self, text):
        """Extract structural features with enhanced analysis"""
        features = {}
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            return {key: 0 for key in ['avg_line_length_words', 'avg_line_length_chars', 'std_line_length_words',
                                      'num_stanzas', 'avg_stanza_length', 'total_lines', 'anaphora_score',
                                      'epiphora_score', 'enjambment_ratio', 'line_length_variance']}
        
        # Line length analysis
        line_lengths_words = [len(self.advanced_tamil_word_tokenize(line)) for line in lines]
        line_lengths_chars = [len(line) for line in lines]
        
        features['avg_line_length_words'] = np.mean(line_lengths_words) if line_lengths_words else 0
        features['avg_line_length_chars'] = np.mean(line_lengths_chars) if line_lengths_chars else 0
        features['std_line_length_words'] = np.std(line_lengths_words) if line_lengths_words else 0
        features['line_length_variance'] = np.var(line_lengths_words) if line_lengths_words else 0
        
        # Stanza analysis
        full_text = '\n'.join(lines)
        stanzas = re.split(r'\n\s*\n', full_text)
        stanzas = [s.strip() for s in stanzas if s.strip()]
        
        stanza_lengths = [len(s.split('\n')) for s in stanzas]
        features['num_stanzas'] = len(stanzas)
        features['avg_stanza_length'] = np.mean(stanza_lengths) if stanza_lengths else 0
        features['total_lines'] = len(lines)
        
        # Repetition analysis
        first_words = []
        last_words = []
        for line in lines:
            line_words = self.advanced_tamil_word_tokenize(line)
            if line_words:
                first_words.append(line_words[0])
                last_words.append(line_words[-1])
        
        features['anaphora_score'] = self.calculate_repetition_score(first_words)
        features['epiphora_score'] = self.calculate_repetition_score(last_words)
        
        # Enjambment analysis
        punctuation = '.,!?;:।'
        enjambed_lines = sum(1 for line in lines if line and line[-1] not in punctuation)
        features['enjambment_ratio'] = enjambed_lines / len(lines) if lines else 0
        
        return features
    
    def extract_syntactic_features(self, text):
        """Extract syntactic and grammatical features with improved heuristics"""
        features = {}
        words = self.advanced_tamil_word_tokenize(text)
        
        if not words:
            return {key: 0 for key in ['estimated_verb_count', 'estimated_noun_count', 'noun_verb_ratio',
                                      'compound_word_ratio', 'postposition_ratio', 'question_word_ratio']}
        
        # Improved verb detection
        verb_count = sum(1 for word in words if any(word.endswith(ending) for ending in self.verb_endings))
        
        # Improved noun detection
        noun_count = sum(1 for word in words if any(word.endswith(ending) for ending in self.noun_endings))
        
        features['estimated_verb_count'] = verb_count
        features['estimated_noun_count'] = noun_count
        features['noun_verb_ratio'] = noun_count / verb_count if verb_count > 0 else 0
        
        # Compound word detection
        compound_indicators = ['்', 'ிய', 'ான', 'ும்', 'ற்', 'ந்']
        compound_count = sum(1 for word in words if any(ind in word for ind in compound_indicators))
        features['compound_word_ratio'] = compound_count / len(words) if words else 0
        
        # Postposition detection (common Tamil postpositions)
        postpositions = ['இல்', 'அல்', 'உள்', 'மேல்', 'கீழ்', 'முன்', 'பின்', 'பக்கம்', 'வழி']
        postposition_count = sum(1 for word in words if any(word.endswith(post) for post in postpositions))
        features['postposition_ratio'] = postposition_count / len(words) if words else 0
        
        # Question word detection
        question_words = ['என்ன', 'எங்கே', 'எப்போது', 'எப்படி', 'எவர்', 'எது', 'எதை', 'எவ்வளவு']
        question_count = sum(1 for word in words if word in question_words)
        features['question_word_ratio'] = question_count / len(words) if words else 0
        
        return features
    
    def extract_stylometric_features(self, text):
        """Extract stylometric features"""
        features = {}
        
        words = self.advanced_tamil_word_tokenize(text)
        sentences = re.split(r'[.!?।]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences and words:
            avg_words_per_sentence = len(words) / len(sentences)
            features['avg_words_per_sentence'] = avg_words_per_sentence
            features['sentence_length_variance'] = np.var([len(self.advanced_tamil_word_tokenize(s)) for s in sentences])
            features['readability_score'] = 100 - (avg_words_per_sentence * 2)  # Simple approximation
        else:
            features['avg_words_per_sentence'] = 0
            features['sentence_length_variance'] = 0
            features['readability_score'] = 0
        
        # Punctuation analysis
        punctuation_count = sum(1 for char in text if char in '.,!?;:।')
        features['punctuation_ratio'] = punctuation_count / len(text) if text else 0
        
        return features
    
    def calculate_entropy(self, counter):
        """Calculate entropy of a frequency distribution"""
        if not counter:
            return 0
        
        total = sum(counter.values())
        entropy = 0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def calculate_repetition_score(self, word_list):
        """Calculate repetition score for a list of words"""
        if not word_list:
            return 0
        
        word_counts = Counter(word_list)
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        return repeated_words / len(word_list)
    
    def analyze_poems(self, directory_path, author, output_csv='tamil_poem_features.csv'):
        """Main function to analyze all poems and save results - MAIN 24 FEATURES ONLY"""
        print("🔍 Reading poem files...")
        poems_data = self.read_poem_files(directory_path, author)
        
        if not poems_data:
            print("❌ No poem files found!")
            return None
        
        print(f"✅ Found {len(poems_data)} poems")
        
        # Define the MAIN 24 features ONLY
        main_features = {
            # 🔤 Lexical Features (8 features)
            'total_word_count': 'TOTAL_WORD_COUNT',
            'unique_word_count': 'UNIQUE_WORD_COUNT',
            'type_token_ratio': 'TYPE_TOKEN_RATIO',
            'avg_word_length': 'AVG_WORD_LENGTH',
            'std_word_length': 'STD_WORD_LENGTH',
            'hapax_legomena': 'HAPAX_LEGOMENA',
            'function_word_freq': 'FUNCTION_WORD_FREQ',
            'stop_word_ratio': 'STOP_WORD_RATIO',
            
            # 🔠 Character-Level Features (6 features)
            'char_unigram_entropy': 'CHAR_UNIGRAM_ENTROPY',
            'char_bigram_entropy': 'CHAR_BIGRAM_ENTROPY',
            'char_trigram_entropy': 'CHAR_TRIGRAM_ENTROPY',
            'char_diversity': 'CHAR_DIVERSITY',
            'rare_char_ratio': 'RARE_CHAR_RATIO',
            'avg_syllables_per_word': 'AVG_SYLLABLES_PER_WORD',
            
            # 🧱 Structural Features (6 features)
            'avg_line_length_words': 'AVG_LINE_LENGTH_WORDS',
            'avg_stanza_length': 'AVG_STANZA_LENGTH',
            'total_lines': 'TOTAL_LINES',
            'anaphora_score': 'ANAPHORA_SCORE',
            'epiphora_score': 'EPIPHORA_SCORE',
            'enjambment_ratio': 'ENJAMBMENT_RATIO',
            
            # 🧠 Syntactic Features (3 features)
            'estimated_noun_count': 'ESTIMATED_NOUN_COUNT',
            'estimated_verb_count': 'ESTIMATED_VERB_COUNT',
            'noun_verb_ratio': 'NOUN_VERB_RATIO',
            
            # 🎭 Stylometric Features (1 feature)
            'readability_score': 'READABILITY_SCORE'
        }
        
        all_features = []
        
        for i, poem in enumerate(poems_data):
            print(f"📝 Processing poem {i+1}/{len(poems_data)}: {poem['poem_name']}")
            
            # Initialize feature dictionary with poem info
            features = {
                'author': poem['author'],
                'poem_name': poem['poem_name']
            }
            
            # Extract ONLY the main 24 features
            try:
                lexical_features = self.extract_lexical_features(poem['content'])
                character_features = self.extract_character_features(poem['content'])
                structural_features = self.extract_structural_features(poem['content'])
                syntactic_features = self.extract_syntactic_features(poem['content'])
                stylometric_features = self.extract_stylometric_features(poem['content'])
                
                # Add only the features that are in main_features
                for feature_name in main_features.keys():
                    if feature_name in lexical_features:
                        features[feature_name] = lexical_features[feature_name]
                    elif feature_name in character_features:
                        features[feature_name] = character_features[feature_name]
                    elif feature_name in structural_features:
                        features[feature_name] = structural_features[feature_name]
                    elif feature_name in syntactic_features:
                        features[feature_name] = syntactic_features[feature_name]
                    elif feature_name in stylometric_features:
                        features[feature_name] = stylometric_features[feature_name]
                
            except Exception as e:
                print(f"⚠️ Error processing {poem['poem_name']}: {e}")
                continue
            
            all_features.append(features)
        
        if not all_features:
            print("❌ No features could be extracted!")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        
        # Rename main features to bold format
        rename_dict = {old_name: bold_name for old_name, bold_name in main_features.items() if old_name in df.columns}
        df = df.rename(columns=rename_dict)
        
        # Reorder columns: author, poem_name, then the 24 main features
        main_feature_cols = [bold_name for bold_name in main_features.values() if bold_name in df.columns]
        cols = ['author', 'poem_name'] + main_feature_cols
        df = df[cols]
        
        # Save to CSV
        try:
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"\n✅ Feature extraction complete! Results saved to: {output_csv}")
            print(f"📊 MAIN 24 FEATURES EXTRACTED")
            
            # Print the 24 main features by category
            print(f"\n🎯 THE 24 MAIN FEATURES:")
            print(f"   🔤 LEXICAL (8): Word count, unique words, type-token ratio, word length stats, hapax, function words, stop words")
            print(f"   🔠 CHARACTER (6): Unigram/bigram/trigram entropy, char diversity, rare chars, syllables")
            print(f"   🧱 STRUCTURAL (6): Line length, stanza length, total lines, anaphora, epiphora, enjambment")
            print(f"   🧠 SYNTACTIC (3): Noun count, verb count, noun-verb ratio")
            print(f"   🎭 STYLOMETRIC (1): Readability score")
            
            # Print summary statistics for key features
            print(f"\n📈 SUMMARY STATISTICS:")
            print(f"   Average words per poem: {df['TOTAL_WORD_COUNT'].mean():.1f}")
            print(f"   Average lines per poem: {df['TOTAL_LINES'].mean():.1f}")
            print(f"   Type-token ratio range: {df['TYPE_TOKEN_RATIO'].min():.3f} - {df['TYPE_TOKEN_RATIO'].max():.3f}")
            print(f"   Average readability: {df['READABILITY_SCORE'].mean():.1f}")
            
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
            return df
        
        return df

# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TamilPoemAnalyzer()
    
    # Example usage - update the path to your actual directory
    directory_path = r"C:\Users\Dell\OneDrive\Desktop\Summer_Intern\Manickavasagar"
    results_df = analyzer.analyze_poems(
        directory_path,
        author='manickavasagar',
        output_csv=r"C:\Users\Dell\OneDrive\Desktop\Summer_Intern\Webpage\aif\aif_manickavasagar.csv"
    )
    
    print("✅ Tamil Poem Analyzer initialized successfully!")
    print("📝 To use: analyzer.analyze_poems('your_directory_path', 'author_name', 'output.csv')")