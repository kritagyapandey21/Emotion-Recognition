import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from PIL import Image, ImageTk
import pickle
import os
import time

class EmotionClassifierApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Emotion Classifier - CS Student Project")
        self.root.geometry("1000x800")
        self.root.minsize(900, 700)
        
        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        self.style.configure('Accent.TButton', font=('Arial', 10, 'bold'), foreground='white', background='#4a6baf')
        self.style.map('Accent.TButton', background=[('active', '#3a5a9f')])
        
        # Emotion configuration
        self.emotion_map = {
            0: 'Neutral', 
            1: 'Joy', 
            2: 'Love', 
            3: 'Anger', 
            4: 'Sadness', 
            5: 'Surprise',
            6: 'Fear'
        }
        self.emotion_colors = {
            'Neutral': '#808080',
            'Joy': '#FFD700',
            'Love': '#FF69B4',
            'Anger': '#FF4500',
            'Sadness': '#1E90FF',
            'Surprise': '#9370DB',
            'Fear': '#8B4513'
        }
        
        # Model variables
        self.model = None
        self.vectorizer = None
        self.current_model_name = "Logistic Regression"
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC(probability=True),
            "Naive Bayes": MultinomialNB(),
            "Random Forest": RandomForestClassifier()
        }
        
        # Dataset variables - Using your provided path
        self.dataset = None
        self.default_dataset_path = r"D:\Semester 4\python_data_science\excel\training.csv"
        
        # UI Setup
        self.create_widgets()
        
        # Try to load the default dataset automatically
        self.try_load_default_dataset()
    
    def try_load_default_dataset(self):
        """Attempt to load the default dataset at startup"""
        if os.path.exists(self.default_dataset_path):
            try:
                self.load_dataset(self.default_dataset_path)
                self.update_status(f"Loaded default dataset from {self.default_dataset_path}")
            except Exception as e:
                self.update_status(f"Failed to load default dataset: {str(e)}")
        else:
            self.update_status("Default dataset not found. Please load a dataset.")
    
    def create_widgets(self):
        """Create all UI components"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header_frame, text="Emotion Classifier", style='Title.TLabel').pack(side=tk.LEFT)
        
        # Load dataset button
        ttk.Button(
            header_frame, 
            text="Load Dataset", 
            command=self.load_dataset_dialog,
            style='Accent.TButton'
        ).pack(side=tk.RIGHT, padx=5)
        
        # Model selection
        ttk.Button(
            header_frame, 
            text="Change Model", 
            command=self.show_model_selection,
            style='Accent.TButton'
        ).pack(side=tk.RIGHT, padx=5)
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Input and results
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Input frame
        input_frame = ttk.LabelFrame(left_panel, text="Text Input", padding=10)
        input_frame.pack(fill=tk.BOTH, pady=(0, 10))
        
        self.text_input = scrolledtext.ScrolledText(input_frame, height=10, wrap=tk.WORD, font=('Arial', 11))
        self.text_input.pack(fill=tk.BOTH, expand=True)
        
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(
            button_frame, 
            text="Predict Emotion", 
            command=self.predict_emotion_threaded,
            style='Accent.TButton'
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            button_frame, 
            text="Clear", 
            command=self.clear_input
        ).pack(side=tk.LEFT, padx=5)
        
        # Result frame
        result_frame = ttk.LabelFrame(left_panel, text="Prediction Results", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_canvas = tk.Canvas(result_frame, bg='white', highlightthickness=0)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Model info and evaluation
        right_panel = ttk.Frame(content_frame, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        # Model info frame
        model_info_frame = ttk.LabelFrame(right_panel, text="Model Information", padding=10)
        model_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.model_name_label = ttk.Label(model_info_frame, text="No model loaded", font=('Arial', 10, 'bold'))
        self.model_name_label.pack(anchor=tk.W)
        
        self.model_status_label = ttk.Label(model_info_frame, text="Status: Not trained", foreground='red')
        self.model_status_label.pack(anchor=tk.W)
        
        self.dataset_info_label = ttk.Label(model_info_frame, text="Dataset: None loaded")
        self.dataset_info_label.pack(anchor=tk.W)
        
        ttk.Button(
            model_info_frame, 
            text="Train Model", 
            command=self.train_model_threaded,
            style='Accent.TButton'
        ).pack(fill=tk.X, pady=(5, 0))
        
        # Evaluation frame
        evaluation_frame = ttk.LabelFrame(right_panel, text="Model Evaluation", padding=10)
        evaluation_frame.pack(fill=tk.BOTH, expand=True)
        
        self.accuracy_label = ttk.Label(evaluation_frame, text="Accuracy: -")
        self.accuracy_label.pack(anchor=tk.W)
        
        # Notebook for evaluation metrics
        eval_notebook = ttk.Notebook(evaluation_frame)
        eval_notebook.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Classification report tab
        report_frame = ttk.Frame(eval_notebook)
        self.report_text = scrolledtext.ScrolledText(report_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.report_text.pack(fill=tk.BOTH, expand=True)
        eval_notebook.add(report_frame, text="Classification Report")
        
        # Confusion matrix tab
        cm_frame = ttk.Frame(eval_notebook)
        self.confusion_matrix_canvas = tk.Canvas(cm_frame)
        self.confusion_matrix_canvas.pack(fill=tk.BOTH, expand=True)
        eval_notebook.add(cm_frame, text="Confusion Matrix")
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, pady=(5, 0))
    
    def load_dataset_dialog(self):
        """Open file dialog to load dataset"""
        file_path = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("CSV files", ".csv"), ("All files", ".*")],
            initialdir=os.path.dirname(self.default_dataset_path)
        )
        
        if file_path:
            self.load_dataset(file_path)
    
    def load_dataset(self, file_path):
        """Load dataset from CSV file"""
        try:
            self.update_status(f"Loading dataset from {file_path}...")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Basic validation
            required_columns = {'text', 'label'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                messagebox.showerror("Error", f"Dataset missing required columns: {', '.join(missing)}")
                return
            
            # Clean data
            df = df.dropna()
            df['text'] = df['text'].astype(str).str.strip()
            df = df[df['text'] != '']
            
            # Map labels to emotion names
            df['emotion'] = df['label'].map(self.emotion_map)
            df = df[df['emotion'].notna()]
            
            if df.empty:
                messagebox.showerror("Error", "No valid data found in the dataset")
                return
            
            # Check if we have all expected emotions
            unique_emotions = df['emotion'].unique()
            missing_emotions = set(self.emotion_map.values()) - set(unique_emotions)
            if missing_emotions:
                self.update_status(f"Warning: Missing emotions in dataset: {', '.join(missing_emotions)}")
            
            self.dataset = df
            self.default_dataset_path = file_path
            self.dataset_info_label.config(text=f"Dataset: {os.path.basename(file_path)}\n{len(df)} samples, {len(unique_emotions)} emotions")
            self.update_status(f"Dataset loaded successfully with {len(df)} samples")
            
            # Reset model since dataset changed
            self.model = None
            self.model_status_label.config(text="Status: Needs training", foreground='orange')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            self.update_status("Error loading dataset")
    
    def show_model_selection(self):
        """Show dialog to select a different model"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Model")
        dialog.geometry("400x300")
        dialog.resizable(False, False)
        
        ttk.Label(dialog, text="Choose a machine learning model:", font=('Arial', 11)).pack(pady=10)
        
        model_frame = ttk.Frame(dialog)
        model_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        selected_model = tk.StringVar(value=self.current_model_name)
        
        for model_name in self.models.keys():
            rb = ttk.Radiobutton(
                model_frame, 
                text=model_name, 
                variable=selected_model, 
                value=model_name
            )
            rb.pack(anchor=tk.W, pady=5)
        
        def apply_selection():
            self.current_model_name = selected_model.get()
            self.model_name_label.config(text=f"Model: {self.current_model_name}")
            self.model = None
            self.model_status_label.config(text="Status: Needs training", foreground='orange')
            dialog.destroy()
            self.update_status(f"Model changed to {self.current_model_name}. Please train the model.")
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(
            button_frame, 
            text="Cancel", 
            command=dialog.destroy
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Apply", 
            command=apply_selection,
            style='Accent.TButton'
        ).pack(side=tk.RIGHT)
    
    def train_model_threaded(self):
        """Start model training in a separate thread"""
        if self.dataset is None:
            messagebox.showerror("Error", "Please load a dataset first")
            return
        
        self.model_status_label.config(text="Status: Training...", foreground='orange')
        self.update_status("Training model...")
        
        threading.Thread(target=self._train_model, daemon=True).start()
    
    def _train_model(self):
        """Train the selected model on the loaded dataset"""
        try:
            start_time = time.time()
            
            # Prepare data
            X = self.dataset['text']
            y = self.dataset['emotion']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Vectorize text
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)
            
            # Get and train selected model
            model = self.models[self.current_model_name]
            model.fit(X_train_tfidf, y_train)
            self.model = model
            
            # Evaluate
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, labels=list(self.emotion_map.values()))
            
            # Update UI
            self.root.after(0, self.update_evaluation_results, accuracy, report, cm)
            self.root.after(0, self.model_status_label.config, 
                          {"text": f"Status: Trained ({(time.time()-start_time):.1f}s", "foreground": "green"})
            self.root.after(0, self.update_status, 
                          f"Model trained successfully with {accuracy:.2%} accuracy")
            
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Training Error", f"Error during training: {str(e)}")
            self.root.after(0, self.model_status_label.config, 
                          {"text": "Status: Training failed", "foreground": "red"})
            self.root.after(0, self.update_status, "Model training failed")
    
    def update_evaluation_results(self, accuracy, report, cm):
        """Update the evaluation metrics display"""
        self.accuracy_label.config(text=f"Accuracy: {accuracy:.2%}")
        
        # Update classification report
        self.report_text.config(state=tk.NORMAL)
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, report)
        self.report_text.config(state=tk.DISABLED)
        
        # Update confusion matrix
        self.plot_confusion_matrix(cm)
    
    def plot_confusion_matrix(self, cm):
        """Plot the confusion matrix on the canvas"""
        # Clear previous plot
        for widget in self.confusion_matrix_canvas.winfo_children():
            widget.destroy()
        
        # Create new plot
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(self.emotion_map.values()),
            yticklabels=list(self.emotion_map.values()),
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.confusion_matrix_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def predict_emotion_threaded(self):
        """Start emotion prediction in a separate thread"""
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to analyze")
            return
        
        if self.model is None or self.vectorizer is None:
            messagebox.showerror("Error", "Please train the model first")
            return
        
        self.update_status("Analyzing text...")
        threading.Thread(target=self._predict_emotion, args=(text,), daemon=True).start()
    
    def _predict_emotion(self, text):
        """Predict emotion from text"""
        try:
            # Vectorize text
            text_tfidf = self.vectorizer.transform([text])
            
            # Predict
            emotion = self.model.predict(text_tfidf)[0]
            probabilities = self.model.predict_proba(text_tfidf)[0]
            
            # Sort probabilities
            emotion_probs = sorted(
                zip(self.model.classes_, probabilities),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Update UI
            self.root.after(0, self.display_prediction_result, emotion, emotion_probs)
            self.root.after(0, self.update_status, "Prediction complete")
            
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Prediction Error", f"Error during prediction: {str(e)}")
            self.root.after(0, self.update_status, "Prediction failed")
    
    def display_prediction_result(self, emotion, emotion_probs):
        """Display the prediction results visually"""
        # Clear previous results
        self.result_canvas.delete("all")
        
        # Draw emotion label
        self.result_canvas.create_text(
            10, 30,
            text="Predicted Emotion:",
            font=('Arial', 12),
            anchor=tk.W
        )
        
        self.result_canvas.create_text(
            10, 60,
            text=emotion,
            font=('Arial', 24, 'bold'),
            fill=self.emotion_colors.get(emotion, 'black'),
            anchor=tk.W
        )
        
        # Draw probability bars
        y_offset = 120
        bar_width = 200
        max_prob = max(prob for _, prob in emotion_probs)
        
        for e, prob in emotion_probs:
            # Bar background
            self.result_canvas.create_rectangle(
                10, y_offset,
                10 + bar_width, y_offset + 25,
                fill='#e0e0e0',
                outline=''
            )
            
            # Bar fill
            fill_width = (prob / max_prob) * bar_width if max_prob > 0 else 0
            self.result_canvas.create_rectangle(
                10, y_offset,
                10 + fill_width, y_offset + 25,
                fill=self.emotion_colors.get(e, '#808080'),
                outline=''
            )
            
            # Text labels
            self.result_canvas.create_text(
                15, y_offset + 12,
                text=e,
                font=('Arial', 10),
                anchor=tk.W,
                fill='black' if prob < 0.5 else 'white'
            )
            
            self.result_canvas.create_text(
                10 + bar_width + 5, y_offset + 12,
                text=f"{prob:.1%}",
                font=('Arial', 10),
                anchor=tk.W
            )
            
            y_offset += 35
    
    def clear_input(self):
        """Clear the input text area"""
        self.text_input.delete("1.0", tk.END)
        self.result_canvas.delete("all")
        self.update_status("Input cleared")
    
    def update_status(self, message):
        """Update the status bar"""
        self.status_bar.config(text=message)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionClassifierApp(root)
    root.mainloop()
