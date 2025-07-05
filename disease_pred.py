import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

class DiseasePredictor:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.mlb = MultiLabelBinarizer()
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def _prepare_data(self):
        data = [
        ['fever', 'cough', 'headache', 'chills', 'fatigue', 'flu'],
        ['joint pain', 'rash', 'fever', 'vomiting', None, 'dengue'],
        ['nausea', 'vomiting', 'diarrhea', 'abdominal pain', 'dehydration', 'food poisoning'],
        ['sneezing', 'runny nose', 'sore throat', 'headache', None, 'common cold'],
        ['fatigue', 'weight loss', 'night sweats', 'cough', None, 'tuberculosis'],
        ['chest pain', 'shortness of breath', 'fatigue', 'cough', 'palpitations', 'pneumonia'],
        ['itchy eyes', 'sneezing', 'nasal congestion', 'runny nose', None, 'allergic rhinitis'],
        ['abdominal pain', 'bloating', 'diarrhea', 'gas', 'fatigue', 'irritable bowel syndrome'],
        ['back pain', 'stiffness', 'limited mobility', 'fatigue', None, 'arthritis'],
        ['dry mouth', 'increased thirst', 'frequent urination', 'blurred vision', None, 'diabetes'],
        ['chest pain', 'sweating', 'nausea', 'shortness of breath', 'pain in left arm', 'heart attack'],
        ['tremor', 'slow movement', 'muscle stiffness', 'balance problems', None, 'parkinson\'s disease'],
        ['forgetfulness', 'confusion', 'difficulty concentrating', 'disorientation', None, 'alzheimer\'s disease'],
        ['high fever', 'stiff neck', 'sensitivity to light', 'seizures', None, 'meningitis'],
        ['sore throat', 'difficulty swallowing', 'swollen tonsils', 'fever', 'ear pain', 'tonsillitis']
    ]
        columns = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 'Prognosis']
        df = pd.DataFrame(data, columns=columns)

        df['symptom_list'] = df[columns[:-1]].values.tolist()
        df['symptom_list'] = df['symptom_list'].apply(lambda x: [i for i in x if i is not None])

        X = self.mlb.fit_transform(df['symptom_list'])
        y = self.label_encoder.fit_transform(df['Prognosis'])

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        X_train, X_test, y_train, y_test = self._prepare_data()
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, symptoms: list) -> str:
        if not self.is_trained:
            raise Exception("Model not trained. Call `train()` first.")
        
        X_input = self.mlb.transform([symptoms])
        pred = self.model.predict(X_input)[0]
        return self.label_encoder.inverse_transform([pred])[0]

    def get_known_symptoms(self):
        return list(self.mlb.classes_)

if __name__ == "__main__":
    predictor = DiseasePredictor()
    predictor.train()

    print("\n📋 Known Symptoms:", ", ".join(predictor.get_known_symptoms()))
    user_input = input("\n Enter symptoms separated by commas: ")
    
    input_symptoms = [s.strip() for s in user_input.lower().split(",")]
    result = predictor.predict(input_symptoms)

    print("\nInput Symptoms:", input_symptoms)
    print("🩺 Predicted Disease:", result)
