# **DiseasePredictor**

## **Overview**  
This machine learning system predicts diseases based on user-reported symptoms. Using a **Random Forest classifier** trained on symptom-disease relationships, it provides quick preliminary diagnoses. The system handles **data preprocessing**, **feature encoding**, and **model training** automatically.

## **Key Features**
- 🩺 **Symptom-based disease prediction**
- 🧠 **Random Forest machine learning model**
- 🧹 **Automatic data cleaning (handles missing values)**
- 📋 **Symptom vocabulary access**
- 🧪 **Built-in validation (20% test set)**
- 🔄 **Consistent feature encoding**

## **System Design**

![alt text](/Users/karthikrajanichenametla/DiseasePredictor/System_image.png)

## **Workflow Sequence**

![alt text](/Users/karthikrajanichenametla/DiseasePredictor/System_image_3.png)

---

## **Sample Prediction**

```bash
$ python3 disease_pred.py
📋 Known Symptoms: abdominal pain, back pain, balance problems, bloating, blurred vision, chest pain, chills, confusion, cough, dehydration, diarrhea, difficulty concentrating, difficulty swallowing, disorientation, dry mouth, ear pain, fatigue, fever, forgetfulness, frequent urination, gas, headache, high fever, increased thirst, itchy eyes, joint pain, limited mobility, muscle stiffness, nasal congestion, nausea, night sweats, pain in left arm, palpitations, rash, runny nose, seizures, sensitivity to light, shortness of breath, slow movement, sneezing, sore throat, stiff neck, stiffness, sweating, swollen tonsils, tremor, vomiting, weight loss

Enter symptoms separated by commas: diarrhea, bloating

Input Symptoms: ['diarrhea', 'bloating']
🩺 Predicted Disease: irritable bowel syndrome

## **requirements.txt**

```txt
pandas
scikit-learn
numpy
joblib