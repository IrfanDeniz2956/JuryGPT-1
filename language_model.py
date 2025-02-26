from sklearn.svm import SVC
import joblib

# Örnek veri seti
X_train = [[1, 0], [0, 1], [1, 1], [0, 0]]
y_train = [0, 1, 1, 0]

# Modeli eğit
speech_model = SVC(probability=True)
speech_model.fit(X_train, y_train)

# Modeli kaydet
joblib.dump(speech_model, r"C:\Users\Acer\models\argument_model.pkl")
