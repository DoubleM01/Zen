import pickle
import pandas as pd


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train = train.drop(columns=['subject', 'Activity', 'ActivityName'], errors='ignore')
    y_train = train['ActivityName']
    X_test = test.drop(columns=['subject', 'Activity', 'ActivityName'], errors='ignore')
    y_test = test['ActivityName']
    return X_train, y_train, X_test, y_test

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
X_train, y_train, X_test, y_test = load_data('train.csv', 'test.csv')
y_pred = model.predict(X_test.loc[79:108])
accuracy = model.score(X_test, y_test)
print(X_test.loc[79].shape)
print(y_pred)
print(y_test.loc[79])
print(y_test.unique())
print(y_test.where(y_test =='WALKING').dropna())

# met_estimator.py

# MET values for each activity (from the Adult Compendium of Physical Activities)
MET_VALUES = {
    'STANDING': 1.3,           # Standing quietly (e.g. standing in a line) :contentReference[oaicite:0]{index=0}
    'SITTING': 1.0,            # Sitting quietly (watching TV, general) :contentReference[oaicite:1]{index=1}
    'LAYING': 1.0,             # Lying quietly, sleeping :contentReference[oaicite:2]{index=2}
    'WALKING': 3.3,            # Walking 2.5–3 mph (~4 km/h) :contentReference[oaicite:3]{index=3}
    'WALKING_DOWNSTAIRS': 3.5, # Descending stairs :contentReference[oaicite:4]{index=4}
    'WALKING_UPSTAIRS': 4.0    # Climbing stairs, slow pace :contentReference[oaicite:5]{index=5}
}

def calories_from_met(activity: str, weight_kg: float, duration_sec: float) -> float:
    """
    Estimate calories burned based on MET.

    - activity: one of the keys in MET_VALUES
    - weight_kg: user weight in kilograms
    - duration_sec: duration of the activity window in seconds

    Formula: Calories = MET × 3.5 × weight_kg / 200  (kcal per minute)
             then × (duration_sec / 60) to scale to the window length.
    """
    met = MET_VALUES.get(activity.upper(), 1.0)
    kcal_per_min = met * 3.5 * weight_kg / 200
    return kcal_per_min * (duration_sec / 60)

# Example usage:
if __name__ == "__main__":
    weight = 85           # kg
    window = 30           # seconds (1‑minute window)
    for act in MET_VALUES:
        if act == y_pred[0]:
            cal = calories_from_met(act, weight, window)
            print(f"{act:22s} → {cal:.2f} kcal burned in {window:.0f}s")
