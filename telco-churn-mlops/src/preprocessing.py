
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer


df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')


df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
encoder=LabelEncoder()
df['Churn'] = encoder.fit_transform(df['Churn'])
X = df.drop('Churn', axis=1)
encoder_target = LabelEncoder()
y = encoder_target.fit_transform(df['Churn'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def get_preprocessor(X: pd.DataFrame):
    
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    return preprocessor