import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Configure GPU for faster training
print("Configuring GPU for faster training...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s): {gpus}")
        print("GPU memory growth enabled for efficient memory usage")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found, using CPU")

# Set mixed precision for faster training on compatible GPUs
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled for faster training")
except:
    print("Mixed precision not available, using default precision")

# 1. Load data
laptops = pd.read_csv('laptops.csv')

# 2. Data cleaning and feature engineering
def clean_cpu(cpu):
    if pd.isnull(cpu):
        return 'Other'
    cpu = cpu.lower()
    if 'i9' in cpu:
        return 'i9'
    elif 'i7' in cpu:
        return 'i7'
    elif 'i5' in cpu:
        return 'i5'
    elif 'i3' in cpu:
        return 'i3'
    elif 'ryzen 9' in cpu:
        return 'ryzen 9'
    elif 'ryzen 7' in cpu:
        return 'ryzen 7'
    elif 'ryzen 5' in cpu:
        return 'ryzen 5'
    elif 'ryzen 3' in cpu:
        return 'ryzen 3'
    elif 'celeron' in cpu:
        return 'celeron'
    elif 'athlon' in cpu:
        return 'athlon'
    elif 'm1' in cpu or 'm2' in cpu:
        return 'apple silicon'
    else:
        return 'other'

def clean_gpu(gpu):
    if pd.isnull(gpu) or gpu == '':
        return 'integrated'
    gpu = gpu.lower()
    if 'rtx' in gpu:
        return 'rtx'
    elif 'gtx' in gpu:
        return 'gtx'
    elif 'radeon' in gpu:
        return 'radeon'
    elif 'mx' in gpu:
        return 'mx'
    elif 'apple' in gpu or 'm1' in gpu or 'm2' in gpu:
        return 'apple silicon'
    else:
        return 'integrated'

laptops['CPU_clean'] = laptops['CPU'].apply(clean_cpu)
laptops['GPU_clean'] = laptops['GPU'].apply(clean_gpu)
laptops['RAM'] = pd.to_numeric(laptops['RAM'], errors='coerce').fillna(8)
laptops['Storage'] = pd.to_numeric(laptops['Storage'], errors='coerce').fillna(256)
laptops['Final Price'] = pd.to_numeric(laptops['Final Price'], errors='coerce').fillna(laptops['Final Price'].median())

# 3. Usage category engineering
def assign_category(row):
    cpu = row['CPU_clean']
    gpu = row['GPU_clean']
    ram = row['RAM']
    price = row['Final Price']
    # Premium: ultra high-end
    if (
        price > 1800 and
        (
            cpu in ['i9', 'ryzen 9', 'apple silicon'] or
            ('m2' in str(row['CPU']).lower() and 'pro' in str(row['CPU']).lower()) or
            ('m2' in str(row['CPU']).lower() and 'max' in str(row['CPU']).lower())
        ) and
        ram >= 32 and
        (
            gpu in ['rtx'] or
            ('4070' in str(row['GPU']).lower() or '4080' in str(row['GPU']).lower() or '4090' in str(row['GPU']).lower()) or
            ('m2' in str(row['GPU']).lower() and ('pro' in str(row['GPU']).lower() or 'max' in str(row['GPU']).lower()))
        )
    ):
        return 'premium'
    # High-end editing: stricter
    elif ((cpu in ['i7', 'i9', 'ryzen 7', 'ryzen 9']) and ram >= 16 and gpu in ['rtx', 'gtx', 'radeon'] and price > 1200):
        return 'high-end editing'
    # Budget gaming
    elif ((cpu in ['i5', 'ryzen 5', 'i7', 'ryzen 7', 'i9', 'ryzen 9']) and ram >= 8 and gpu in ['gtx', 'rtx'] and 700 <= price <= 1200):
        return 'budget gaming'
    # Basic office
    elif ((cpu in ['i3', 'celeron', 'athlon']) and ram <= 8 and gpu == 'integrated' and price < 700):
        return 'basic office'
    else:
        return 'general use'

laptops['category'] = laptops.apply(assign_category, axis=1)

# 4. Encode features for ML
features = ['CPU_clean', 'GPU_clean', 'RAM', 'Storage', 'Final Price', 'Brand']
laptops['Brand'] = laptops['Brand'].fillna('Unknown')
le_cpu = LabelEncoder()
le_gpu = LabelEncoder()
le_brand = LabelEncoder()
laptops['CPU_enc'] = le_cpu.fit_transform(laptops['CPU_clean'])
laptops['GPU_enc'] = le_gpu.fit_transform(laptops['GPU_clean'])
laptops['Brand_enc'] = le_brand.fit_transform(laptops['Brand'])

# Prepare features for Wide & Deep model
categorical_features = ['CPU_enc', 'GPU_enc', 'Brand_enc']
numerical_features = ['RAM', 'Storage', 'Final Price']

# 5. Build Wide & Deep Neural Network
def create_wide_deep_model(num_categories, num_numerical, num_classes):
    # Input layers
    categorical_inputs = []
    for i, feature in enumerate(categorical_features):
        input_layer = tf.keras.layers.Input(shape=(1,), name=f'categorical_{feature}')
        categorical_inputs.append(input_layer)
    
    numerical_input = tf.keras.layers.Input(shape=(num_numerical,), name='numerical_features')
    
    # Wide part (linear)
    categorical_embeddings = []
    for i, (input_layer, feature) in enumerate(zip(categorical_inputs, categorical_features)):
        embedding = tf.keras.layers.Embedding(
            input_dim=le_cpu.classes_.shape[0] if feature == 'CPU_enc' else 
                     le_gpu.classes_.shape[0] if feature == 'GPU_enc' else 
                     le_brand.classes_.shape[0],
            output_dim=8,
            name=f'embedding_{feature}'
        )(input_layer)
        categorical_embeddings.append(embedding)
    
    # Concatenate embeddings
    categorical_concat = tf.keras.layers.Concatenate()(categorical_embeddings)
    categorical_flat = tf.keras.layers.Flatten()(categorical_concat)
    
    # Wide part: concatenate categorical and numerical
    wide_input = tf.keras.layers.Concatenate()([categorical_flat, numerical_input])
    wide_output = tf.keras.layers.Dense(64, activation='relu')(wide_input)
    wide_output = tf.keras.layers.Dropout(0.3)(wide_output)
    
    # Deep part
    deep_input = tf.keras.layers.Concatenate()([categorical_flat, numerical_input])
    deep_output = tf.keras.layers.Dense(128, activation='relu')(deep_input)
    deep_output = tf.keras.layers.Dropout(0.3)(deep_output)
    deep_output = tf.keras.layers.Dense(64, activation='relu')(deep_output)
    deep_output = tf.keras.layers.Dropout(0.3)(deep_output)
    deep_output = tf.keras.layers.Dense(32, activation='relu')(deep_output)
    deep_output = tf.keras.layers.Dropout(0.3)(deep_output)
    
    # Combine wide and deep
    combined = tf.keras.layers.Concatenate()([wide_output, deep_output])
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(combined)
    
    # Create model
    model = tf.keras.Model(
        inputs=categorical_inputs + [numerical_input],
        outputs=output
    )
    
    return model

# Prepare data for Wide & Deep model
X_categorical = [laptops[feature].values for feature in categorical_features]
X_numerical = laptops[numerical_features].values
y = laptops['category']

# Encode labels
le_category = LabelEncoder()
y_encoded = le_category.fit_transform(y)

# Split data - fix the splitting issue
# First, create indices for splitting
indices = np.arange(len(laptops))
train_indices, test_indices = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y_encoded
)

# Split categorical features
X_cat_train = [laptops[feature].values[train_indices] for feature in categorical_features]
X_cat_test = [laptops[feature].values[test_indices] for feature in categorical_features]

# Split numerical features
X_num_train = laptops[numerical_features].values[train_indices]
X_num_test = laptops[numerical_features].values[test_indices]

# Split labels
y_train = y_encoded[train_indices]
y_test = y_encoded[test_indices]

# Scale numerical features
scaler = StandardScaler()
X_num_train_scaled = scaler.fit_transform(X_num_train)
X_num_test_scaled = scaler.transform(X_num_test)

# Create and compile model
num_classes = len(le_category.classes_)
model = create_wide_deep_model(
    num_categories=len(categorical_features),
    num_numerical=len(numerical_features),
    num_classes=num_classes
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Add early stopping for better training
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Train model with GPU acceleration
print("\nStarting training with GPU acceleration...")
history = model.fit(
    X_cat_train + [X_num_train_scaled],
    y_train,
    epochs=50,
    batch_size=64,  # Increased batch size for GPU efficiency
    validation_data=(X_cat_test + [X_num_test_scaled], y_test),
    callbacks=[early_stopping],
    verbose=1
)

# 6. Train Nearest Neighbors recommender
# Prepare data for NN (using encoded features)
X_nn = laptops[['CPU_enc', 'GPU_enc', 'RAM', 'Storage', 'Final Price', 'Brand_enc']]
scaler_nn = StandardScaler()
X_nn_scaled = scaler_nn.fit_transform(X_nn)

nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
nn.fit(X_nn_scaled)

# 7. Save everything
joblib.dump({
    'wide_deep_model': model,
    'nn': nn,
    'scaler': scaler,
    'scaler_nn': scaler_nn,
    'le_cpu': le_cpu,
    'le_gpu': le_gpu,
    'le_brand': le_brand,
    'le_category': le_category,
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'laptops': laptops
}, 'lap_rec.joblib')

print('Wide & Deep model and encoders saved as lap_rec.joblib')

