from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import joblib
import os
import numpy as np
import pandas as pd

# Create your views here.

def home(request):
    return render(request, 'frontend/home.html')

# Load model once at module level
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lap_rec.joblib')
if os.path.exists(MODEL_PATH):
    model_data = joblib.load(MODEL_PATH)
else:
    model_data = None

@csrf_exempt
def recommend_laptops(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            use_case = data.get('use_case', 'general use')
            max_budget = int(data.get('max_budget', 1000))
            preferred_brand = data.get('preferred_brand', '').strip()
            if not model_data:
                return JsonResponse({'error': 'Model not loaded.'}, status=500)
            laptops = model_data['laptops']
            filtered = laptops[laptops['Final Price'] <= max_budget]
            if preferred_brand:
                filtered = filtered[filtered['Brand'].str.lower() == preferred_brand.lower()]
            # Prioritize premium for high budgets
            results = None
            if max_budget > 1800:
                premium = filtered[filtered['category'] == 'premium'].sort_values('Final Price', ascending=False)
                if not premium.empty:
                    results = premium.head(5)
                else:
                    highend = filtered[filtered['category'] == 'high-end editing'].sort_values('Final Price', ascending=False)
                    if not highend.empty:
                        results = highend.head(5)
            if results is None or results.empty:
                # Fallback to use_case or general
                if use_case != 'general use':
                    filtered = filtered[filtered['category'] == use_case]
                filtered = filtered.sort_values('Final Price', ascending=False)
                results = filtered.head(5)
            recs = []
            similar = []
            for idx, (i, laptop) in enumerate(results.iterrows()):
                rec = {
                    'Laptop': str(laptop['Laptop']),
                    'Brand': str(laptop['Brand']),
                    'CPU': str(laptop['CPU']),
                    'RAM': int(laptop['RAM']) if not pd.isnull(laptop['RAM']) else None,
                    'Storage': int(laptop['Storage']) if not pd.isnull(laptop['Storage']) else None,
                    'Storage_type': str(laptop.get('Storage type', '')),
                    'GPU': str(laptop['GPU']) if laptop['GPU'] else 'Integrated',
                    'Screen': str(laptop.get('Screen', '')),
                    'Price': float(laptop['Final Price']) if not pd.isnull(laptop['Final Price']) else None,
                    'Category': str(laptop['category']),
                }
                recs.append(rec)
                # For the top recommendation, find similar laptops
                if idx == 0 and 'nn' in model_data:
                    nn = model_data['nn']
                    scaler_nn = model_data['scaler_nn']
                    X_nn = model_data['laptops'][['CPU_enc', 'GPU_enc', 'RAM', 'Storage', 'Final Price', 'Brand_enc']]
                    top_vec = X_nn.loc[i].values.reshape(1, -1)
                    top_vec_scaled = scaler_nn.transform(top_vec)
                    dists, indices = nn.kneighbors(top_vec_scaled, n_neighbors=4)  # 1st is itself
                    for j in indices[0][1:]:
                        sim_lap = model_data['laptops'].iloc[j]
                        similar.append({
                            'Laptop': str(sim_lap['Laptop']),
                            'Brand': str(sim_lap['Brand']),
                            'CPU': str(sim_lap['CPU']),
                            'RAM': int(sim_lap['RAM']) if not pd.isnull(sim_lap['RAM']) else None,
                            'Storage': int(sim_lap['Storage']) if not pd.isnull(sim_lap['Storage']) else None,
                            'Storage_type': str(sim_lap.get('Storage type', '')),
                            'GPU': str(sim_lap['GPU']) if sim_lap['GPU'] else 'Integrated',
                            'Screen': str(sim_lap.get('Screen', '')),
                            'Price': float(sim_lap['Final Price']) if not pd.isnull(sim_lap['Final Price']) else None,
                            'Category': str(sim_lap['category']),
                        })
            return JsonResponse({'recommendations': recs, 'similar': similar})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=405)
