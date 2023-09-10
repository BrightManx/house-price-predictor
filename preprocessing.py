import joblib
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

'''
# Processing
myHouse = encode(myHouse)
myHouse = process_outliers(myHouse)
myHouse = process_nan(myHouse)

# Feature Augmentation
myHouse = add_city(myHouse)
myHouse = add_closest(myHouse, public_transport, 'closest_public_transport')
myHouse = add_closest(myHouse, schools, 'closest_school', radius=1.0)
myHouse = add_closest(myHouse, universities, 'closest_university')
myHouse = add_closest(myHouse, kindergartens, 'closest_kindergarten')

# Scale
X_myHouse = myHouse.to_numpy()
X_myHouse = scaler.transform(X_myHouse)
'''

def encode(df, istraining = False):
    
    # 1. Garden
    df['garden'] = df.garden.fillna(0)
    df['garden'] = df.garden.astype(float)        

    # 2. Balcony
    df['balcony'] = df.balcony.fillna(0)
    df['balcony'] = df.balcony.astype(float)

    # 3. Conditions
    enc = joblib.load('enc.joblib')
    df['conditions'] = enc.transform(df[['conditions']])

    return df

def process_outliers(df, istraining = False):

    if istraining:
        
        df = df.drop(df.price.nlargest(2).index) # the two prices above 40,000,000
        df = df.drop(df.price.nsmallest(2).index) # the two prices below 1,000

    df.loc[df.energy_efficiency > 1e7, 'energy_efficiency'] = np.nan
    df.loc[df.floor > 37, 'floor'] = np.nan
    df.loc[df.total_floors > 37, 'total_floors'] = np.nan
    df.loc[df.construction_year > 2100, 'construction_year'] = np.nan

    return df

def process_nan(df, istraining = False):

    lm = joblib.load('lm.joblib')
    imputer = joblib.load('imputer.joblib')
    cols = joblib.load('cols.joblib')

    # Rest of NaN's
    df[cols] = imputer.transform(df[cols])

    # Surface
    if any(df.surface.isna()):
        df.loc[df.surface.isna(), 'surface'] = lm.predict(df.loc[df.surface.isna(), ['n_rooms', 'n_bathrooms']]).ravel()
    
    return df

def geo_distance(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    R = 6373.0
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return  R * c

def add_closest(df, targets, header, radius = False):
    
    lat1 = df.latitude.to_numpy()
    lon1 = df.longitude.to_numpy()
    
    distances = np.zeros((lat1.shape[0], targets.shape[0]))

    for i in range(targets.shape[0]):

        lat2 = targets.lat.iloc[i]
        lon2 = targets.lon.iloc[i]

        distances[:, i] = geo_distance(lat1, lon1, lat2, lon2)

    df[header] = np.min(distances, axis = 1)
    if radius: df[header] = (np.min(distances, axis = 1) < radius).astype(float)
        

    return df

def add_city(df):

    city_centers = {
        'Milan' : np.array((45.464098, 9.191926)),      # Piazza del Duomo
        'Venice': np.array((45.434132, 12.338334)),     # Piazza San Marco
        'Rome'  : np.array((41.890210, 12.492231))      # Colosseo
        }
    
    lat1 = df.latitude.to_numpy()
    lon1 = df.longitude.to_numpy()

    distances = np.zeros((lat1.shape[0], len(city_centers)))

    for i, key in enumerate(city_centers.keys()):

        lat2 = city_centers[key][0]
        lon2 = city_centers[key][1]
        
        distances[:, i] = geo_distance(lat1, 
                                       lon1,
                                       city_centers[key][0],
                                       city_centers[key][1])

    df['city'] = distances.argmin(axis = 1)             # Cities as 0,1,2
    df['from_city_center'] = distances.min(axis = 1)    # Distance in km from city center

    return df

def preprocess(df):

    df = encode(df)
    df = process_outliers(df)
    df = process_nan(df)
    df = add_city(df)
    df = df.reindex(sorted(df.columns), axis=1)

    return df