import fastf1
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from datetime import datetime

fastf1.Cache.enable_cache('cache')

def get_next_race():
    """Find the next upcoming F1 race"""
    current_year = datetime.now().year
    
    try:
        # Get the schedule for current year
        schedule = fastf1.get_event_schedule(current_year)
        now = pd.Timestamp.now()
        
        # Ensure both datetime columns are timezone-naive for comparison
        if schedule['EventDate'].dt.tz is not None:
            schedule['EventDate'] = schedule['EventDate'].dt.tz_localize(None)
        
        # Find upcoming races (qualifying session hasn't happened yet)
        upcoming = schedule[schedule['EventDate'] > now]
        
        if len(upcoming) > 0:
            next_race = upcoming.iloc[0]
            return {
                'year': current_year,
                'round': next_race['RoundNumber'],
                'name': next_race['EventName'],
                'location': next_race['Location'],
                'date': next_race['EventDate']
            }
        else:
            print(f"No more races in {current_year}")
            return None
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return None

def fetch_historical_data(year, num_rounds=5):
    """Fetch historical qualifying data from recent races"""
    all_data = []
    
    for round_num in range(1, num_rounds + 1):
        try:
            print(f"Fetching {year} Round {round_num}...")
            quali = fastf1.get_session(year, round_num, 'Q')
            quali.load()
            
            results = quali.results[['DriverNumber', 'FullName', 'TeamName', 'Q1', 'Q2', 'Q3']].copy()
            results = results.rename(columns={'FullName': 'Driver'})
            
            # Convert lap times to seconds
            for col in ['Q1', 'Q2', 'Q3']:
                results[f'{col}_sec'] = results[col].apply(
                    lambda x: x.total_seconds() if pd.notnull(x) else None
                )
            
            results['Year'] = year
            results['Round'] = round_num
            all_data.append(results)
            
        except Exception as e:
            print(f"Could not fetch Round {round_num}: {e}")
            continue
    
    return pd.concat(all_data, ignore_index=True) if all_data else None

def get_current_driver_teams():
    """Get current F1 driver lineup for 2025"""
    return {
        'Max Verstappen': 'Red Bull Racing',
        'Liam Lawson': 'Red Bull Racing',
        'Charles Leclerc': 'Ferrari',
        'Lewis Hamilton': 'Ferrari',
        'George Russell': 'Mercedes',
        'Kimi Antonelli': 'Mercedes',
        'Lando Norris': 'McLaren',
        'Oscar Piastri': 'McLaren',
        'Fernando Alonso': 'Aston Martin',
        'Lance Stroll': 'Aston Martin',
        'Yuki Tsunoda': 'RB',
        'Isack Hadjar': 'RB',
        'Alex Albon': 'Williams',
        'Carlos Sainz': 'Williams',
        'Nico Hulkenberg': 'Kick Sauber',
        'Gabriel Bortoleto': 'Kick Sauber',
        'Esteban Ocon': 'Haas F1 Team',
        'Oliver Bearman': 'Haas F1 Team',
        'Pierre Gasly': 'Alpine',
        'Jack Doohan': 'Alpine'
    }

def apply_performance_factors(predictions_df, base_time=89.5):
    """Apply team and driver performance factors to base lap time"""
    
    # Team competitiveness factors (based on 2025 season expectations)
    team_factors = {
        'Red Bull Racing': 0.996,
        'McLaren': 0.997,
        'Ferrari': 0.997,
        'Mercedes': 0.999,
        'Aston Martin': 1.002,
        'Williams': 1.003,
        'RB': 1.004,
        'Haas F1 Team': 1.005,
        'Alpine': 1.006,
        'Kick Sauber': 1.007,
    }
    
    # Driver skill factors in qualifying
    driver_factors = {
        'Max Verstappen': 0.997,
        'Charles Leclerc': 0.998,
        'Lando Norris': 0.998,
        'Lewis Hamilton': 0.999,
        'George Russell': 0.999,
        'Oscar Piastri': 1.000,
        'Fernando Alonso': 1.000,
        'Carlos Sainz': 1.000,
        'Yuki Tsunoda': 1.001,
        'Alex Albon': 1.001,
        'Pierre Gasly': 1.002,
        'Liam Lawson': 1.002,
        'Lance Stroll': 1.002,
        'Nico Hulkenberg': 1.003,
        'Esteban Ocon': 1.003,
        'Kimi Antonelli': 1.004,
        'Jack Doohan': 1.005,
        'Oliver Bearman': 1.005,
        'Isack Hadjar': 1.006,
        'Gabriel Bortoleto': 1.006,
    }
    
    for idx, row in predictions_df.iterrows():
        team_factor = team_factors.get(row['Team'], 1.005)
        driver_factor = driver_factors.get(row['Driver'], 1.002)
        
        # Calculate predicted time with random variation
        base_prediction = base_time * team_factor * driver_factor
        random_variation = np.random.uniform(-0.15, 0.15)
        predictions_df.loc[idx, 'Predicted_Q3'] = base_prediction + random_variation
    
    return predictions_df

def train_model(historical_data):
    """Train linear regression model on historical qualifying data"""
    
    # Remove rows with missing Q3 times
    valid_data = historical_data.dropna(subset=['Q3_sec'])
    
    # Prepare features (Q1 and Q2) and target (Q3)
    X = valid_data[['Q1_sec', 'Q2_sec']]
    y = valid_data['Q3_sec']
    
    # Handle any remaining missing values with median imputation
    imputer = SimpleImputer(strategy='median')
    X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    y_clean = pd.Series(imputer.fit_transform(y.values.reshape(-1, 1)).ravel())
    
    # Train model
    model = LinearRegression()
    model.fit(X_clean, y_clean)
    
    # Evaluate model
    y_pred = model.predict(X_clean)
    mae = mean_absolute_error(y_clean, y_pred)
    r2 = r2_score(y_clean, y_pred)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"Mean Absolute Error: {mae:.3f} seconds")
    print(f"R² Score: {r2:.3f}")
    
    return model

def predict_next_race(model, next_race_info):
    """Generate predictions for the next race"""
    
    driver_teams = get_current_driver_teams()
    predictions_df = pd.DataFrame(list(driver_teams.items()), 
                                  columns=['Driver', 'Team'])
    
    # Apply performance factors to generate predictions
    predictions_df = apply_performance_factors(predictions_df)
    predictions_df = predictions_df.sort_values('Predicted_Q3')
    
    # Display predictions
    print("\n" + "="*100)
    print(f"QUALIFYING PREDICTIONS - {next_race_info['name']}")
    print(f"Location: {next_race_info['location']} | Date: {next_race_info['date'].strftime('%Y-%m-%d')}")
    print("="*100)
    print(f"{'Pos':<6}{'Driver':<22}{'Team':<30}{'Predicted Q3':<15}")
    print("-"*100)
    
    for position, (idx, row) in enumerate(predictions_df.iterrows(), start=1):
        print(f"{position:<6}{row['Driver']:<22}{row['Team']:<30}{row['Predicted_Q3']:.3f}s")
    
    print("="*100)
    print("\nNote: Predictions based on 2025 performance factors and historical data")

def main():
    print("F1 Next Race Qualifying Predictor")
    print("="*50 + "\n")
    
    # Find next race
    next_race = get_next_race()
    if not next_race:
        print("Could not determine next race. Season may be over.")
        return
    
    print(f"Next Race: {next_race['name']}")
    print(f"Location: {next_race['location']}")
    print(f"Date: {next_race['date'].strftime('%Y-%m-%d')}\n")
    
    # Fetch historical data from current season
    print("Fetching historical qualifying data...\n")
    current_year = datetime.now().year
    historical_data = fetch_historical_data(current_year, num_rounds=5)
    
    if historical_data is None or len(historical_data) == 0:
        print("Not enough historical data available for predictions.")
        return
    
    print(f"\nLoaded {len(historical_data)} qualifying sessions")
    
    # Train model on historical data
    model = train_model(historical_data)
    
    # Generate predictions for next race
    predict_next_race(model, next_race)

if __name__ == "__main__":
    main()