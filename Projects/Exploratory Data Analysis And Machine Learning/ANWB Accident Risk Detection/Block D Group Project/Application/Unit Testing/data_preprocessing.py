import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
import time

class DataPreprocessor:
    def __init__(self, db_params, default_timestamp, landmark_coords, columns_info):
        self.db_params = db_params
        self.default_timestamp = default_timestamp
        self.landmark_coords = landmark_coords
        self.columns_info = columns_info

    """
    A class to handle data preprocessing tasks including reading from a database, 
    cleaning, normalizing, and inserting data back into the database.
    """

    def __init__(self, db_params, default_timestamp, landmark_coords, columns_info):
        """
        Initializes the DataPreprocessor with database parameters, default timestamp, landmark coordinates, and table columns information.

        Args:
            db_params (dict): Database connection parameters.
            default_timestamp (pd.Timestamp): Default timestamp to fill missing datetime values.
            landmark_coords (tuple): Coordinates of a landmark (latitude, longitude).
            columns_info (dict): Dictionary containing columns and their data types for each table.
        """
        self.db_params = db_params
        self.default_timestamp = default_timestamp
        self.landmark_coords = landmark_coords
        self.columns_info = columns_info

    def connect_to_db(self):
        """
        Establish a connection to the database.

        Returns:
            psycopg2.extensions.connection: Database connection object or None if connection fails.
        """
        try:
            conn = psycopg2.connect(**self.db_params)
            return conn
        except Exception as e:
            return None

    def read_table(self, query):
        """
        Read a table from the database into a DataFrame.

        Args:
            query (str): SQL query to fetch data.

        Returns:
            pd.DataFrame: DataFrame containing the fetched data.
        """
        conn = self.connect_to_db()
        if conn is None:
            return pd.DataFrame()

        try:
            df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            return pd.DataFrame()
        finally:
            conn.close()

    def clean_dataframe(self, df, table_name):
        """
        Clean and preprocess the DataFrame according to the table schema.
    
        Args:
            df (pd.DataFrame): DataFrame to be cleaned.
            table_name (str): Name of the table for schema reference.
    
        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        df = df.copy()
        columns_info = self.columns_info[table_name]
    
        lat_col, lon_col = 'latitude', 'longitude'
        event_start_col = next((col for col, dtype in columns_info if dtype == 'TIMESTAMP'), 'event_start')
        event_end_col = event_start_col  # Default end column same as start if not specified
    
        # Ensure event_start and event_end columns are properly handled
        df[event_start_col] = pd.to_datetime(df[event_start_col], errors='coerce', utc=True).fillna(self.default_timestamp)
        df[event_end_col] = pd.to_datetime(df[event_end_col], errors='coerce', utc=True).fillna(self.default_timestamp)
    
        # Check if latitude and longitude columns are present
        if lat_col in df.columns and lon_col in df.columns:
            landmark_latitude, landmark_longitude = self.landmark_coords
            df['distance_to_landmark'] = np.sqrt((df[lat_col] - landmark_latitude) ** 2 + (df[lon_col] - landmark_longitude) ** 2)
        else:
            print("Latitude or Longitude columns are missing.")
            df['distance_to_landmark'] = np.nan  # Ensure the column is added even if data is missing
    
        def time_of_day(hour):
            """Categorize time into periods of the day."""
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:
                return 'Night'
    
        if pd.api.types.is_datetime64_any_dtype(df[event_start_col]):
            df['time_of_day'] = df[event_start_col].dt.hour.apply(time_of_day)
    
        for col, dtype in columns_info:
            if col in df.columns:  # Check if the column is present in the DataFrame
                if dtype in ['REAL', 'INTEGER', 'BIGINT']:
                    df[col] = df[col].fillna(0)
                elif dtype == 'TIMESTAMP':
                    df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).fillna(self.default_timestamp)
                else:
                    df[col] = df[col].fillna('')
            else:
                print(f"Column {col} is missing in the DataFrame.")
                if dtype in ['REAL', 'INTEGER', 'BIGINT']:
                    df[col] = 0  # Add the missing column with default value
                elif dtype == 'TIMESTAMP':
                    df[col] = self.default_timestamp
                else:
                    df[col] = ''
    
        # Ensure the returned DataFrame has the correct columns in order
        df = df[[col for col, _ in columns_info if col in df.columns]]
        return df



    def preprocess_dataframe(self, df, table_name):
        """
        Preprocess the DataFrame by filling missing values and converting data types.
    
        Args:
            df (pd.DataFrame): DataFrame to preprocess.
            table_name (str): Name of the table for schema reference.
    
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        columns_info = self.columns_info[table_name]
        datetime_columns = [col for col, dtype in columns_info if dtype == 'TIMESTAMP']
    
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).fillna(self.default_timestamp)
            else:
                print(f"Warning: Column '{col}' expected as datetime is missing in the DataFrame.")
    
        for col, dtype in columns_info:
            if col in df.columns:  # Only process if the column exists
                if dtype in ['REAL', 'INTEGER', 'BIGINT']:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna('')
            else:
                print(f"Warning: Column '{col}' expected as '{dtype}' is missing in the DataFrame.")
    
        return df


    def insert_data(self, df, insert_query):
        """
        Insert data into the database.

        Args:
            df (pd.DataFrame): DataFrame containing data to insert.
            insert_query (str): SQL insert query with placeholders.
        """
        conn = self.connect_to_db()
        if conn is None:
            return

        try:
            cursor = conn.cursor()
            data_tuples = list(df.itertuples(index=False, name=None))
            execute_values(cursor, insert_query, data_tuples)
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
        finally:
            if conn:
                cursor.close()
                conn.close()


    def fetch_and_preprocess_data(self):
        """
        Fetch and preprocess data from the database.

        Returns:
            tuple: A tuple containing cleaned DataFrames for safe driving, breda road, precipitation, temperature, and greenery.
        """
        # Fetch data using predefined queries
        safe_driving_df = self.read_table(safe_driving_query)
        breda_road_df = self.read_table(breda_road_query)
        precipitation_df = self.read_table(precipitation_query)
        temperature_df = self.read_table(temperature_query)
        greenery_df = self.read_table(greenery_query)

        # Clean and preprocess the data
        safe_driving_df = self.clean_dataframe(safe_driving_df, 'safe_driving')
        breda_road_df = self.clean_dataframe(breda_road_df, 'breda_road')
        precipitation_df = self.clean_dataframe(precipitation_df, 'precipitation')
        temperature_df = self.clean_dataframe(temperature_df, 'temperature')
        greenery_df = self.clean_dataframe(greenery_df, 'greenery')

        safe_driving_df = self.preprocess_dataframe(safe_driving_df, 'safe_driving')
        breda_road_df = self.preprocess_dataframe(breda_road_df, 'breda_road')
        precipitation_df = self.preprocess_dataframe(precipitation_df, 'precipitation')
        temperature_df = self.preprocess_dataframe(temperature_df, 'temperature')
        greenery_df = self.preprocess_dataframe(greenery_df, 'greenery')

        return safe_driving_df, breda_road_df, precipitation_df, temperature_df, greenery_df

    def join_data(self, conn, join_query):
        """
        Perform SQL joins on the database.

        Args:
            conn (psycopg2.extensions.connection): Database connection object.
            join_query (str): SQL join query.

        Returns:
            pd.DataFrame: DataFrame resulting from the join query.
        """
        try:
            cursor = conn.cursor()
            cursor.execute(join_query)
            result_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
            cursor.close()
            return result_df
        except Exception as e:
            return pd.DataFrame()

    def normalize_and_insert_data(self, df, normalize=True, method='minmax'):
        """
        Normalize and insert consolidated data into the database.

        Args:
            df (pd.DataFrame): DataFrame to normalize and insert.
            normalize (bool): Flag indicating whether to normalize the data.
            method (str): Method to use for normalization ('minmax', 'standard', 'robust').
        """
        conn = self.connect_to_db()
        if conn is None:
            return

        try:
            cursor = conn.cursor()

            # Normalize data if required
            if normalize:
                df = self.normalize_dataframe(df, method)

            insert_query = """
            INSERT INTO group24_warehouse.normalised (
                event_start, event_end, duration_seconds, latitude, longitude, speed_kmh, end_speed_kmh, maxwaarde,
                category, incident_severity, is_valid, road_manager_type, road_number, road_name,
                place_name, municipality_name, road_manager_name, breda_begintijd, breda_eindtijd, bgt_status, plus_status,
                bgt_functie, plus_functie, bgt_fysiekvoorkomen, plus_fysiekvoorkomen,
                precip_dtg, dr_pws_10, dr_regenm_10, ww_cor_10, ri_pws_10, ri_regenm_10, temp_dtg, t_dryb_10, tn_10cm_past_6h_10,
                t_dewp_10, t_wetb_10
            ) VALUES %s
            ON CONFLICT DO NOTHING;
            """

            # Insert data in chunks to avoid overwhelming the database
            for chunk in self.get_chunks(df):
                data_tuples = [tuple(x) for x in chunk.to_numpy()]
                execute_values(cursor, insert_query, data_tuples)
                conn.commit()

        except Exception as e:
            if conn:
                conn.rollback()
        finally:
            if conn:
                cursor.close()
                conn.close()

    @staticmethod
    def get_chunks(df, chunk_size=1000):
        """
        Yield successive chunks of DataFrame of a specified size.

        Args:
            df (pd.DataFrame): DataFrame to be chunked.
            chunk_size (int): Size of each chunk.

        Yields:
            pd.DataFrame: DataFrame chunk.
        """
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size]

    @staticmethod
    def normalize_dataframe(df, method):
        """
        Normalize a DataFrame using a specified method.

        Args:
            df (pd.DataFrame): DataFrame to normalize.
            method (str): Method to use for normalization ('minmax', 'standard', 'robust').

        Returns:
            pd.DataFrame: Normalized DataFrame.
        """
        # Define scalers based on the method
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Normalization method '{method}' is not supported.")

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        return df


# Define the queries to fetch data
safe_driving_query = """
SELECT DISTINCT eventid, event_start, event_end, duration_seconds, latitude, longitude, speed_kmh, end_speed_kmh,
       maxwaarde, category, incident_severity, is_valid, road_segment_id, road_manager_type, road_number, 
       road_name, place_name, municipality_name, road_manager_name 
FROM data_lake.safe_driving
WHERE is_valid = True;
"""
breda_road_query = """
SELECT DISTINCT _id, objectbegintijd, objecteindtijd, bgt_status, plus_status, bgt_functie, plus_functie,
       bgt_fysiekvoorkomen, plus_fysiekvoorkomen, wegdeeloptalud, relatievehoogteligging 
FROM data_lake.breda_road;
"""
precipitation_query = """
SELECT DISTINCT dtg, latitude, longitude, altitude, dr_pws_10, dr_regenm_10, ww_cor_10, ri_pws_10, ri_regenm_10 
FROM data_lake.precipitation;
"""
temperature_query = """
SELECT DISTINCT dtg, latitude, longitude, altitude, t_dryb_10, tn_10cm_past_6h_10, t_dewp_10, t_wetb_10 
FROM data_lake.temperature;
"""
greenery_query = """
SELECT DISTINCT _id, _key, namespace, lokaalid, objectbegintijd, objecteindtijd, tijdstipregistratie, eindregistratie,
       lv_publicatiedatum, bronhouder, inonderzoek, relatievehoogteligging, bgt_status, plus_status, bgt_type, plus_type 
FROM data_lake.greenery;
"""

# Database connection parameters
db_params = {
    'host': '194.171.191.226',
    'port': '6379',
    'database': 'postgres',
    'user': 'group24',
    'password': 'blockd_2024group24_77'
}

# Define the columns and their data types for each table
columns_info = {
    "safe_driving": [
        ('eventid', 'INTEGER'),
        ('event_start', 'TIMESTAMP'),
        ('event_end', 'TIMESTAMP'),
        ('duration_seconds', 'REAL'),
        ('latitude', 'REAL'),
        ('longitude', 'REAL'),
        ('speed_kmh', 'REAL'),
        ('end_speed_kmh', 'REAL'),
        ('maxwaarde', 'REAL'),
        ('category', 'VARCHAR'),
        ('incident_severity', 'VARCHAR'),
        ('is_valid', 'BOOLEAN'),
        ('road_segment_id', 'INTEGER'),
        ('road_manager_type', 'VARCHAR'),
        ('road_number', 'VARCHAR'),
        ('road_name', 'VARCHAR'),
        ('place_name', 'VARCHAR'),
        ('municipality_name', 'VARCHAR'),
        ('road_manager_name', 'VARCHAR')
    ],
    "breda_road": [
        ('_id', 'BIGINT'),
        ('objectbegintijd', 'TIMESTAMP'),
        ('objecteindtijd', 'TIMESTAMP'),
        ('bgt_status', 'VARCHAR'),
        ('plus_status', 'VARCHAR'),
        ('bgt_functie', 'VARCHAR'),
        ('plus_functie', 'VARCHAR'),
        ('bgt_fysiekvoorkomen', 'VARCHAR'),
        ('plus_fysiekvoorkomen', 'VARCHAR'),
        ('wegdeeloptalud', 'BIGINT'),
        ('relatievehoogteligging', 'BIGINT'),
        ('latitude', 'REAL'),
        ('longitude', 'REAL')
    ],
    "precipitation": [
        ('dtg', 'TIMESTAMP'),
        ('latitude', 'REAL'),
        ('longitude', 'REAL'),
        ('altitude', 'REAL'),
        ('dr_pws_10', 'REAL'),
        ('dr_regenm_10', 'REAL'),
        ('ww_cor_10', 'REAL'),
        ('ri_pws_10', 'REAL'),
        ('ri_regenm_10', 'REAL')
    ],
    "temperature": [
        ('dtg', 'TIMESTAMP'),
        ('latitude', 'REAL'),
        ('longitude', 'REAL'),
        ('altitude', 'REAL'),
        ('t_dryb_10', 'REAL'),
        ('tn_10cm_past_6h_10', 'REAL'),
        ('t_dewp_10', 'REAL'),
        ('t_wetb_10', 'REAL')
    ],
    "greenery": [
        ('_id', 'BIGINT'),
        ('_key', 'VARCHAR'),
        ('namespace', 'VARCHAR'),
        ('lokaalid', 'VARCHAR'),
        ('objectbegintijd', 'TIMESTAMP'),
        ('objecteindtijd', 'TIMESTAMP'),
        ('tijdstipregistratie', 'TIMESTAMP'),
        ('eindregistratie', 'TIMESTAMP'),
        ('lv_publicatiedatum', 'TIMESTAMP'),
        ('bronhouder', 'VARCHAR'),
        ('inonderzoek', 'BIGINT'),
        ('relatievehoogteligging', 'BIGINT'),
        ('bgt_status', 'VARCHAR'),
        ('plus_status', 'VARCHAR'),
        ('bgt_type', 'VARCHAR'),
        ('plus_type', 'VARCHAR'),
        ('latitude', 'REAL'),
        ('longitude', 'REAL')
    ]
}

# Define default timestamp for missing datetime values
default_timestamp = pd.Timestamp('1970-01-01', tz='UTC')

# Define the landmark coordinates for distance calculation
landmark_coords = (51.5890, 4.7745)

# Create an instance of DataPreprocessor
preprocessor = DataPreprocessor(db_params, default_timestamp, landmark_coords, columns_info)

# Fetch and preprocess the data
safe_driving_df, breda_road_df, precipitation_df, temperature_df, greenery_df = preprocessor.fetch_and_preprocess_data()

# Define the insert queries for each table
insert_safe_driving_query = """
INSERT INTO group24_warehouse.safe_driving (
    eventid, event_start, event_end, duration_seconds, latitude, longitude, 
    speed_kmh, end_speed_kmh, maxwaarde, category, incident_severity, is_valid, 
    road_segment_id, road_manager_type, road_number, road_name, place_name, 
    municipality_name, road_manager_name
) VALUES %s
ON CONFLICT (eventid) DO NOTHING;
"""

insert_breda_road_query = """
INSERT INTO group24_warehouse.breda_road (
    _id, objectbegintijd, objecteindtijd, bgt_status, plus_status, 
    bgt_functie, plus_functie, bgt_fysiekvoorkomen, plus_fysiekvoorkomen, 
    wegdeeloptalud, relatievehoogteligging, latitude, longitude
) VALUES %s
ON CONFLICT (_id) DO NOTHING;
"""

insert_precipitation_query = """
INSERT INTO group24_warehouse.precipitation (
    dtg, latitude, longitude, altitude, dr_pws_10, dr_regenm_10, 
    ww_cor_10, ri_pws_10, ri_regenm_10
) VALUES %s
ON CONFLICT (dtg, latitude, longitude) DO NOTHING;
"""

insert_temperature_query = """
INSERT INTO group24_warehouse.temperature (
    dtg, latitude, longitude, altitude, t_dryb_10, tn_10cm_past_6h_10, 
    t_dewp_10, t_wetb_10
) VALUES %s
ON CONFLICT (dtg, latitude, longitude) DO NOTHING;
"""

insert_greenery_query = """
INSERT INTO group24_warehouse.greenery (
    _id, _key, namespace, lokaalid, objectbegintijd, objecteindtijd, 
    tijdstipregistratie, eindregistratie, lv_publicatiedatum, bronhouder, 
    inonderzoek, relatievehoogteligging, bgt_status, plus_status, 
    bgt_type, plus_type, latitude, longitude
) VALUES %s
ON CONFLICT (_id) DO NOTHING;
"""

# Insert data into respective tables
preprocessor.insert_data(safe_driving_df, insert_safe_driving_query)
preprocessor.insert_data(breda_road_df, insert_breda_road_query)
preprocessor.insert_data(precipitation_df, insert_precipitation_query)
preprocessor.insert_data(temperature_df, insert_temperature_query)
preprocessor.insert_data(greenery_df, insert_greenery_query)

# Define the join queries
join_safe_driving_breda_road_query = """
SELECT sd.eventid, sd.event_start, sd.event_end, sd.duration_seconds, sd.latitude, sd.longitude,
       sd.speed_kmh, sd.end_speed_kmh, sd.maxwaarde, sd.category, sd.incident_severity, sd.is_valid,
       sd.road_segment_id, sd.road_manager_type, sd.road_number, sd.road_name, sd.place_name,
       sd.municipality_name, sd.road_manager_name, br._id AS breda_id, br.objectbegintijd, br.objecteindtijd,
       br.bgt_status, br.plus_status, br.bgt_functie, br.plus_functie, br.bgt_fysiekvoorkomen,
       br.plus_fysiekvoorkomen, br.wegdeeloptalud, br.relatievehoogteligging
FROM group24_warehouse.safe_driving sd
LEFT JOIN group24_warehouse.breda_road br
ON sd.event_start = br.objectbegintijd;
"""

join_safe_driving_precipitation_query = """
SELECT sd.eventid, sd.event_start, sd.event_end, sd.duration_seconds, sd.latitude, sd.longitude,
       sd.speed_kmh, sd.end_speed_kmh, sd.maxwaarde, sd.category, sd.incident_severity, sd.is_valid,
       sd.road_segment_id, sd.road_manager_type, sd.road_number, sd.road_name, sd.place_name,
       sd.municipality_name, sd.road_manager_name, pc.dtg AS precip_dtg, pc.latitude AS precip_latitude,
       pc.longitude AS precip_longitude, pc.altitude AS precip_altitude, pc.dr_pws_10, pc.dr_regenm_10,
       pc.ww_cor_10, pc.ri_pws_10, pc.ri_regenm_10
FROM group24_warehouse.safe_driving sd
LEFT JOIN group24_warehouse.precipitation pc
ON sd.event_start = pc.dtg;
"""

join_temperature_precipitation_query = """
SELECT tp.dtg AS temp_dtg, tp.latitude AS temp_latitude, tp.longitude AS temp_longitude, tp.altitude AS temp_altitude,
       tp.t_dryb_10, tp.tn_10cm_past_6h_10, tp.t_dewp_10, tp.t_wetb_10, pc.dtg AS precip_dtg, pc.latitude AS precip_latitude,
       pc.longitude AS precip_longitude, pc.altitude AS precip_altitude, pc.dr_pws_10, pc.dr_regenm_10, pc.ww_cor_10,
       pc.ri_pws_10, pc.ri_regenm_10
FROM group24_warehouse.temperature tp
LEFT JOIN group24_warehouse.precipitation pc
ON tp.dtg = pc.dtg;
"""

# Connect to the database and perform join operations
conn = preprocessor.connect_to_db()
if conn is not None:
    safe_driving_breda_road = preprocessor.join_data(conn, join_safe_driving_breda_road_query)
    safe_driving_precipitation = preprocessor.join_data(conn, join_safe_driving_precipitation_query)
    temperature_precipitation = preprocessor.join_data(conn, join_temperature_precipitation_query)
    conn.close()

    # Vertically concatenate the DataFrames
    pre_final_data = pd.concat([safe_driving_breda_road, safe_driving_precipitation, temperature_precipitation], ignore_index=True)

# Normalize and insert the consolidated data into the normalised table
preprocessor.normalize_and_insert_data(pre_final_data, normalize=True, method='standard')
