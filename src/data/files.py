import io
import boto3
import pandas as pd

EVENTS = "s3://pmpf-data/iot_pmfp_labels.feather"
TELEMETRY = "s3://pmpf-data/iot_pmfp_data.feather"

class Data:
    @staticmethod
    def read_feather_file_from_s3(s3_url: str) -> pd.DataFrame:
        assert s3_url.startswith("s3://")
        bucket_name, key_name = s3_url[5:].split("/", 1)

        s3 = boto3.client('s3')
        retr = s3.get_object(Bucket=bucket_name, Key=key_name)
        
        return pd.read_feather(io.BytesIO(retr['Body'].read()))
    
    @classmethod
    def get_events(cls) -> pd.DataFrame:
        return cls.read_feather_file_from_s3(EVENTS)

    @classmethod
    def get_telemetry(cls) -> pd.DataFrame:
        return cls.read_feather_file_from_s3(TELEMETRY)
