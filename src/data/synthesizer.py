import pandas as pd
import copy
from datetime import timedelta
from typing import Union, Tuple
import numpy as np

MACHINE_ID = "machineID"
TRAIN_TEST_SPLIT = timedelta(days=90)


class LinearSingleStepSingleMachine:
    DEL_T = timedelta(days=3)
    
    def generate(self, tel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        df = copy.deepcopy(tel)
        fail_win_values = [False for _ in range(len(df.index))]
        failures = events.loc[events["failure"] == True]
        t_failures = failures["datetime"]
        
        idx = 0
        for index, row in df.iterrows():
            t = row["datetime"]
            for t_fail in t_failures:
                if abs(t - t_fail) <= self.DEL_T:
                    fail_win_values[idx] = True
            idx += 1
        
        df["fail_window"] = fail_win_values
        return df


class LinearSingleStepMultiMachine:
    DEL_T = timedelta(days=3)
    
    def generate(self, tel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        df = copy.deepcopy(tel)
        fail_win_values = [False for _ in range(len(df.index))]
        failures = events.loc[events["failure"] == True]
        
        idx = 0
        for index, row in df.iterrows():
            t = row["datetime"]
            machine_id = row[MACHINE_ID]
            t_failures = failures.loc[failures["machineID"] == machine_id]["datetime"]
            for t_fail in t_failures:
                if abs(t - t_fail) <= self.DEL_T:
                    fail_win_values[idx] = True
            idx += 1
        
        df["fail_window"] = fail_win_values
        return df


class OptLinearSingleStepMultiMachine:
    DEL_T = np.timedelta64(3, 'D')
    COLUMNS = ["volt", "rotate", "vibration", "pressure"]
    
    def generate(self, tel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        failures = events.loc[events["failure"] == True]
        
        return_df = pd.DataFrame()
        machine_ids = tel[MACHINE_ID].unique()
        for machine_id in machine_ids:
            df = copy.deepcopy(tel.loc[tel[MACHINE_ID] == machine_id])
            fails_in_window = np.full((len(df.index)), False)
            telemetric_times = df["datetime"].to_numpy()
            t_failures = failures.loc[failures["machineID"] == machine_id]["datetime"]
            t_failures = t_failures.to_numpy()
            for t in t_failures:
                delta_t = abs(telemetric_times -t)
                # print(delta_t)
                in_window = delta_t <= self.DEL_T
                fails_in_window = np.logical_or(in_window, fails_in_window)
            
            df["fail_window"] = fails_in_window
            for col in self.COLUMNS:
                df["std_" + col] = df[col].rolling(window=96, min_periods=1).std()
                df["mean_" + col] = df[col].rolling(window=96, min_periods=1).mean()
            return_df = pd.concat([return_df, df])
        return return_df


class TimeSplit:
    
    def __init__(self):
        self.train_signal = None
        self.test_signal = None

    def split(self, signal: pd.DataFrame) -> None:
        
        t_max_tele = signal["datetime"].max()
        t_seperator = t_max_tele - TRAIN_TEST_SPLIT
        
        self.train_signal = signal.loc[signal["datetime"] <= t_seperator]
        self.test_signal = signal.loc[signal["datetime"] > t_seperator]

        return self.train_signal, self.test_signal        


class MachineSignalSynth:
    
    def __init__(self,
                 tel: pd.DataFrame,
                 events: pd.DataFrame,
                 method: Union[LinearSingleStepSingleMachine,
                               LinearSingleStepMultiMachine,
                               OptLinearSingleStepMultiMachine],
                 splitter: Union[TimeSplit]=None):
        self.method = method
        self.tel = tel
        self.events = events
        self.splitter = splitter
        
    def generate(self) -> pd.DataFrame:
        machine_signal = self.method.generate(self.tel, self.events)
        return machine_signal
    
    def split_signal(self, signal: pd.DataFrame) -> Tuple[pd.DataFrame]:
        train_sig, test_sig = self.splitter.split(signal)
        return train_sig, test_sig
