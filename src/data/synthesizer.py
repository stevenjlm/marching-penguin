import pandas as pd
import copy
from datetime import timedelta

class LinearSingleStep:
    DEL_T = timedelta(days=3)
    
    def generate(self, tel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        df = copy.deepcopy(tel)
        fail_win_values = [False for _ in range(len(df.index))]
        failures = events.loc[events["failure"] == True]
        t_failures = failures["datetime"]
        
        """
        def in_range(row):
            t = row["datetime"]
            for t_fail in t_failures:
                if abs(t - t_fail) <= self.DEL_T:
                    row.at["fail_window"] = True
            return row

        df.apply(in_range, axis=1)
        """
        
        idx = 0
        for index, row in df.iterrows():
            t = row["datetime"]
            for t_fail in t_failures:
                if abs(t - t_fail) <= self.DEL_T:
                    fail_win_values[idx] = True
            idx += 1
        
        df["fail_window"] = fail_win_values
        return df


class MachineSignalSynth:
    
    def __init__(self, tel: pd.DataFrame, events: pd.DataFrame, method: LinearSingleStep):
        self.method = method
        self.tel = tel
        self.events = events
        
    def generate(self) -> pd.DataFrame:
        machine_signal = self.method.generate(self.tel, self.events)
        return machine_signal