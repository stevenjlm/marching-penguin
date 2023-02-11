import pandas as pd
import copy
from datetime import timedelta
from typing import Union, Tuple, List
import numpy as np

MACHINE_ID = "machineID"
TRAIN_TEST_SPLIT = timedelta(days=90)


class SignalSynthesizerMethod:
    DEL_T = np.timedelta64(3, 'D')
    COLUMNS = ["volt", "rotate", "vibration", "pressure"]


class OptLinearSingleStepMultiMachine(SignalSynthesizerMethod):
    
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
    

class FailureEventWithinDelT(SignalSynthesizerMethod):
    
    def __init__(self, components: List[str]):
        self.components = components
    
    def generate(self, tel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        output_signal = pd.DataFrame()
        
        machine_ids = tel[MACHINE_ID].unique()
        for machine_id in machine_ids:
            m_signal = self.generate_machine_signal(machine_id, tel, events)
            output_signal = pd.concat([output_signal, m_signal])
        # output_signal = output_signal.sort_values(by="datetime")
        return output_signal
        
    def generate_machine_signal(self, machine_id: int, tel: pd.DataFrame, events: pd.DataFrame):
        machine_tel = copy.deepcopy(tel.loc[tel[MACHINE_ID] == machine_id])
        for comp in self.components:
            new_column = np.full((len(machine_tel.index)), False)
            machine_tel_times = machine_tel["datetime"].to_numpy()
            
            machine_events = events.loc[events["machineID"] == machine_id]
            component_failures = machine_events.loc[machine_events["failure_" + comp] == True]
            
            t_failures = component_failures["datetime"]
            t_failures = t_failures.to_numpy()
            for t in t_failures:
                delta_t = t - machine_tel_times
                in_window = np.logical_and(np.timedelta64(0, 'D') <= delta_t, delta_t <= self.DEL_T)
                new_column = np.logical_or(in_window, new_column)

            machine_tel["fail_window_" + comp] = new_column
            machine_tel["fail_window_" + comp] = machine_tel["fail_window_" + comp].astype(int)
        return machine_tel


class AddErrorAndMaintainCols(SignalSynthesizerMethod):

    def __init__(self) -> None:
        super().__init__()
        self.columns = ["error" + str(i) for i in range(1,6)]
        self.columns.extend(["maint_comp" + str(i) for i in range(1,5)])
        return

    def generate(self, raw_signal: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        new_cols = events[self.columns]
        raw_signal = raw_signal.join(new_cols)
        return raw_signal


class OneHotColumn(SignalSynthesizerMethod):
    
    def __init__(self, col_name: str) -> None:
        super().__init__()
        self.col_name = col_name
        return
    
    def generate(self, raw_signal: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        y = pd.get_dummies(raw_signal[self.col_name])
        raw_signal = raw_signal.drop(self.col_name,axis = 1)
        raw_signal = raw_signal.join(y)
        return raw_signal


class MachineSignalSynth:
    
    def __init__(self,
                 tel: pd.DataFrame,
                 events: pd.DataFrame,
                 methods: List[Union[OptLinearSingleStepMultiMachine,
                               FailureEventWithinDelT,
                               OneHotColumn]]):
        self.methods = methods
        self.tel = tel
        self.events = events
        
    def generate(self) -> pd.DataFrame:
        new_signal =  self.tel
        for method in self.methods:
            new_signal = method.generate(new_signal, self.events)
        return new_signal
