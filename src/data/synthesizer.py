import pandas as pd
import copy
from datetime import timedelta
from typing import Union, Tuple, List
import numpy as np

MACHINE_ID = "machineID"
TRAIN_TEST_SPLIT = timedelta(days=90)


class SignalSynthesizerMethod:
    DEL_T = np.timedelta64(0, 'D')
    COLUMNS = ["volt", "rotate", "vibration", "pressure"]


class JoinColumns(SignalSynthesizerMethod):

    def __init__(self, columns: List[str]) -> None:
        super().__init__()
        # self.columns = ["error" + str(i) for i in range(1,6)]
        # self.columns.extend(["maint_comp" + str(i) for i in range(1,5)])
        self.columns_to_join = columns
        return

    def generate(self, raw_signal: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        new_cols = events[self.columns_to_join]
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


class ConvolveBoleanColumns(SignalSynthesizerMethod):
    DEFAULT_DELTA_T = 3

    def __init__(self, cols: List[str], in_place: List[bool], label: str, delta_ts: List[int] = None) -> None:
        super().__init__()
        self.col_names = cols
        self.in_place = in_place
        self.delta_ts = delta_ts
        self.label = label

    def generate(self, raw_signal: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """
        Convolutions need to happen per machine, one at a time.
        """
        machine_ids = raw_signal[MACHINE_ID].unique()
        output_signal = pd.DataFrame()
        for machine_id in machine_ids:
            machine_signal = raw_signal.loc[raw_signal[MACHINE_ID] == machine_id]
            machine_signal = self.generate_machine(machine_signal)
            output_signal = pd.concat([output_signal, machine_signal])
        return output_signal

    def generate_machine(self, raw_signal: pd.DataFrame) -> pd.DataFrame:
        for idx_col in range(len(self.col_names)):
            col_name = self.col_names[idx_col]
            in_place = self.in_place[idx_col]
            if self.delta_ts:
                del_t = self.delta_ts[idx_col]
            else:
                del_t = self.DEFAULT_DELTA_T

            numpy_column_values = raw_signal[col_name].to_numpy()
            convolution_filter = np.ones(2 * del_t)
            if col_name==self.label:
                convolution_filter[del_t:] = 0
            else:
                convolution_filter[:del_t-1] = 0
            column_result = np.convolve(numpy_column_values, convolution_filter, 'same')

            if in_place:
                new = {col_name: column_result}
                raw_signal = raw_signal.assign(**new)
            else:
                new = {col_name + "_in_past_" + str(del_t): column_result}
                raw_signal = raw_signal.assign(**new)
        return raw_signal


class MachineSignalSynth:
    
    def __init__(self,
                 tel: pd.DataFrame,
                 events: pd.DataFrame,
                 methods: List[Union[JoinColumns, OneHotColumn, ConvolveBoleanColumns]]):
        self.methods = methods
        self.tel = tel
        self.events = events
        
    def generate(self) -> pd.DataFrame:
        new_signal =  self.tel
        for method in self.methods:
            new_signal = method.generate(new_signal, self.events)
        new_signal = new_signal.sort_values(by="datetime")
        return new_signal
