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


class MachineSignalSynth:
    
    def __init__(self,
                 tel: pd.DataFrame,
                 events: pd.DataFrame,
                 methods: List[Union[JoinColumns, OneHotColumn]]):
        self.methods = methods
        self.tel = tel
        self.events = events
        
    def generate(self) -> pd.DataFrame:
        new_signal =  self.tel
        for method in self.methods:
            new_signal = method.generate(new_signal, self.events)
        new_signal = new_signal.sort_values(by="datetime")
        return new_signal
