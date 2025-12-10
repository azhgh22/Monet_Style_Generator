from typing import List
from pathlib import Path
from sklearn.model_selection import train_test_split

class TrainTestSplit:
  def __init__(self, dir_path:str,ext:str='jpg',random_seed:int=42) -> None:
    path = Path(dir_path)
    self.files = list(path.glob("*.jpg"))
    self.random_seed = random_seed

  def split(self, train_p:float,val_p:float,test_p:float)->tuple[List[str],List[str],List[str]]:
    train, temp = train_test_split(self.files, test_size=(val_p+test_p), random_state=42)
    val, test = train_test_split(temp, test_size=test_p/(val_p+test_p), random_state=42)
    return train, val, test