from code_local.datasets.parking import ParkingCocoDataset
#from code_local.datasets.parking import ParkingCocoDataset
from mmdet.datasets.custom import CustomDataset
dataset = ParkingCocoDataset('data/parking/GT_data/',[])
dataset.evaluate([])
print(dataset.get_ann_info(0))

