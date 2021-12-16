from mmsc.common.registry import registry
from mmsc.datasets.base_dataset_builder import BaseDatasetBuilder
from mmsc.datasets.builders.voxceleb1.dataset import VoxCeleb1Dataset

@registry.register_builder("voxceleb1")
class VoxCeleb1Builder(BaseDatasetBuilder):
    def __init__(self,
                 dataset_type='train', 
                 dataset_name='voxceleb1', 
                 dataset_class=VoxCeleb1Dataset):
        super().__init__(dataset_name, dataset_class)
        self.dataset_class = dataset_class
    
    def load(self, config, dataset_type):
        if dataset_type in ['train']:
            self.config = config
            self.dataset = self.dataset_class(config, dataset_type)
            self.dataset.load()
            return self.dataset
        
    def build(self, config, dataset_type):
        pass

    @classmethod
    def config_path(cls):
        return 'configs/datasets/voxceleb1/defaults.yaml'