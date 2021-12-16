from mmsc.common.registry import registry
from mmsc.datasets.base_dataset_builder import BaseDatasetBuilder
from mmsc.datasets.builders.voxceleb2.dataset import VoxCeleb2Dataset

@registry.register_builder("voxceleb2")
class VoxCeleb2Builder(BaseDatasetBuilder):
    def __init__(self,
                 dataset_type='train', 
                 dataset_name='voxceleb2', 
                 dataset_class=VoxCeleb2Dataset):
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
        return 'configs/datasets/voxceleb2/defaults.yaml'