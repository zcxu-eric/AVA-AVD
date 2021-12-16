import warnings
from mmsc.common.sample import Sample
from mmsc.common.registry import registry
from mmsc.datasets.base_dataset_builder import BaseDatasetBuilder
from mmsc.datasets.builders.avaavd.dataset import AVAAVDDataset

@registry.register_builder("avaavd")
class AVAAVDBuilder(BaseDatasetBuilder):
    def __init__(self,
                 dataset_type='val', 
                 dataset_name='avaavd', 
                 dataset_class=AVAAVDDataset):
        super().__init__(dataset_name, dataset_class)
        self.dataset_class = dataset_class
    
    def load(self, config, dataset_type):
        if dataset_type in ['val', 'test']:
            self.config = config
            self.dataset = self.dataset_class(config, dataset_type)
            self.dataset.load()
            return self.dataset
        
    def build(self, config, dataset_type):
        pass

    @classmethod
    def config_path(cls):
        return 'configs/datasets/avaavd/defaults.yaml'