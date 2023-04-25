import torch
from transformers import StoppingCriteria


class StopWordsCriteria(StoppingCriteria):

    def __init__(self, stop_indices: list):
        self.stop_indices = stop_indices

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # do not support batch inference
        for i in range(len(self.stop_indices)):
            if self.stop_indices[-1-i] != input_ids[0][-1-i]:
                return False
        return True
