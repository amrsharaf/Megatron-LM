from megatron.data.gpt_dataset import _num_tokens
from megatron.data.indexed_dataset import MMapIndexedDataset
from megatron.data.indexed_dataset import index_file_path
from megatron.utils import print_rank_0
import numpy as np
import json

def count_file_tokens(path):
    indexed_dataset = MMapIndexedDataset.Index(index_file_path(path), skip_warmup=True)
    tokens_per_epoch = _num_tokens(np.arange(indexed_dataset.sizes.shape[0]), indexed_dataset.sizes)
    print_rank_0(' > tokens per epoch: {:,}'.format(tokens_per_epoch))
    return int(tokens_per_epoch)

def main():
    with open('tlgv5.2.0.json', 'r') as f:
        data = json.load(f)
        # we will save the tokens_per_epoch for each dataset in this dict
        tokens_per_epoch_dict = {}
        # print keys in a nice format
        for key in data.keys():
            data_prefix = key.replace('/mnt/stream/', '/turingnorwayeastpremium_data/')
            print_rank_0(' > data prefix: {}'.format(data_prefix))
            tokens_per_epoch = count_file_tokens(data_prefix)
            print_rank_0(' > tokens per epoch: {:,}'.format(tokens_per_epoch))
            tokens_per_epoch_dict[key] = tokens_per_epoch
        # Add entries for the Flan data
        flan_paths = ["/turingnorwayeastpremium_data/datasets/Instruction_finetuning/flanv2_data_fianl/processed_flanv2/binarized/flanv2_cot/_text_document", 
                    "/turingnorwayeastpremium_data/datasets/Instruction_finetuning/flanv2_data_fianl/processed_flanv2/binarized/flanv2_dialog/_text_document", 
                    "/turingnorwayeastpremium_data/datasets/Instruction_finetuning/flanv2_data_fianl/processed_flanv2/binarized/flanv2_flanv1_cappedat1M/_text_document", 
                    "/turingnorwayeastpremium_data/datasets/Instruction_finetuning/flanv2_data_fianl/processed_flanv2/binarized/flanv2_niv2/_text_document", 
                    "/turingnorwayeastpremium_data/datasets/Instruction_finetuning/flanv2_data_fianl/processed_flanv2/binarized/flanv2_t0/_text_document",
                    "/turingnorwayeastpremium_data/datasets/Instruction_finetuning/flanv2_data_fianl/processed_flanv2/binarized_original_flanv2_hf/conceptofmind_cot_submix_original/_text_document",
                    "/turingnorwayeastpremium_data/datasets/Instruction_finetuning/flanv2_data_fianl/processed_flanv2/binarized_original_flanv2_hf/conceptofmind_flan2021_submix_original/_text_document",
                    "/turingnorwayeastpremium_data/datasets/Instruction_finetuning/flanv2_data_fianl/processed_flanv2/binarized_original_flanv2_hf/conceptofmind_t0_submix_original/_text_document",
                    "/turingnorwayeastpremium_data/datasets/Instruction_finetuning/flanv2_data_fianl/processed_flanv2/binarized_original_flanv2_hf/conceptofmind_dialog_submix_original/_text_document",
                    "/turingnorwayeastpremium_data/datasets/Instruction_finetuning/flanv2_data_fianl/processed_flanv2/binarized_original_flanv2_hf/conceptofmind_niv2_submix_original/_text_document",
                    ] 
        for path in flan_paths:
            tokens_per_epoch_dict[path] = count_file_tokens(path)
        # Save the tokens_per_epoch_dict to a file in a readable format
        with open('tokens_per_epoch_dict.json', 'w') as outfile:
            json.dump(tokens_per_epoch_dict, outfile, indent=4) 

if __name__ == "__main__":
    path = '/turingnorwayeastpremium_data/datasets/phi/synthetic_textbooks/book_mergedv4_text_document'
#    path = "/turingnorwayeastpremium_data/datasets/Instruction_finetuning/flanv2_data_fianl/processed_flanv2/binarized_original_flanv2_hf/conceptofmind_niv2_submix_original/_text_document"
    count_file_tokens(path)
#    main()