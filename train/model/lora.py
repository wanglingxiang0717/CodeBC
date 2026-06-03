from model.base_model import CL_Base_Model
import os
import time
from utils.utils import print_rank_0


class lora(CL_Base_Model):
    def __init__(self,
                 model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)

    
    def save_model(self, i_task):
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(i_task))
            if not os.path.exists(peft_model_id):
                os.makedirs(peft_model_id)
            self.model.save_pretrained(peft_model_id)  
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(f'Sucessfully saving the final model to {peft_model_id}', self.args.global_rank)
        
    def save_task_model(self, task, round):
        self.args.task_output_dir = os.path.join(self.args.output_dir, task)
        if self.args.task_output_dir is not None:
            print_rank_0('saving model to ' + self.args.task_output_dir + "/" + str(round) + '...', self.args.global_rank)

        if self.args.global_rank == 0:
            # save_hf_format_task(self.model, self.tokenizer, self.args, sub_folder=str(round))
            peft_model_id = os.path.join(self.args.task_output_dir, str(round))
            if not os.path.exists(peft_model_id):
                os.makedirs(peft_model_id)
            self.model.save_pretrained(peft_model_id)  
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(f'Sucessfully saving the final model to {peft_model_id}', self.args.global_rank)

        # if self.args.zero_stage == 3:
        #     # For zero stage 3, each gpu only has a part of the model, so we need a special save function
        #     save_zero_three_model(self.model,
        #                           self.args.global_rank,
        #                           self.args.task_output_dir,
        #                           zero_stage=self.args.zero_stage,
        #                           sub_folder=str(round))
        # print_rank_0('Sucessful saving model after round {}'.format(round), self.args.global_rank)
            
