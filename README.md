# CodeBC

CodeBC: A More Secure Large Language Model for Smart Contract Code Generation in Blockchain

  This project temporarily opens the training dataset, test dataset, and generated results, and we will follow up by disclosing our model parameters

### Demo



https://github.com/user-attachments/assets/094a48ad-2d52-4afd-bb9b-f1733759f461



### Directory Structure
<!-- eg: -->

```
filetree 
├── /dataset/               
│  ├── /result_dataset/
│  │  ├── /python,C,C++/                 Task of SVEN 
│  │  │  ├──/prompt/                       Prompts of the tasks
│  │  │  ├──/codellama/                    Test results of base model codellama
│  │  │  ├──/result_with_bug_tag/          Test results with addition of a < vulnerable > tag to the original prompt
│  │  │  └──/result_with_safe_tag_25/      Test results with addition of a < security > tag to the original prompt
│  │  ├── /solidity/                     Task of CodeBC
│  │  │  ├──/BlockChain-Humaneval/         Prompts and answers of the tasks
│  │  │  ├──/codellama/                    Test results of base model codellama
│  │  │  ├──/codegen25/                    Test results of base model codegen
│  │  │  ├──/deepseek_coder/               Test results of base model deepseek_coder
│  │  │  ├──/CodeBC-CI/                    Test results of CodeBC-CI
│  │  │  ├──/CodeBC-VD/                    Test results of CodeBC-VD
│  │  │  └──/CodeBC/                       Test results of CodeBC
│  ├── /train_dataset/
│  │  ├── /SASCsmall-F/                
│  │  ├── /SASCsmall-F for CI/                    
│  │  └── /SASCsmall-F for VD/
├── /train/  codebc train code   
```

### Performance Improvement of Stage Model on Blockchain-HumanEval Dataset
This section analyzes the performance of stage models on the 'Address' task, which involves completing a library for address manipulation in Solidity. The task is represented in the Blockchain-HumanEval dataset. The instructions specify the Solidity version, provide an overview of the contract's functionality, and give details about each function. In the code section, annotations are removed, and only the code is included as the target for generation.
![img 1](https://github.com/wanglingxiang1298/CodeBC/blob/master/pic/Blockchain-HumanEval-Address_page-0001.jpg)

The performance of the CodeLlama model on the 'Address' task is shown in the figure.
1. Functions that should be declared as internal are mistakenly declared as public, which could affect access control and data integrity.
2. The contract uses a direct, complete rollback strategy to handle errors, unlike the target code, which applies different rollback strategies based on return results. This approach may cause irreversible operations, preventing the previous state from being restored, and could make the system vulnerable to denial-of-service attacks.
3. The `functionCall()` function is incorrectly declared as a public view function. While it should allow only read operations, the statement `target.call(data)` inside the function modifies the state, which prevents the contract from compiling.
![img 2](https://github.com/wanglingxiang1298/CodeBC/blob/master/pic/basemodel-result_page-0001.jpg)

The performance of the CodeBC-CI model on the 'Address' task is shown in the figure. The results show slight differences in function implementations, but overall functionality remains the same.
Function declarations have been improved, with all functions now correctly declared as internal. However, the rollback strategy has not been improved. Additionally, the `functionCall()` function incorrectly declares a return type of `bytes memory` but returns a `bool` from `target.call(data)`, causing the code to fail to compile.
![img 3](https://github.com/wanglingxiang1298/CodeBC/blob/master/pic/CodeBC-CI-result_page-0001.jpg)

The performance of the CodeBC-VD model on the 'Address' task is shown in the figure. The results show that the model has improved the security of various functionalities. Functions that should be internal are now correctly specified. The rollback strategy has also been improved. Additionally, the generated contract code is error-free and can be compiled and executed successfully.
![img 4](https://github.com/wanglingxiang1298/CodeBC/blob/master/pic/CodeBC-VD-result_page-0001.jpg)

The performance of the CodeBC model on the 'Address' task is shown in the figure. Compared to CodeBC-VD, which already demonstrates excellent functionality and security, CodeBC does not show significant improvements. However, some redundant code has been removed, indicating an improvement in the model's understanding of instructions.
![img 5](https://github.com/wanglingxiang1298/CodeBC/blob/master/pic/CodeBC-result_page-0001.jpg)

### Train CodeBC
```
conda create -n codebc python==3.10
cd train
pip install -r requirements.txt
sh scripts/train_seq_cl.sh
```
simple test
```
python simple_model_test.py
```

### Reference
<!-- links -->


- [SVEN](https://github.com/eth-sri/sven)  
- [Slither Audited Smart Contracts](https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts)
