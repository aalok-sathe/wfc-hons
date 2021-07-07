#    conda activate wfc
#     cd ~/wfc-hons
    
#     CUDA_VISIBLE_DEVICES=2     ./run_wfc.py --model_name_or_path aloxatel/bert-base-mnli --do_train --do_eval --output_dir out_mar16     --top_k 1 --logging_steps 50 --num_train_epochs 3 --warmup_steps 0 --gradient_accumulation_steps 4     --learning_rate 0.0001  --overwrite_output_dir --maxnum 1700 
    
    
          
#     conda activate wfc
#     cd ~/wfc-hons
    
#     CUDA_VISIBLE_DEVICES=3     ./run_wfc.py --model_name_or_path aloxatel/bert-base-mnli --do_train --do_eval --output_dir out_mar16     --top_k 1 --logging_steps 50 --num_train_epochs 3 --warmup_steps 0 --gradient_accumulation_steps 4     --learning_rate 5e-05  --overwrite_output_dir --maxnum 1700 
    
    
          
    conda activate wfc
    cd ~/wfc-hons
    
    CUDA_VISIBLE_DEVICES=0,1,2     ./run_wfc.py --model_name_or_path aloxatel/bert-base-mnli --do_train --do_eval --output_dir out_fever_dbg     --top_k 5 --logging_steps 50 --num_train_epochs 3 --warmup_steps 0 --gradient_accumulation_steps 4     --learning_rate 0.0001  --overwrite_output_dir --maxnum 30_000
    
    
          
#     conda activate wfc
#     cd ~/wfc-hons
    
#     CUDA_VISIBLE_DEVICES=3     ./run_wfc.py --model_name_or_path aloxatel/bert-base-mnli --do_train --do_eval --output_dir out_mar16     --top_k 3 --logging_steps 50 --num_train_epochs 3 --warmup_steps 0 --gradient_accumulation_steps 4     --learning_rate 5e-05  --overwrite_output_dir --maxnum 1700 
    
    
          
#     conda activate wfc
#     cd ~/wfc-hons
    
#     CUDA_VISIBLE_DEVICES=2     ./run_wfc.py --model_name_or_path aloxatel/bert-base-mnli --do_train --do_eval --output_dir out_mar16     --top_k 5 --logging_steps 50 --num_train_epochs 3 --warmup_steps 0 --gradient_accumulation_steps 4     --learning_rate 0.0001  --overwrite_output_dir --maxnum 1700 
    
    
          
#     conda activate wfc
#     cd ~/wfc-hons
    
#     CUDA_VISIBLE_DEVICES=3     ./run_wfc.py --model_name_or_path aloxatel/bert-base-mnli --do_train --do_eval --output_dir out_mar16     --top_k 5 --logging_steps 50 --num_train_epochs 3 --warmup_steps 0 --gradient_accumulation_steps 4     --learning_rate 5e-05  --overwrite_output_dir --maxnum 1700 
    
    