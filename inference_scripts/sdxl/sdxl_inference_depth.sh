python inference.py \
--model_name "sdxl" \
--control_types "depth" \
--huggingface_checkpoint_folder "sdxl_depth" \
--eval_input_type "images" \
--evaluation_input_folder "assets/evaluation/images" \
--num_inference_steps 50 \
--control_guidance_end 0.6 \
--height 1024 \
--width 1024 
