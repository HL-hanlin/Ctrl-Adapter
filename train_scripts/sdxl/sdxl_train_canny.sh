accelerate launch train.py \
--yaml_file configs/sdxl_train_canny.yaml \
--evaluation_input_folder "assets/evaluation/images" \
--num_inference_steps 50 \
--control_guidance_end 0.8 \
--snr_gamma 5.0 \
--save_n_steps 1000 \
--validate_every_steps 1000 \
--save_starting_step 1000
