./prune_script.py --gpu 0 \
	--original_dir pruned_model_test/pruned_0.12/ \
	--output_dir pruned_model/ \
    --density 0.1 \
	--existing_mask pruned_model/pruned_0.12/prune_mask_0.12.pkl

	# --original_dir pruned_model_test/pruned_0.7/ \
	# --original_dir pruned_model_test/pruned_0.6/ \
	# --original_dir pruned_model_test/pruned_0.5/ \
	# --original_dir pruned_model_test/pruned_0.4/ \

	# --density 0.7 \
	# --density 0.6 \
	# --density 0.5 \
	# --density 0.4 \
	# --density 0.35 \

	# --existing_mask pruned_model/pruned_0.7/prune_mask_0.7.pkl
	# --existing_mask pruned_model/pruned_0.6/prune_mask_0.6.pkl
	# --existing_mask pruned_model/pruned_0.5/prune_mask_0.5.pkl
	# --existing_mask pruned_model/pruned_0.4/prune_mask_0.4.pkl

	# create pruned_$density dir in output_dir and put pruned model in pruned_$density dir
