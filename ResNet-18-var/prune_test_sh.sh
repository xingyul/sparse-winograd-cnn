./prune_script.py --gpu 0 \
	--existing_mask pruned_model/pruned_0.1/prune_mask_0.1.pkl \
	--output_dir pruned_model_test/ \
	--density 0.1 \
	--original_dir train_log/imagenet-resnet-transWino-prune-0.1 \

	# --density 0.7 \
	# --density 0.6 \
	# --density 0.5 \
	# --density 0.4 \
	# --density 0.35 \

	# --original_dir train_log/imagenet-resnet-transWino-prune-0.7 \
	# --original_dir train_log/imagenet-resnet-transWino-prune-0.6 \
	# --original_dir train_log/imagenet-resnet-transWino-prune-0.5 \
	# --original_dir train_log/imagenet-resnet-transWino-prune-0.4 \
	# --original_dir train_log/imagenet-resnet-transWino-prune-0.35 \

	# --existing_mask pruned_model/pruned_0.7/prune_mask_0.7.pkl \
	# --existing_mask pruned_model/pruned_0.6/prune_mask_0.6.pkl \
	# --existing_mask pruned_model/pruned_0.5/prune_mask_0.5.pkl \
	# --existing_mask pruned_model/pruned_0.4/prune_mask_0.4.pkl \
	# --existing_mask pruned_model/pruned_0.35/prune_mask_0.35.pkl \


	# --output_dir pruned_model \

    # before pruning: 0.3065
