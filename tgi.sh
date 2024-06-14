model=$(pwd)/outputs/output_models_lr_0.000005-v4-all/checkpoint-6652
max_total_tokens=8192
max_input_length=6144
volume=$(pwd)/.tgi-data
token=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
sudo docker run --gpus '"device=3"' --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8081:80 -v $volume:/data -v $model:/model ghcr.io/huggingface/text-generation-inference:1.4 --model-id /model --max-total-tokens $max_total_tokens --max-input-length $max_input_length --max-stop-sequences 10 --max-batch-prefill-tokens $max_input_length