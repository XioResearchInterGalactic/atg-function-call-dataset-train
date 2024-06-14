model=MerlynMind/Merlyn-reformulation_v2-llama2-7b
volume=$(pwd)/.tgi-data-reformulator
token=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
sudo docker run -d --gpus '"device=3"' --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8082:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.4 --model-id $model --max-total-tokens 4096 --max-input-length 2048