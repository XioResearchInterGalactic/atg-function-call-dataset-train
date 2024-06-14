sudo docker kill $(docker ps -q --filter ancestor=ghcr.io/huggingface/text-generation-inference:1.4)
sudo rm -r .tgi-data
./tgi.sh