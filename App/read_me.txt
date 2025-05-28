build the image:
    sudo docker build -t rp_bot .


run the image and keep the model in cach :
    sudo docker run \
      --env-file /home/simon/Desktop/AI-chat-bot/.env \
      -v $(pwd)/input:/input \
      -v $(pwd)/output:/output \
      -v /home/simon/.cache/huggingface:/root/.cache/huggingface \
      rp_bot

builder :

sudo docker rm -f rp_bot_container
sudo docker build -t rp_bot .
sudo docker run --gpus all -d \
  -p 8000:8000 \
  --name rp_bot_container \
  --env-file /home/simon/Desktop/AI-chat-bot/.env \
  rp_bot