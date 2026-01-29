git clone https://github.com/LUOyk1999/tunedGNN.git
cp download_data.py tunedGNN/medium_graph
cp run_download.sh tunedGNN/medium_graph
cd tunedGNN/medium_graph
bash run_download.sh
cp -r save_dataset_with_splits ../../
