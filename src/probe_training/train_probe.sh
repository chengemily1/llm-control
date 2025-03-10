# Layers: 1 through 40
for layer in {1..40}; do
    echo "Training layer $layer"
    python train_probe.py --layer $layer --num_epochs 3000 --save 1
done
