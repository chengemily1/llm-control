# Layers: 1 through 40
for layer in {1..40}; do
    python train_probe.py --layer $layer
done
