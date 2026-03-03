import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from PIL import Image
OUTPUT_DIR = "outputs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_training_curves(train_history):
    # Loss plot
    plt.figure()
    plt.plot(train_history['loss'], label='Train Loss')
    plt.plot(train_history['val_loss'], label='Validation Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    loss_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.savefig(loss_path)
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(train_history['accuracy'], label='Train Acc')
    plt.plot(train_history['val_accuracy'], label='Validation Acc')
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    acc_path = os.path.join(OUTPUT_DIR, "accuracy_curve.png")
    plt.savefig(acc_path)
    plt.close()

def evaluate_model(model, test_generator):
    test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")