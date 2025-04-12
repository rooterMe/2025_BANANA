import os
import matplotlib.pyplot as plt

def plot_freshness_from_labels(label_folder):
    x_vals = []
    y_vals = []

    txt_files = sorted(os.listdir(label_folder), key=lambda x: int(os.path.splitext(x)[0]))

    for file in txt_files:
        file_path = os.path.join(label_folder, file)
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
            if len(lines) >= 2:
                x = int(lines[0])
                y = float(lines[1])
                x_vals.append(x)
                y_vals.append(y)

    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, y_vals, linestyle='-', color='blue', label='Freshness Ratio')
    plt.title("Time vs Banana Freshness")
    plt.xlabel("Time (Image Index)")
    plt.ylabel("Freshness Ratio")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 실행
plot_freshness_from_labels("Dataset/banana_0_label")
