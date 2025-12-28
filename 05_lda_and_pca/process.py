import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 入力ファイル
DATA_PATH = "04_preprocess/outputs/data.csv"
CLASS_PATH = "02_classify/output/assets_native_pre2025.csv"
OUTPUT_DIR = "05_lda_and_pca/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. データ読み込み
df = pd.read_csv(DATA_PATH)
df_class = pd.read_csv(CLASS_PATH)

# 2. クラス情報の付与
symbols = df.columns[1:]  # 1列目はdate
symbol2class = dict(zip(df_class['symbol_binance'], df_class['class']))
symbol2name = dict(zip(df_class['symbol_binance'], df_class['name']))
classes = [symbol2class.get(sym, "Unknown") for sym in symbols]
names = [symbol2name.get(sym, sym) for sym in symbols]

# 3. データ整形: (サンプル, 特徴量) = (銘柄, 時系列ベクトル)
X = df[symbols].T.values  # shape: (n_assets, n_days)
y = np.array(classes)
unique_classes = sorted(set(y))
class_to_idx = {c: i for i, c in enumerate(unique_classes)}
y_idx = np.array([class_to_idx[c] for c in y])

# 4. LDA
lda = LDA(n_components=3)
X_lda = lda.fit_transform(X, y_idx)

# 5. PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

def plot_3d_with_gui(X_proj, y, unique_classes, title, axis_labels, filename):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for c in unique_classes:
        idx = y == c
        ax.scatter(X_proj[idx, 0], X_proj[idx, 1], X_proj[idx, 2], label=c, s=30)
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.legend()
    plt.tight_layout()

    # ボタン配置
    ax_button = plt.axes([0.8, 0.01, 0.15, 0.06])
    btn = Button(ax_button, 'Save Figure', color='lightgray', hovercolor='gray')

    def on_save(event):
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
        print(f"Figure saved as {filename}")

    btn.on_clicked(on_save)
    plt.show()
    plt.close(fig)

def plot_basis_subplots_and_save(basis, title, labels, filename):
    n_vec = min(3, basis.shape[0] if basis.ndim == 2 else basis.shape[1])
    fig, axes = plt.subplots(1, n_vec, figsize=(5*n_vec, 4))
    if n_vec == 1:
        axes = [axes]
    for i in range(n_vec):
        ax = axes[i]
        vec = basis[i] if basis.ndim == 2 else basis[:, i]
        ax.plot(vec)
        ax.set_title(labels[i])
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Weight")
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    # LDA 3Dプロット（GUI付き）
    plot_3d_with_gui(
        X_lda, y, unique_classes,
        "LDA: 3D Scatter by Class", ["LD1", "LD2", "LD3"],
        "lda_3d_scatter.png"
    )
    # PCA 3Dプロット（GUI付き, 寄与率付きタイトル）
    pca_ratio = pca.explained_variance_ratio_
    pca_ratio_pct = [f"{v*100:.1f}%" for v in pca_ratio[:3]]
    pca_title = f"PCA: 3D Scatter by Class (PC1: {pca_ratio_pct[0]}, PC2: {pca_ratio_pct[1]}, PC3: {pca_ratio_pct[2]})"
    plot_3d_with_gui(
        X_pca, y, unique_classes,
        pca_title, ["PC1", "PC2", "PC3"],
        "pca_3d_scatter.png"
    )
    # LDA基底ベクトルの可視化（3サブタイルLine Plot）
    plot_basis_subplots_and_save(
        lda.scalings_.T, "LDA Discriminant Vectors (LD1~3)",
        [f"LD{i+1}" for i in range(min(3, lda.scalings_.shape[1]))],
        "lda_basis_lineplot.png"
    )
    # PCA基底ベクトルの可視化（3サブタイルLine Plot, 寄与率付きラベル）
    pca_labels = [f"PC{i+1} ({pca_ratio_pct[i]})" for i in range(min(3, len(pca_ratio_pct)))]
    plot_basis_subplots_and_save(
        pca.components_, "PCA Principal Components (PC1~3)",
        pca_labels,
        "pca_basis_lineplot.png"
    )
    # 全クラス平均＋各クラス平均＋BTC＋ETHの可視化（7サブタイルLine Plot）
    means = []
    labels = ["All Classes Mean"]
    # 全クラス平均
    means.append(np.mean(X, axis=0))
    # 各クラス平均
    for c in unique_classes:
        means.append(np.mean(X[y == c], axis=0))
        labels.append(f"{c} Mean")
    # 全クラス平均＋各クラス平均の可視化（5サブタイルLine Plot, BTC/ETH除去）
    means = []
    labels = ["All Classes Mean"]
    # 全クラス平均
    means.append(np.mean(X, axis=0))
    # 各クラス平均
    for c in unique_classes:
        means.append(np.mean(X[y == c], axis=0))
        labels.append(f"{c} Mean")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        axes[i].plot(means[i])
        axes[i].set_title(labels[i])
        axes[i].set_xlabel("Time Index")
        axes[i].set_ylabel("Value")
    fig.suptitle("Class-wise Mean Time Series")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "class_means_lineplot.png"), dpi=300)
    plt.close(fig)

    # 各クラスから4銘柄ずつ代表を選び、4x4サブプロットで可視化
    reps_per_class = 4
    rep_vectors = []
    rep_titles = []
    for c in unique_classes:
        # クラスcに属する銘柄のsymbolリストとそのインデックス
        class_syms_idx = [(i, sym) for i, (sym, cls) in enumerate(zip(symbols, y)) if cls == c]
        # 分散が大きい順に上位4つ
        if class_syms_idx:
            variances = [(np.var(X[i]), i, sym) for i, sym in class_syms_idx]
            variances.sort(reverse=True)
            for _, i, sym in variances[:reps_per_class]:
                rep_vectors.append(X[i])
                rep_titles.append(f"{c}: {sym}")
    n_reps = len(rep_vectors)
    nrows = 4
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 16))
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i < n_reps:
            ax.plot(rep_vectors[i])
            ax.set_title(rep_titles[i])
        else:
            ax.axis('off')
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Value")
    fig.suptitle("Representative Asset Time Series by Class (4x4 grid, variance top4/class)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "class_representatives_grid.png"), dpi=300)
    plt.close(fig)

    # 各種データ保存
    np.save(os.path.join(OUTPUT_DIR, "lda_basis.npy"), lda.scalings_)
    np.save(os.path.join(OUTPUT_DIR, "pca_basis.npy"), pca.components_)
    np.save(os.path.join(OUTPUT_DIR, "lda_projected.npy"), X_lda)
    np.save(os.path.join(OUTPUT_DIR, "pca_projected.npy"), X_pca)
    np.save(os.path.join(OUTPUT_DIR, "lda_explained_variance_ratio.npy"), getattr(lda, "explained_variance_ratio_", np.zeros(3)))
    np.save(os.path.join(OUTPUT_DIR, "pca_explained_variance_ratio.npy"), pca.explained_variance_ratio_)

    print("LDA/PCA analysis and visualization complete. All plots saved to output directory.")