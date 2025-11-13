"""
analysis_figures.py
==================

Generate analysis figures:
1. Participant-level AUROC scatter (personalized vs global)
2. Aggregated attention importance (lag heat-bar)
3. UMAP/t-SNE of latent vectors
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap  # Add this line
from sklearn.manifold import TSNE
from umap import UMAP
import warnings
warnings.filterwarnings('ignore')

# Set style (matplotlib)
plt.style.use('seaborn-v0_8-darkgrid')

# -----------------------------------------
def load_personalized_aurocs(pids):
    """Load personalized AUROCs from participant-level auroc_data.pkl"""
    personalized_aurocs = {}
    for pid in pids:
        path = os.path.join('results', 'personalized', f'pid_{pid}', 'auroc_data.pkl')
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                personalized_aurocs[pid] = data.get('personalized_auroc', np.nan)
        except Exception:
            personalized_aurocs[pid] = np.nan
    return personalized_aurocs

# -----------------------------------------
def load_global_aurocs():
    """Load global model AUROCs per participant from global_aurocs.pkl"""
    path = os.path.join('results', 'global', 'global_aurocs.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

# -----------------------------------------
def create_auroc_scatter(personalized_aurocs, global_aurocs, output_dir, filename='auroc_scatter.png'):
    """Figure 1: Participant-level AUROC scatter"""
    pids = sorted(set(personalized_aurocs) & set(global_aurocs))
    pers = np.array([personalized_aurocs[pid] for pid in pids], dtype=float)
    glob = np.array([global_aurocs[pid] for pid in pids], dtype=float)
    mask = ~np.isnan(pers) & ~np.isnan(glob)
    pids, pers, glob = np.array(pids)[mask], pers[mask], glob[mask]

    fig, ax = plt.subplots(figsize=(8,8))
    m, M = min(pers.min(), glob.min()) - 0.02, max(pers.max(), glob.max()) + 0.02
    ax.plot([m, M], [m, M], 'k--', alpha=0.5, label='y=x')
    ax.scatter(pers, glob, s=60, edgecolor='k', alpha=0.7)
    for pid, x, y in zip(pids, pers, glob):
        ax.text(x, y, str(pid), fontsize=8, ha='right', va='bottom')
    ax.set_xlabel('Personalized AUROC')
    ax.set_ylabel('Global AUROC')
    # Set different titles for stress vs BP
    if 'stress' in filename.lower():
        ax.set_title('Participant-level AUROC: Personalized vs Global (Stress Prediction)')
    else:
        ax.set_title('Participant-level AUROC: Personalized vs Global (BP Spike Prediction)')
    ax.set_xlim(m, M)
    ax.set_ylim(m, M)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Counts
    better_p = np.sum(pers > glob)
    better_g = np.sum(glob > pers)
    ax.text(0.05, 0.95,
            f'Personalized better: {better_p}/{len(pids)}\nGlobal better: {better_g}/{len(pids)}',
            transform=ax.transAxes, va='top', bbox=dict(facecolor='white', alpha=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)
    print(f"✅ Created AUROC scatter plot with {len(pids)} participants")

import os
import pickle
import numpy as np

def load_stress_personalized_aurocs(pids):
    """Load personalized AUROCs for stress prediction"""
    personalized_aurocs = {}
    for pid in pids:
        path = os.path.join('results_stress', 'personalized', f'pid_{pid}', 'auroc_data.pkl')
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                personalized_aurocs[pid] = data.get('personalized_auroc', np.nan)
        except Exception:
            personalized_aurocs[pid] = np.nan
    return personalized_aurocs

def load_stress_global_aurocs():
    """Load global model AUROCs per participant for stress prediction"""
    path = os.path.join('results_stress', 'global', 'global_aurocs.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

# -----------------------------------------
def load_attention_weights(pids):
    """Load and aggregate attention weights, handling different dimensions"""
    all_weights = []
    max_dim = 0
    
    # First pass: find maximum dimension
    # personalized
    for pid in pids:
        path = os.path.join('results', 'personalized', f'pid_{pid}', 'analysis_data.pkl')
        try:
            with open(path, 'rb') as f:
                d = pickle.load(f)
                w = d.get('attention_weights')
                if w is not None and w.ndim == 2:
                    max_dim = max(max_dim, w.shape[1])
        except Exception:
            pass
    
    # global
    path = os.path.join('results', 'global', 'global_analysis_data.pkl')
    if not os.path.exists(path):
        path = os.path.join('results', 'global', 'global_analysis.pkl')
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
            g = d.get('global', {})
            w = g.get('attention_weights')
            if w is not None and w.ndim == 2:
                max_dim = max(max_dim, w.shape[1])
    except Exception:
        pass
    
    print(f"Maximum attention dimension found: {max_dim}")
    
    # Second pass: load and pad weights
    # personalized
    for pid in pids:
        path = os.path.join('results', 'personalized', f'pid_{pid}', 'analysis_data.pkl')
        try:
            with open(path, 'rb') as f:
                d = pickle.load(f)
                w = d.get('attention_weights')
                if w is not None and w.ndim == 2:
                    if w.shape[1] < max_dim:
                        # Pad with zeros
                        pad_width = ((0, 0), (0, max_dim - w.shape[1]))
                        w = np.pad(w, pad_width, mode='constant', constant_values=0)
                    all_weights.append(w)
                    print(f"Loaded attention weights for PID {pid}: shape {w.shape}")
        except Exception as e:
            print(f"Failed to load weights for PID {pid}: {e}")
    
    # global
    path = os.path.join('results', 'global', 'global_analysis_data.pkl')
    if not os.path.exists(path):
        path = os.path.join('results', 'global', 'global_analysis.pkl')
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
            g = d.get('global', {})
            w = g.get('attention_weights')
            if w is not None and w.ndim == 2:
                if w.shape[1] < max_dim:
                    pad_width = ((0, 0), (0, max_dim - w.shape[1]))
                    w = np.pad(w, pad_width, mode='constant', constant_values=0)
                all_weights.append(w)
                print(f"Loaded global attention weights: shape {w.shape}")
    except Exception as e:
        print(f"Failed to load global weights: {e}")

    return all_weights, max_dim

# -----------------------------------------
def create_attention_heatbar(all_weights, max_dim, output_dir):
    """Figure 2: Aggregated attention importance with participant-level bootstrapping"""
    if not all_weights:
        print("⚠️ No attention weights found")
        return
    
    # First compute per-participant means
    participant_means = []
    for weights in all_weights:
        p_mean = weights.mean(axis=0)
        participant_means.append(p_mean)
    participant_means = np.array(participant_means)
    
    # Overall mean
    mean_w = participant_means.mean(axis=0)
    
    # Bootstrap at participant level
    B = 1000
    rng = np.random.default_rng(42)
    boots = np.array([
        participant_means[rng.choice(len(participant_means), len(participant_means), replace=True)].mean(axis=0)
        for _ in range(B)
    ])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)

    # Exclude lag-feature indices (previously 44-49) from the plot entirely
    start_lag = 44
    end_lag = min(50, len(mean_w))  # end exclusive
    mask = np.ones(len(mean_w), dtype=bool)
    if start_lag < len(mask):
        mask[start_lag:end_lag] = False

    # Filtered values to plot
    orig_indices = np.arange(len(mean_w))[mask]       # original feature indices kept
    mean_plot = mean_w[mask]
    lo_plot = lo[mask]
    hi_plot = hi[mask]

    # Remap x to consecutive positions to remove empty gap
    x = np.arange(len(mean_plot))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, mean_plot, yerr=[mean_plot - lo_plot, hi_plot - mean_plot], capsize=3, alpha=0.85)
    
    # Color by feature group (approximate), applied to plotted indices
    colors = []
    for idx in orig_indices:
        i = int(idx)
        if i < 10:  # HR features
            colors.append('red')
        elif i < 20:  # Steps features
            colors.append('blue')
        elif i < 30:  # HR (later window)
            colors.append('darkred')
        elif i < 40:  # Steps (later window)
            colors.append('darkblue')
        elif i < 44:  # Stress features
            colors.append('green')
        else:  # Derived / remaining features
            colors.append('purple')
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Feature Index (filtered)')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('Aggregated Attention Weights Across Features (with 95% CI)')
    ax.grid(True, alpha=0.3)

    # Show original feature indices as sparse xtick labels (every 5 or so) to avoid crowding
    if len(orig_indices) <= 20:
        xticks = x
        xlabels = [str(i) for i in orig_indices]
    else:
        step = max(1, len(orig_indices)//12)
        sel = np.arange(0, len(orig_indices), step)
        xticks = x[sel]
        xlabels = [str(orig_indices[i]) for i in sel]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=0)

    # Add legend for feature groups (lag features removed)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='HR (5-10min)'),
        Patch(facecolor='blue', label='Steps (5-10min)'),
        Patch(facecolor='darkred', label='HR (30-60min)'),
        Patch(facecolor='darkblue', label='Steps (30-60min)'),
        Patch(facecolor='green', label='Stress'),
        Patch(facecolor='purple', label='Derived features')
    ]

    # Place legend centered below the plot
    ax.legend(handles=legend_elements,
              bbox_to_anchor=(0.5, -0.18),
              loc='upper center',
              ncol=6,
              borderaxespad=0.)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.20)  # make room for legend
    
    fig.savefig(os.path.join(output_dir, 'attention_heatbar.png'), 
                dpi=300, 
                bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Created attention heatbar with {len(mean_plot)} plotted features")

# -----------------------------------------
def load_latent_vectors(pids, max_samples=20000):
    """Load latent (context) vectors and labels, handling different dimensions"""
    lat_list = []
    lab_list = []
    pid_list = []
    max_dim = 0
    
    # First pass: find maximum dimension
    all_data = []
    
    # Load personalized data
    for pid in pids:
        path = os.path.join('results', 'personalized', f'pid_{pid}', 'analysis_data.pkl')
        try:
            with open(path, 'rb') as f:
                d = pickle.load(f)
                L = d.get('context_vectors', d.get('latent_vectors'))
                Y = d.get('y_test')
                if L is not None and Y is not None:
                    if L.ndim == 3:
                        L = L.mean(axis=1)
                    max_dim = max(max_dim, L.shape[1])
                    all_data.append((L, Y, pid, 'personalized'))
        except Exception as e:
            print(f"Failed to load personalized vectors for PID {pid}: {e}")
    
    # Load global data
    path = os.path.join('results', 'global', 'global_analysis_data.pkl')
    if not os.path.exists(path):
        path = os.path.join('results', 'global', 'global_analysis.pkl')
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
            for pid, dic in d.get('per_pid', {}).items():
                L = dic.get('context_vectors', dic.get('latent_vectors'))
                Y = dic.get('y_test')
                if L is not None and Y is not None:
                    if L.ndim == 3:
                        L = L.mean(axis=1)
                    max_dim = max(max_dim, L.shape[1])
                    all_data.append((L, Y, pid, 'global'))
    except Exception as e:
        print(f"Failed to load global vectors: {e}")
    
    print(f"Maximum vector dimension found: {max_dim}")
    
    # Second pass: pad vectors to max dimension
    for L, Y, pid, source in all_data:
        if L.shape[1] < max_dim:
            # Pad with zeros
            pad_width = ((0, 0), (0, max_dim - L.shape[1]))
            L = np.pad(L, pad_width, mode='constant', constant_values=0)
        lat_list.append(L)
        lab_list.append(Y)
        pid_list.extend([pid] * len(Y))
        print(f"Loaded {len(Y)} {source} vectors for PID {pid} (padded to {max_dim})")
    
    if not lat_list:
        return None, None, None
    
    lat = np.vstack(lat_list)
    lab = np.hstack(lab_list)
    pids = np.array(pid_list)
    
    # subsample if too many
    if len(lat) > max_samples:
        rng = np.random.default_rng(1)
        idx = rng.choice(len(lat), max_samples, replace=False)
        lat = lat[idx]
        lab = lab[idx]
        pids = pids[idx]
    
    print(f"Total vectors loaded: {len(lat)}")
    return lat, lab, pids

# -----------------------------------------
def create_latent_embedding(lat, lab, pids, output_dir):
    """Create a single figure with t-SNE and UMAP embeddings, each with colored and density plots, labeled a/b/c/d."""
    if lat is None:
        print(f"⚠️ No latent vectors found for embedding")
        return

    # Compute embeddings
    reducer_umap = UMAP(n_components=2, random_state=1, n_neighbors=15, min_dist=0.1)
    reducer_tsne = TSNE(n_components=2, random_state=1, perplexity=30)
    emb_umap = reducer_umap.fit_transform(lat)
    emb_tsne = reducer_tsne.fit_transform(lat)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Create custom colormap with just two colors
    colors = ['blue', 'red']
    cmap = ListedColormap(colors)

    # (a) t-SNE colored by BP spike
    sc = axes[0, 0].scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=lab, cmap=cmap, s=5, alpha=0.6)
    axes[0, 0].set_title('a) t-SNE colored by BP spike', loc='center')
    axes[0, 0].set_xlabel('TSNE 1')
    axes[0, 0].set_ylabel('TSNE 2')
    # Create custom legend instead of colorbar
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=c, label=l, markersize=8)
                      for c, l in zip(colors, ['No Spike', 'BP Spike'])]
    axes[0, 0].legend(handles=legend_elements)

    # (b) t-SNE density
    axes[0, 1].hist2d(emb_tsne[:, 0], emb_tsne[:, 1], bins=50, cmap='viridis')
    axes[0, 1].set_title('b) t-SNE density', loc='center')
    axes[0, 1].set_xlabel('TSNE 1')
    axes[0, 1].set_ylabel('TSNE 2')

    # (c) UMAP colored by BP spike
    sc2 = axes[1, 0].scatter(emb_umap[:, 0], emb_umap[:, 1], c=lab, cmap=cmap, s=5, alpha=0.6)
    axes[1, 0].set_title('c) UMAP colored by BP spike', loc='center')
    axes[1, 0].set_xlabel('UMAP 1')
    axes[1, 0].set_ylabel('UMAP 2')
    axes[1, 0].legend(handles=legend_elements)

    # (d) UMAP density
    axes[1, 1].hist2d(emb_umap[:, 0], emb_umap[:, 1], bins=50, cmap='viridis')
    axes[1, 1].set_title('d) UMAP density', loc='center')
    axes[1, 1].set_xlabel('UMAP 1')
    axes[1, 1].set_ylabel('UMAP 2')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'latent_umap_tsne_panel.png'), dpi=300)
    plt.close(fig)
    print(f"✅ Created combined UMAP/t-SNE embedding")

# -----------------------------------------
def create_bp_no_lags_panel(ax, data_table):
    """Create panel for BP prediction without lags using provided table data"""
    # Convert table data to arrays
    pids = []
    pers = []
    glob = []
    
    # Parse the table data line by line
    for line in data_table.strip().split('\n'):
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0].isdigit():  # Skip header
            try:
                pid = int(parts[0])
                pers_val = float(parts[1])  # Just take the first number
                glob_val = float(parts[2])  # Just take the first number
                pids.append(pid)
                pers.append(pers_val)
                glob.append(glob_val)
            except (ValueError, IndexError):
                continue
    
    pids = np.array(pids)
    pers = np.array(pers)
    glob = np.array(glob)
    
    # Create scatter plot
    m, M = min(pers.min(), glob.min()) - 0.02, max(pers.max(), glob.max()) + 0.02
    ax.plot([m, M], [m, M], 'k--', alpha=0.5, label='y=x')
    ax.scatter(pers, glob, s=60, edgecolor='k', alpha=0.7)
    
    for pid, x, y in zip(pids, pers, glob):
        ax.text(x, y, str(pid), fontsize=8, ha='right', va='bottom')
        
    ax.set_xlabel('Personalized AUROC')
    ax.set_ylabel('Global AUROC')
    ax.set_xlim(m, M)
    ax.set_ylim(m, M)
    ax.grid(True, alpha=0.3)
    
    better_p = np.sum(pers > glob)
    better_g = np.sum(glob > pers)
    ax.text(0.05, 0.95,
            f'Personalized better: {better_p}/{len(pids)}\nGlobal better: {better_g}/{len(pids)}',
            transform=ax.transAxes, va='top', 
            bbox=dict(facecolor='white', alpha=0.8))

def create_unified_comparison(pids, output_dir):
    """Create unified figure with all three panels in order: 
    a) BP spike prediction (without BP lags)
    b) Stress prediction
    """
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

    # (a) BP prediction without lags (renumbered from b)
    ax1 = fig.add_subplot(gs[0])
    bp_table = """PID Personalized Global
10 0.824 0.577
15 0.701 0.639
16 0.748 0.657
18 0.828 0.542
20 0.967 0.692
22 0.840 0.607
23 0.841 0.348
24 0.633 0.409
25 0.828 0.808
26 0.671 0.651
30 0.783 0.760
31 0.729 0.523
32 0.538 0.494
33 0.846 0.768
34 0.751 0.614
35 0.801 0.519
36 0.926 0.477
39 1.000 0.508
40 1.000 0.204"""
    
    create_bp_no_lags_panel(ax1, bp_table)
    ax1.set_title('a) BP Spike Prediction', loc='center')

    # (b) Stress prediction (renumbered from c)
    ax2 = fig.add_subplot(gs[1])
    stress_pers = load_stress_personalized_aurocs(pids)
    stress_glob = load_stress_global_aurocs()
    create_auroc_scatter_panel(ax2, stress_pers, stress_glob, 'b) Stress Prediction')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'unified_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✅ Created unified comparison figure")

# Helper function to avoid code duplication
def create_auroc_scatter_panel(ax, personalized_aurocs, global_aurocs, title):
    """Create a single panel for AUROC scatter plot"""
    pids = sorted(set(personalized_aurocs) & set(global_aurocs))
    pers = np.array([personalized_aurocs[pid] for pid in pids], dtype=float)
    glob = np.array([global_aurocs[pid] for pid in pids], dtype=float)
    mask = ~np.isnan(pers) & ~np.isnan(glob)
    pids, pers, glob = np.array(pids)[mask], pers[mask], glob[mask]
    
    m, M = min(pers.min(), glob.min()) - 0.02, max(pers.max(), glob.max()) + 0.02
    ax.plot([m, M], [m, M], 'k--', alpha=0.5, label='y=x')
    ax.scatter(pers, glob, s=60, edgecolor='k', alpha=0.7)
    
    for pid, x, y in zip(pids, pers, glob):
        ax.text(x, y, str(pid), fontsize=8, ha='right', va='bottom')
    
    ax.set_xlabel('Personalized AUROC')
    ax.set_ylabel('Global AUROC')
    ax.set_title(title, loc='center')
    ax.set_xlim(m, M)
    ax.set_ylim(m, M)
    ax.grid(True, alpha=0.3)
    
    better_p = np.sum(pers > glob)
    better_g = np.sum(glob > pers)
    ax.text(0.05, 0.95,
            f'Personalized better: {better_p}/{len(pids)}\nGlobal better: {better_g}/{len(pids)}',
            transform=ax.transAxes, va='top', 
            bbox=dict(facecolor='white', alpha=0.8))

# -----------------------------------------
def main():
    pids = [17, 40, 39, 36, 35, 34, 32, 25, 16, 10, 15, 18, 20, 22, 23, 24, 26, 33, 31, 30]
    out = os.path.join('results', 'analysis_figures')
    os.makedirs(out, exist_ok=True)

    print("=" * 50)
    print("Starting analysis figure generation...")
    print("=" * 50)

    # Fig 1: AUROC scatter
    print("\n📊 Creating AUROC scatter plot...")
    pers = load_personalized_aurocs(pids)
    glob = load_global_aurocs()
    create_auroc_scatter(pers, glob, out)

    # Stress AUROC scatter
    print("\n📊 Creating AUROC scatter plot (Stress)...")
    stress_pers = load_stress_personalized_aurocs(pids)
    stress_glob = load_stress_global_aurocs()
    create_auroc_scatter(stress_pers, stress_glob, out, filename='auroc_scatter_stress.png')

    # Fig 2: Attention heatbar
    print("\n📊 Creating attention heatbar...")
    w, max_dim = load_attention_weights(pids)
    create_attention_heatbar(w, max_dim, out)

    # Fig 3: Embeddings
    print("\n📊 Creating latent embeddings...")
    lat, lab, participant_ids = load_latent_vectors(pids)
    create_latent_embedding(lat, lab, participant_ids, out)

    # Unified comparison figure
    print("\n📊 Creating unified comparison figure...")
    create_unified_comparison(pids, out)  # Pass pids as argument

    print("\n✅ All analysis figures created successfully!")

if __name__ == '__main__':
    main()