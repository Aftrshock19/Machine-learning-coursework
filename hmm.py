
import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM
import pymc as pm
import matplotlib.pyplot as plt




def load_and_discretise(temp_bins=3, death_bins=3):

    csv_path = pm.get_data("deaths_and_temps_england_wales.csv")
    df = pd.read_csv(csv_path)


    temp_col = "temp"    
    death_col = "deaths"

    if temp_col not in df.columns:
        raise ValueError(f"Temperature column '{temp_col}' not found in CSV columns: {df.columns}")
    if death_col not in df.columns:
        raise ValueError(f"Deaths column '{death_col}' not found in CSV columns: {df.columns}")

    df = df.dropna(subset=[temp_col, death_col])
    df["temp_state"] = pd.qcut(
        df[temp_col],
        q=temp_bins,
        labels=False,
        duplicates="drop"
    )

    df["death_state"] = pd.qcut(
        df[death_col],
        q=death_bins,
        labels=False,
        duplicates="drop"
    )

    temp_states = df["temp_state"].to_numpy(dtype=int)
    death_states = df["death_state"].to_numpy(dtype=int)

    return temp_states, death_states, df

 

def train_supervised_hmm(
    temp_states,
    death_states,
    n_temp_states,
    n_death_levels,
    laplace_smoothing=1.0,
):
    T = len(temp_states)
    assert T == len(death_states)
 
    pi_counts = np.zeros(n_temp_states)
    first_state = temp_states[0]
    pi_counts[first_state] += 1

    pi = pi_counts + laplace_smoothing
    pi /= pi.sum()
 
    trans_counts = np.zeros((n_temp_states, n_temp_states))
    for t in range(T - 1):
        i = temp_states[t]
        j = temp_states[t + 1]
        trans_counts[i, j] += 1

    A = trans_counts + laplace_smoothing
    A /= A.sum(axis=1, keepdims=True)
 
    emit_counts = np.zeros((n_temp_states, n_death_levels))
    for s, o in zip(temp_states, death_states):
        emit_counts[s, o] += 1

    B = emit_counts + laplace_smoothing
    B /= B.sum(axis=1, keepdims=True)
 
    hmm1 = CategoricalHMM( n_components=n_temp_states,  n_features=n_death_levels, init_params="", n_iter=1,  )

    hmm1.startprob_ = pi
    hmm1.transmat_ = A
    hmm1.emissionprob_ = B

    return hmm1, pi, A, B


 

def train_unsupervised_hmm(  death_states,  n_hidden_states, n_death_levels,  n_iter=100, random_state=0,):
    T = len(death_states)
    X = death_states.reshape(-1, 1)
    lengths = [T]

    hmm2 = CategoricalHMM(  n_components=n_hidden_states, n_features=n_death_levels,  n_iter=n_iter, random_state=random_state,
    )

    hmm2.fit(X, lengths)

    return hmm2

def sample_death_sequences(hmm_model, n_sequences=10, seq_length=120):
    samples = []
    for _ in range(n_sequences):
        X, Z = hmm_model.sample(seq_length)
        samples.append(X.reshape(-1))
    return samples

def compute_freqs(states, n_levels):
    counts = np.bincount(states, minlength=n_levels)
    return counts / counts.sum()

if __name__ == "__main__":
    temp_states, death_states, df = load_and_discretise(
        temp_bins=3,
        death_bins=3,
    )

    print("T =", len(temp_states))
    print("Unique temp states:", np.unique(temp_states))
    print("Unique death states:", np.unique(death_states))

    n_temp_states = len(np.unique(temp_states))
    n_death_levels = len(np.unique(death_states))

    hmm1, pi, A, B = train_supervised_hmm(
        temp_states=temp_states,
        death_states=death_states,
        n_temp_states=n_temp_states,
        n_death_levels=n_death_levels,
    )
    print("\nHMM1 parameters")
    print("startprob_:\n", hmm1.startprob_)
    print("transmat_:\n", hmm1.transmat_)
    print("emissionprob_:\n", hmm1.emissionprob_)

    hmm2 = train_unsupervised_hmm(
        death_states=death_states,
        n_hidden_states=n_temp_states, 
        n_death_levels=n_death_levels,
    )
    print("\nHMM2 parameters")
    print("startprob_:\n", hmm2.startprob_)
    print("transmat_:\n", hmm2.transmat_)
    print("emissionprob_:\n", hmm2.emissionprob_)

    
    samples_hmm1 = sample_death_sequences(hmm1, n_sequences=3, seq_length=120)
    samples_hmm2 = sample_death_sequences(hmm2, n_sequences=3, seq_length=120)

    print("\nExample sampled sequence from HMM1:", samples_hmm1[0][:30])
    print("Example sampled sequence from HMM2:", samples_hmm2[0][:30])

    real_freqs = compute_freqs(death_states, n_death_levels)

    flat_hmm1 = np.concatenate(samples_hmm1)
    flat_hmm2 = np.concatenate(samples_hmm2)

    hmm1_freqs = compute_freqs(flat_hmm1, n_death_levels)
    hmm2_freqs = compute_freqs(flat_hmm2, n_death_levels)

    print("\nEmpirical frequencies (low, med, high):")
    print("Real: ", real_freqs)
    print("HMM1:", hmm1_freqs)
    print("HMM2:", hmm2_freqs)


    real_segment = death_states[:120]
    hmm1_segment = samples_hmm1[0][:120]
    hmm2_segment = samples_hmm2[0][:120]
    
    months = np.arange(120)

    dist_hmm1 = np.abs(hmm1_segment - real_segment)
    dist_hmm2 = np.abs(hmm2_segment - real_segment)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]

    ax.step(months, real_segment, where="mid",
            color="0.3", linewidth=2.0, label="Real deaths")

    for d, color, label in [
        (0, "tab:green", "HMM1: exact match"),
        (1, "tab:orange", "HMM1: off by 1"),
        (2, "tab:red", "HMM1: off by 2"),
    ]:
        mask = dist_hmm1 == d
        if mask.any():
            ax.scatter(months[mask], hmm1_segment[mask],
                    s=30, color=color, alpha=0.9, label=label)

    ax.set_ylabel("Death category")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Low", "Medium", "High"])
    ax.set_title("HMM1 vs real deaths (closeness per month)")
    ax.grid(alpha=0.3, linestyle="--")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
            ncol=1, fontsize=8,
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.)

    ax = axes[1]

    ax.step(months, real_segment, where="mid",
            color="0.3", linewidth=2.0, label="Real deaths")

    for d, color, label in [
        (0, "tab:green", "HMM2: exact match"),
        (1, "tab:orange", "HMM2: off by 1"),
        (2, "tab:red", "HMM2: off by 2"),
    ]:
        mask = dist_hmm2 == d
        if mask.any():
            ax.scatter(months[mask], hmm2_segment[mask],
                    s=30, color=color, alpha=0.9, label=label)

    ax.set_xlabel("Time (months, t)")
    ax.set_ylabel("Death category")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Low", "Medium", "High"])
    ax.set_title("HMM2 vs real deaths (closeness per month)")
    ax.grid(alpha=0.3, linestyle="--")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
            ncol=1, fontsize=8,
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.)

    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    plt.savefig("hmm_scatter_closeness.png", dpi=150)
    plt.show()


    # 1) Real deaths only
    plt.figure(figsize=(8, 3))
    plt.step(months, real_segment, where="mid", linewidth=2.0, label="Real deaths")
    plt.xlabel("Time (months, t)")
    plt.ylabel("Death category")
    plt.yticks([0, 1, 2], ["Low", "Medium", "High"])
    plt.title("Real death categories (first 120 months)")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hmm_real_only.png", dpi=150)
    plt.show()

    # 2) Real vs HMM1
    plt.figure(figsize=(8, 3))
    plt.step(months, real_segment, where="mid", linewidth=2.0, label="Real deaths")
    plt.step(months, hmm1_segment, where="mid", linewidth=1.5, linestyle="--", label="HMM1 sample")
    plt.xlabel("Time (months, t)")
    plt.ylabel("Death category")
    plt.yticks([0, 1, 2], ["Low", "Medium", "High"])
    plt.title("Real vs HMM1 death categories (first 120 months)")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hmm_real_vs_hmm1.png", dpi=150)
    plt.show()

    # 3) Real vs HMM2
    plt.figure(figsize=(8, 3))
    plt.step(months, real_segment, where="mid", linewidth=2.0, label="Real deaths")
    plt.step(months, hmm2_segment, where="mid", linewidth=1.5, linestyle="--", label="HMM2 sample")
    plt.xlabel("Time (months, t)")
    plt.ylabel("Death category")
    plt.yticks([0, 1, 2], ["Low", "Medium", "High"])
    plt.title("Real vs HMM2 death categories (first 120 months)")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hmm_real_vs_hmm2.png", dpi=150)
    plt.show()







        