# Dynamic Directed Transfer Function（dDTF）

Here’s an English summary of the visualized **dDTF analysis** and the example provided:

---

### 📊 Visualization Overview (from the generated plots):

1. **Connectivity Strength (Condition 1)**

   * This heatmap shows the average **directed coupling strength** from one channel to another (i.e., how strongly channel *j* influences channel *i*) based on the dDTF values in the alpha band (8–12 Hz).
   * Higher values = stronger directed influence.

2. **Directionality Index**

   * Computed as:

     $$
     \text{Directionality}_{ij} = \frac{dDTF_{ij} - dDTF_{ji}}{dDTF_{ij} + dDTF_{ji}}
     $$
   * Values near **+1** indicate dominant flow from channel *j → i*,
     near **–1** means dominant *i → j*,
     near **0** means bidirectional or no clear dominance.

3. **P-values (Condition 1 vs Condition 2)**

   * Paired **t-tests** were used to compare dDTF values between two simulated conditions across subjects.
   * This matrix shows where connectivity differs significantly.
   * **Red areas** highlight lower p-values (more statistically significant differences).

---

### 🔧 How to Use for Your Own EEG Data

To adapt this example for your own EEG time series:

1. **Input shape**: Your data should be shaped like:

   ```
   data[subject_index, channel_index, time]
   ```
2. **Frequency band**: Change `freqs = np.linspace(...)` to analyze other bands (e.g., beta 13–30 Hz).
3. **Sliding window** (optional): You can extend the code to apply dDTF in sliding time windows to visualize time-varying connectivity.
