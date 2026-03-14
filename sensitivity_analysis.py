from lz76 import LZ76

# calculate entropy rate for a binary string

def calc_er(X):
    lz = LZ76(X) # Compute Lempel-Ziv complexity
    er = lz*np.log2(len(X))/len(X) # Normalize using log2(length)
    return er

# computes the average entropy rate over non-overlapping windows of a binary sequence X

def get_er_windows(X,window_size):
    er_vals = [] # List to store entropy rates of each window
    for i in range(0,len(X),window_size): # Iterate over X in steps of window_size
        er_vals.append(calc_er(X[i:i+window_size])) # Compute entropy rate for each window
    return np.mean(er_vals) # Return the average entropy rate

X = np.random.randint(0,2,100000) # Generates a random binary sequence of size 100000
win_sizes = np.arange(100,1500,100) # Define window sizes from 100 to 1400 in steps of 100
er_vals = [get_er_windows(X,win_size) for win_size in win_sizes] # Compute entropy rate for each window size
plt.plot(win_sizes,er_vals) # Plot window size vs. entropy rate

plt.xlabel('Window size')
plt.ylabel('Entropy rate')
plt.title('Arbitrary example')

# As the window size increases, the entropy rate stabilizes
# Larger window size > less surprise

# Purely random X should have ER = 1
# Structured sequence's ER would decrease for larger windows