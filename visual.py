import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def plot(df) :
    plt.figure(figsize=(6, 6))
    # Sets up a plot with a size of 6x6 inches
    sns.histplot(df['Temperature (K)'], bins=30, kde=True)
    #Plots a histogram of the Temperature (K) column with 30 bins and a kernel density estimate (smooth curve).
    plt.title('Frequency Distribution of Temperature (K)')
    #Adds a title to the plot.
    plt.xlabel('Temperature (K)')
    # Labels the x-axis.
    plt.ylabel('Frequency')
    #Labels the y-axis
    plt.show()

def MaxScale(df) :
    df['Temperature (K)'] = np.log(df['Temperature (K)'])
    df['Luminosity(L/Lo)'] = np.log(df['Luminosity(L/Lo)'])
    df['Radius(R/Ro)'] = np.log(df['Radius(R/Ro)'])

    # define the scaler
    scaler = MinMaxScaler()

    # variables now scaled with the minmax scaler
    df['Temperature (K)'] = scaler.fit_transform(np.expand_dims(df['Temperature (K)'], axis=1))
    df['Luminosity(L/Lo)'] = scaler.fit_transform(np.expand_dims(df['Luminosity(L/Lo)'], axis=1))
    df['Radius(R/Ro)'] = scaler.fit_transform(np.expand_dims(df['Radius(R/Ro)'], axis=1))
    df['Absolute magnitude(Mv)'] = scaler.fit_transform(np.expand_dims(df['Absolute magnitude(Mv)'], axis=1))
    return df

def plot_all(df) :
    features = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)',]
    # List of features to plot

    # Number of subplots
    n_features = len(features)

    # Calculate the number of rows and columns for subplots
    n_rows = (n_features + 1) // 2
    n_cols = 2

    # Create a figure with a specified size
    plt.figure(figsize=(12, 6 * n_rows))

    # Iterate over the features and plot each one
    for i, feature in enumerate(features):
        # Create subplot for each feature
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(df[feature], bins=30, edgecolor='k')
        plt.title(f'Frequency Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')


    # Adjust the layout to prevent overlapping of subplots
    plt.tight_layout()

    # Display the plots
    return plt.show()

def plot_output(df) :
    plt.figure(figsize=(6, 6))
    # Sets up a plot with a size of 6x6 inches
    sns.histplot(df['Spectral Class'], bins=30, kde=True)
    #Plots a histogram of the Temperature (K) column with 30 bins and a kernel density estimate (smooth curve).
    plt.title('Frequency Distribution of Spectral Class ')
    #Adds a title to the plot.
    plt.xlabel('Spectral Class')
    # Labels the x-axis.
    plt.ylabel('Frequency')
    #Labels the y-axis
    plt.show()