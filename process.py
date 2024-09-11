from category_encoders.one_hot import OneHotEncoder
# Initialize the OneHotEncoder to transform the 'Star color' column
# 'use_cat_names=True' ensures that the new columns are named with the original category names
def map(data) :
    data['Spectral Class'] = data['Spectral Class'].map({'M': 0, 'K': 1, 'G': 2, 'F': 3, 'A': 4, 'B': 5, 'O': 6})
    return data

def oneHot(df) :
    from category_encoders.one_hot import OneHotEncoder
    one_hot = OneHotEncoder(cols=['Star color'], use_cat_names=True)
    df = one_hot.fit_transform(df)
    return df
    # Fit the OneHotEncoder on the DataFrame and transform the 'Star color' column

def correlation(df) :
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create a heatmap to visualize the correlation matrix of the dataset
    # This helps to understand the relationships between different features
    plt.figure(figsize=(8, 8))
    # Generate a heatmap with correlation coefficients
    # Rounded to 2 decimal places, using the 'coolwarm' color map,
    # and displaying the correlation values within the cells
    ax = sns.heatmap(data=df.corr().round(1), cmap='coolwarm', annot=True)
    ax.set_title('Correlation', fontsize=18)
    plt.show()