import pandas as pd

def load_data():
    print("Loading big and small matrices...")

    small_matrix = pd.read_csv("data_final_project/KuaiRec 2.0/data/small_matrix.csv")
    big_matrix = pd.read_csv("data_final_project/KuaiRec 2.0/data/big_matrix.csv")

    print("Loading social network...")
    social_network = pd.read_csv("data_final_project/KuaiRec 2.0/data/social_network.csv")
    social_network["friend_list"] = social_network["friend_list"].map(eval)

    print("Loading item features...")
    item_categories = pd.read_csv("data_final_project/KuaiRec 2.0/data/item_categories.csv")
    item_categories["feat"] = item_categories["feat"].map(eval)

    print("Loading user features...")
    user_features = pd.read_csv("data_final_project/KuaiRec 2.0/data/user_features.csv")

    print("Loading items' daily features...")
    item_daily_features = pd.read_csv("data_final_project/KuaiRec 2.0/data/item_daily_features.csv")

    print("All data loaded.")
    return big_matrix, small_matrix, social_network, item_categories, user_features, item_daily_features

def clean_data(big_matrix, small_matrix, social_network, item_categories, user_features, item_daily_features):
    print("Cleaning data...")
    # Remove NA values
    big_matrix.dropna(inplace=True)
    small_matrix.dropna(inplace=True)
    social_network.dropna(inplace=True)
    item_categories.dropna(inplace=True)
    user_features.dropna(inplace=True)
    item_daily_features.dropna(inplace=True)

    # Remove duplicates
    big_matrix.drop_duplicates(inplace=True)
    small_matrix.drop_duplicates(inplace=True)
    social_network['friend_list'] = social_network['friend_list'].apply(frozenset)
    social_network.drop_duplicates(inplace=True)
    item_categories['feat'] = item_categories['feat'].apply(frozenset)
    item_categories.drop_duplicates(inplace=True)
    user_features.drop_duplicates(inplace=True)
    item_daily_features.drop_duplicates(inplace=True)

    # Reset index
    big_matrix.reset_index(drop=True, inplace=True)
    small_matrix.reset_index(drop=True, inplace=True)
    social_network.reset_index(drop=True, inplace=True)
    item_categories.reset_index(drop=True, inplace=True)
    user_features.reset_index(drop=True, inplace=True)
    item_daily_features.reset_index(drop=True, inplace=True)

    print("Data cleaned.")

def load_and_clean_data():
    # Load data, clean it and print the percentages of cleaned data
    big_matrix, small_matrix, social_network, item_categories, user_features, item_daily_features = load_data()
    big_matrix_size, small_matrix_size, social_network_size, item_categories_size, user_features_size, item_daily_features_size = (
        len(big_matrix), len(small_matrix), len(social_network), len(item_categories), len(user_features), len(item_daily_features)
    )
    clean_data(big_matrix, small_matrix, social_network, item_categories, user_features, item_daily_features)
    # Print the percentage of cleaned data
    print(f"Big matrix: {100 - (len(big_matrix) / big_matrix_size * 100):.2f}% cleaned")
    print(f"Small matrix: {100 - (len(small_matrix) / small_matrix_size * 100):.2f}% cleaned")
    print(f"Social network: {100 - (len(social_network) / social_network_size * 100):.2f}% cleaned")
    print(f"Item categories: {100 - (len(item_categories) / item_categories_size * 100):.2f}% cleaned")
    print(f"User features: {100 - (len(user_features) / user_features_size * 100):.2f}% cleaned")
    print(f"Item daily features: {100 - (len(item_daily_features) / item_daily_features_size * 100):.2f}% cleaned")
    
    return big_matrix, small_matrix, social_network, item_categories, user_features, item_daily_features