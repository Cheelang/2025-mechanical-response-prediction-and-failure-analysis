import os
import pandas as pd


def get_filenames_from_directory(directory):

    filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return filenames


def save_filenames_to_csv(filenames, csv_path):

    df = pd.DataFrame(filenames, columns=['Filename'])

    df.to_csv(csv_path, index=False)


if __name__ == "__main__":

    directory_path = r""

    csv_file_path = 'output_filenames.csv'


    filenames = get_filenames_from_directory(directory_path)

    save_filenames_to_csv(filenames, csv_file_path)

    print(f"Save to {csv_file_path}")