import fastparquet
import pandas as pd
import sys

# Create a ParquetFile object
#parquet_file = fastparquet.ParquetFile("../datasets/SimpleRL-Zoo-Data/simplelr_qwen_level1to4/test.parquet")
parquet_file = fastparquet.ParquetFile("../datasets/SimpleRL-Zoo-Data/simplelr_qwen_level1to4/train.parquet")
#parquet_file = fastparquet.ParquetFile("../datasets/MATH-lighteval/data/train-00000-of-00001.parquet")

data = parquet_file.to_pandas()
print(data.columns.tolist())

print(data['gt_answer'][:5])
print(data['answer'][:5])
print(data['subject'][:5])
print(data['level'][:5])
print(data['question'][:5])
print(data['target'][:5])
print(data['data_source'][:5])
print(data['prompt'][:5])
print(data['ability'][:5])

sys.exit()
# Iterate over the row groups
batch_size=10
for df in parquet_file.iter_row_groups(batch_size):
    a += 1
    #print(df)
    # Read the row group into a Pandas DataFrame
    #df = parquet_file.read_row_group(row_group_number)

    #print(df)
    sys.exit()
    # Modify the DataFrame as needed
    # Example: add a new column
    #df["new_column"] = df["existing_column"] * 2

    # Write the modified DataFrame to a new Parquet file
    #with fastparquet.ParquetWriter("test.parquet", df, row_group_size=parquet_file.row_group_size) as writer:
    #    writer.write_table(df)
