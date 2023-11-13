from model import Model
from utils import load_txt, split_data, get_dataframe

data_filepath = './data/human.txt'
data_column_1 = 'sequence'
data_column_2 = 'class'

sequence_data, class_data = load_txt(data_filepath, data_column_1, data_column_2)
sequence_data_train, sequence_data_test, class_data_train, class_data_test = split_data(sequence_data, class_data)

kmer = 12
model = Model(kmer)

# Fit the vectorizer on the training data
# Transform both the training and test data using the same vectorizer
sequence_data_train_vec = model.vectorizer.fit_transform(sequence_data_train)
sequence_data_test_vec = model.vectorizer.transform(sequence_data_test)

model.train(sequence_data_train_vec, class_data_train)
model.test(sequence_data_test_vec, class_data_test)

print(model.accuracy)


input_sequence = ["ATGCGATCGATCGATC", "ATGATGATGATGATG","ATGCCCCAACTAA","TTCGCTTCATTCGCTGCCCCCAC","CAGCTGGCTCCCAGGGTT"]
input_sequence_vec = model.vectorizer.transform(input_sequence)
pred_class = model.get_prediction(input_sequence_vec)

result_df = get_dataframe(input_sequence,pred_class)
result_df.to_csv('./results/result.csv',index=True)
print(result_df)
