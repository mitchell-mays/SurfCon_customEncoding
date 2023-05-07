import pandas as pd
import pickle
import numpy as np
term_2_concept_df = pd.read_csv('../data/mappings/3_term_ID_to_concept_ID.txt', delim_whitespace=True,header=None)
term_2_concept_df.columns = ['term','concept']
#sample for testing
#term_2_concept_df = term_2_concept_df.head(1000)
terms = term_2_concept_df['term'].unique()
num_terms = len(terms)
print("Total terms : ", num_terms)
term_2_concept_dict = {}
i = 0
for term in terms:
    i = i + 1
    term_concepts = list(term_2_concept_df[term_2_concept_df['term'] == term]['concept'].to_numpy())
    term_2_concept_dict[term] = term_concepts
    if  i % 50000 == 0:
        print(f"Processed {i} terms")

print("Created the dict ! ")
#print("Sample Output:")
#print(term_2_concept_dict)

print("Writing to pickle file")
#pickle.dump(term_2_concept_dict,open('../data/sym_data/sample_term_2_concept.pkl', "wb"), protocol=-1)
pickle.dump(term_2_concept_dict,open('../data/sym_data/term_concept_mapping.pkl', "wb"), protocol=-1)
print('File write complete !')

