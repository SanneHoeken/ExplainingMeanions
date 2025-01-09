import numpy as np
from scipy.stats import pearsonr
import json

def subword_to_gold_level(subword_explanation, subwords):
    
    # DATASET SPECIFIC AND SIMPLISTIC RULE-BASED METHOD
    gold_level_explanation = [] 
    word_exps = []

    for i, (subword, exp) in enumerate(zip(subwords, subword_explanation)):

        if subword in ['[CLS]', '[SEP]', '.', ',']:
            continue
        else:
            word_exps.append(exp)
            if not any([subwords[i+1].startswith('##'),
                        subword in ['#', '-'],
                        subwords[i+1] == '-', 
                        subwords[i-1] == '#']):
                gold_level_explanation.append(np.mean(word_exps))
                word_exps = []
            
    return gold_level_explanation


def main(input_dir, output_dir, attribution_type, human_feature):
    
    with open(input_dir+'/Default_Polarity.json', 'r') as infile:
        labels = json.load(infile)
    with open(output_dir+'/subwords.json', 'r') as infile:
        subwords = json.load(infile)
    with open(output_dir+f'/{attribution_type}.json', 'r') as infile:
        model_attributions = json.load(infile)
    with open(input_dir+f'/{human_feature}.json', 'r') as infile:
        human_features = json.load(infile)

    assert len(labels) == len(subwords) == len(model_attributions) == len(human_features)

    correlations = []
    for i, cls in enumerate(labels):    
        human_feat = human_features[i]
        model_att = subword_to_gold_level(model_attributions[i][str(cls)], subwords[i])
        assert len(human_feat) == len(model_att)
        r, p = pearsonr(human_feat, model_att)
        correlations.append(r)

    print('Mean correlation:', np.nanmean(correlations))


if __name__ == '__main__':

    input_dir = '../data/test_0_prepro_lists'
    output_dir = '../output/test_0'
    attribution_type = 'input_x_gradient'
    human_feature = 'AFP'
    main(input_dir, output_dir, attribution_type, human_feature)