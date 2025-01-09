from captum.attr import ShapleyValueSampling, InputXGradient, Saliency
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, json
from tqdm import tqdm

# inspired by: XAI Benchmark (https://github.com/copenlu/xai-benchmark) and Captum Tutorials (https://captum.ai/tutorials/)
# this code only works for BERT classification models

class ShapleyModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ShapleyModelWrapper, self).__init__()
        self.model = model

    def forward(self, input, mask):
        return self.model(input, attention_mask=mask)[0]

class GradientModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(GradientModelWrapper, self).__init__()
        self.model = model

    def forward(self, input, mask):
        return self.model(inputs_embeds=input, attention_mask=mask)[0]


def explain_input(encodings, classes, model, ablator_name, ablator, device):
    
    # prepare inputs
    masks = torch.tensor([[int(i > 0) for i in encodings]], device=device)
    input = torch.tensor([encodings], dtype=torch.long, device=device)
      
    if ablator_name != 'shapley_value':  
      input = model.model.bert.embeddings(input)

    # get attributions for every class
    cls2attributions = dict()
    for cls in classes:
        attributions = ablator.attribute(input, target=cls, additional_forward_args=masks)[0]
        if ablator_name != 'shapley_value':
            # l2 summarization
            attributions = attributions.norm(p=1, dim=-1).squeeze(0)
            
        cls2attributions[cls] = attributions.tolist()
    
    return cls2attributions

    

def main(input_dir, output_dir, model_dir, tokenizer_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get input
    with open(input_dir+'/Cleaned_Text.json', 'r') as infile:
        texts = json.load(infile)

    # prepare tokenizer and encode input
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    encodings = [tokenizer.encode(t) for t in texts]
    subwords = [tokenizer.convert_ids_to_tokens(e) for e in encodings]
    with open(output_dir+'/subwords.json', 'w') as outfile:
        json.dump(subwords, outfile)
    
    # load classification model
    clf_model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device) 
    classes = list(clf_model.config.id2label.keys())

    # get predictions (optional)
    tokenized_input = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    output = clf_model(input_ids=tokenized_input['input_ids'], attention_mask=tokenized_input['attention_mask'])
    preds = torch.argmax(output[0], dim=1).tolist()
    with open(output_dir+'/predictions.json', 'w') as outfile:
        json.dump(preds, outfile)


    """
    config = BertConfig.from_json_file(config_file)
    model = Summarizer(args, device, load_pretrained_bert=True, bert_config=config)
    sent_scores, mask = model(input_ids, segment_ids, clss_ids, input_mask, clss_mask,
                                ffds=ffds,
                                gds=gds,
                                gpts=gpts,
                                trts=trts,
                                nfixs=nfixs)
    """
    
    # compute attributions
    for ablator_name in ['input_x_gradient', 'saliency', 'shapley_value']:

        if ablator_name == 'shapley_value':
            wrapped_model = ShapleyModelWrapper(clf_model)
            ablator = ShapleyValueSampling(wrapped_model)
        else:
            wrapped_model = GradientModelWrapper(clf_model)
            if ablator_name == 'input_x_gradient':
                ablator = InputXGradient(wrapped_model)
            elif ablator_name == 'saliency':
                ablator = Saliency(wrapped_model)

        cls2attributions = []
        for encoding in tqdm(encodings):
            cls2attributions.append(explain_input(encoding, classes, wrapped_model, ablator_name, ablator, device))

        # export output
        with open(output_dir+f'/{ablator_name}.json', 'w') as outfile:
            json.dump(cls2attributions, outfile)

if __name__ == '__main__':

    input_dir = '../data/test_0_prepro_lists'
    output_dir = '../output/test_0'
    model_dir = '../models/org_gaze4hate_fold_0'
    tokenizer_dir = 'dbmdz/bert-base-german-uncased'
    
    main(input_dir, output_dir, model_dir, tokenizer_dir)