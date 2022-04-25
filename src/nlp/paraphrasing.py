import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

num_beams = 10
num_return_sequences = 10

def get_response(input_text,num_return_sequences,num_beams):
    batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

def paraphrase_sentence(sentence):
    context = [sentence]
    outputs_ = []
    for j in context:
        u = get_response(j, num_return_sequences, num_beams)
        for i in u:
            i = i.replace('slut','person')
            i = i.replace('whore','person')
            i = i.replace('faggot','person')
            i = i.replace('dyke','person')
            outputs_.append(i)
    # print(outputs_)
    return outputs_
    

if __name__ == "__main__":
    # paraphrase_sentence("you are a fucking slut! Go suck some dick!")
    input_ = ["you are a fucking slut! Go suck some dick!", "faggots in this webspace are over-smart"]
    context = input_
    for j in context:
        u = get_response(j, num_return_sequences, num_beams)
        for i in u:
            i = i.replace('slut','person')
            i = i.replace('whore','person')
            i = i.replace('faggot','homosexual')
            i = i.replace('dyke','person')
            print(i)
        print("**************************************")