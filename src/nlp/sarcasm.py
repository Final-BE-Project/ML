import torch

model_save_name = 'sarcasm.pt'
path = F"src/models/{model_save_name}"

model = ""

def initiate_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(torch.cuda.get_device_name())
    global model
    model = torch.load(path, map_location=torch.device("cuda"))

def cuda_info():
    print(torch.cuda.is_available())
    print(f"Torch Version: {torch.__version__}")

def run_sample():
    samples = ["I totally understand man! 😏",'I am going to kill you man 🤣','I will kill you nigga',"Ohhhh she's like...soooo pretty🤣","they atleast don 't make you feel socially unaccepted # hahahaha # lol # truestory " ]

    predictions, _ = model.predict(samples) 
    label_dict = {0: 'not sarcastic', 1: 'sarcastic'}

    ff = ['🙂','😂','🤣','😌','😒','😏','💩','🤭','👀','🍌','🍆','🍒','🍑','🚩','🌝','🌚']
    for idx, sample in enumerate(samples):
        f=0
        for j in ff:
            if j in sample:
                print('{}: {} ===> {}'.format(idx, sample, 'sarcastic'))
                f=1
        if f==0:
            print('{}: {} ===> {}'.format(idx, sample, label_dict[predictions[idx]]))
            
    return predictions

def run_input(sentence):
    predictions, _ = model.predict([sentence]) 
    label_dict = {0: 'not sarcastic', 1: 'sarcastic'}
    output_ = ""

    ff = ['🙂','😂','🤣','😌','😒','😏','💩','🤭','👀','🍌','🍆','🍒','🍑','🚩','🌝','🌚']
    for idx, sample in enumerate([sentence]):
        f=0
        for j in ff:
            if j in sample:
                print('{}: {} ===> {}'.format(idx, sample, 'sarcastic'))
                output_ = "sarcastic"
                f=1
        if f==0:
            print('{}: {} ===> {}'.format(idx, sample, label_dict[predictions[idx]]))
            output_ = label_dict[predictions[idx]]

        return output_


def detect_sarcasm(sentence):
    initiate_model()
    cuda_info()
    output_ = run_input(sentence)
    print(f"Output of the Sarcasm detection is: {output_}")


if __name__ == "__main__":
    initiate_model()
    cuda_info()
    print(run_sample())
    # print(run_input("I will kill you nigga"))
