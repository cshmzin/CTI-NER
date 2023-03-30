import json


class Load_dict():
    def __init__(self):
        self.path = '../Ner_data/Ner_data1757'
        self.attackers = []
        self.malwares = []

    def save(self):
        with open('dicts/attackers.txt','w',encoding='utf-8') as f:
            for a in list(set(self.attackers)):
                f.write(a+'\n')

        with open('dicts/malwares.txt','w',encoding='utf-8') as f:
            for m in list(set(self.malwares)):
                f.write(m+'\n')



    def load_data(self):
        for i in range(1,1758):
            with open(f'{self.path}/{i}.json','r',encoding='utf-8') as f:
                file = json.load(f)
            records = file["records"]
            for r in records:
                if r["tag"]=="malware":
                    self.malwares.append(r["span"])
                elif r["tag"]=="attacker":
                    self.attackers.append(r["span"])

if __name__ == '__main__':
    Load = Load_dict()
    Load.load_data()
    Load.save()