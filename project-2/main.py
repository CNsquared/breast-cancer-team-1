from code.dataloader import load
from code.dnds import dNdS
from code.eval import eval

if __name__ == "__main__":
    datapath = "data/"
    data = load(datapath)
    
    rankings = dNdS(data)
    
    eval(rankings)
    
    