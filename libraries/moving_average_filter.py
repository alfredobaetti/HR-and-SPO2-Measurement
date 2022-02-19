
##-- Moving Average Filter --##

class MovingAverageFilter:
    
    def __init__(self):
        self.MAF_level = 8
        self.data = [0 for i in range(8)]
        print(self.data)

    def start(self, newData):
        self.data.pop()
        self.data.insert(0, newData)
        return (sum(self.data)/len(self.data))

