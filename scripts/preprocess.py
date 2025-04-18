class Preprocess:

    def __init__(self, data):
        self.data = data
        self.data = self.clean_data(self.data)
        self.data = self.normalize(self.data)

    def clean_data(self):
        # Implement data cleaning logic here
        pass

    def normalize(self):
        # Implement TPM
        pass