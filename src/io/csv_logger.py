import csv
class CSVLogger:
    def __init__(self, filename):
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(["frame", "x", "y"])
    def log_frame(self, idx, pos):
        self.writer.writerow([idx, pos[0], pos[1]])