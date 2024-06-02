import os

if __name__ == "__main__":
    _ = os.popen("python trainAutoencoder.py").read()
    _ = os.popen("python trainDiscriminator.py").read()
    _ = os.popen("python trainRMSSD.py").read()
    _ = os.popen("python buildModel.py").read()
    